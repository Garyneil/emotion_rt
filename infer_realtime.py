import time
import json
import yaml
import argparse
import threading
import asyncio
from typing import Optional, Callable, Dict, Any

import numpy as np
import onnxruntime as ort

from rt.sources import build_source_auto
from rt.autodetect import AutoState, pick_notch_freq
from rt.preprocess import preprocess_eeg, preprocess_ecg
from rt.labels import decode_label  # 保留

# -------------------------
# Config
# -------------------------
def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -------------------------
# Utils
# -------------------------
class RingBuffer:
    def __init__(self, max_len, ch):
        self.max_len = int(max_len)
        self.ch = int(ch)
        self.x = np.zeros((self.max_len, self.ch), dtype=np.float32)
        self.t = np.zeros((self.max_len,), dtype=np.float64)
        self.n = 0

    def push(self, t, x):
        if x is None or t is None:
            return
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]

        # if [C,N] and C small, transpose -> [N,C]
        if x.ndim == 2 and x.shape[0] < x.shape[1] and x.shape[0] <= 512:
            x = x.T

        # === HARD ALIGN to buffer channels self.ch (pad/crop) ===
        if x.ndim == 2:
            T, C = x.shape
            if C > self.ch:
                x = x[:, :self.ch]
            elif C < self.ch:
                pad = np.zeros((T, self.ch - C), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)

        # 允许 t 是标量：补成与 x 同长度
        if t.ndim == 0:
            t = np.full((x.shape[0],), float(t), dtype=np.float64)
        elif t.ndim == 1 and len(t) != x.shape[0]:
            # 极端情况：时间戳长度对不上 -> 用 arrival time 兜底
            t = np.linspace(time.time() - 0.001 * x.shape[0], time.time(), x.shape[0], dtype=np.float64)

        for i in range(len(t)):
            idx = self.n % self.max_len
            self.t[idx] = t[i]
            self.x[idx] = x[i]  # now guaranteed shape == (self.ch,)
            self.n += 1

    def get_last(self, length):
        length = int(length)
        if self.n < length:
            return None, None
        end = self.n % self.max_len
        if self.n >= self.max_len:
            x = np.concatenate([self.x[end:], self.x[:end]], axis=0)
            t = np.concatenate([self.t[end:], self.t[:end]], axis=0)
        else:
            x = self.x[:self.n]
            t = self.t[:self.n]
        return t[-length:], x[-length:]


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def decode_va_semantic_from_idx(idx: int):
    idx = int(idx)
    vh = (idx >> 1) & 1
    ah = idx & 1
    valence = "High" if vh == 1 else "Low"
    arousal = "High" if ah == 1 else "Low"

    if vh == 0 and ah == 0:
        semantic = "Calm / Bored"
    elif vh == 0 and ah == 1:
        semantic = "Stress / Anxiety"
    elif vh == 1 and ah == 0:
        semantic = "Relaxed / Content"
    else:
        semantic = "Excited / Happy"

    return valence, arousal, semantic


def pick_providers():
    """
    Jetson Nano / PC：只用当前可用 providers
    Nano 上一般是 CPUExecutionProvider（除非你装的是带 CUDA 的 onnxruntime-gpu）
    """
    avail = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in avail]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return providers


def _as_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            return None
        return int(x)
    except Exception:
        return None


def infer_expected_channels_from_onnx(sess: ort.InferenceSession) -> Dict[str, Optional[int]]:
    exp: Dict[str, Optional[int]] = {}
    for inp in sess.get_inputs():
        shape = inp.shape  # e.g. [1, 14, 'T']
        c = None
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            c = _as_int(shape[1])
        exp[inp.name] = c
    return exp


def ensure_channels(x: Optional[np.ndarray], c_expected: Optional[int]) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]

    # 尝试转成 [T,C]
    if x.ndim == 2 and x.shape[0] < x.shape[1] and x.shape[0] <= 512:
        x = x.T

    if c_expected is None:
        return x

    T, C = x.shape
    if C == c_expected:
        return x
    if C > c_expected:
        return x[:, :c_expected]

    pad = np.zeros((T, c_expected - C), dtype=x.dtype)
    return np.concatenate([x, pad], axis=1)


def to_model_tensor_tc(x_tc: Optional[np.ndarray], c_expected: Optional[int], fallback_T: int = 1) -> np.ndarray:
    x_tc = ensure_channels(x_tc, c_expected)
    if x_tc is None or x_tc.size == 0:
        C = c_expected if (c_expected is not None and c_expected > 0) else 1
        return np.zeros((1, C, max(1, fallback_T)), dtype=np.float32)

    # [T,C] -> [1,C,T]
    x_in = np.transpose(x_tc[None, ...], (0, 2, 1)).astype(np.float32)
    return x_in


# -------------------------
# Runtime config override for REAL SOURCE (Nano)
# -------------------------
def apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    让你在 Nano 上可以用启动参数覆盖 config.yaml，避免频繁改配置文件
    """
    cfg = dict(cfg)
    if "runtime" not in cfg:
        cfg["runtime"] = {}
    rt = dict(cfg["runtime"])

    # 默认策略：Nano 部署优先真实源 udp（除非你显式指定 sim）
    # 如果 config.yaml 已经写了 source，就尊重 config；否则默认 udp。
    if "source" not in rt:
        rt["source"] = "udp"

    # CLI 覆盖
    if getattr(args, "source", None):
        rt["source"] = args.source

    if getattr(args, "udp_host", None):
        rt["udp_host"] = args.udp_host
    if getattr(args, "udp_port", None) is not None:
        rt["udp_port"] = int(args.udp_port)

    # 真实数据常见：发送端不带 fs / channels，这里给兜底值（若 chunk 自带会覆盖）
    if getattr(args, "fs_eeg", None) is not None:
        rt["fs_eeg"] = float(args.fs_eeg)
    if getattr(args, "fs_ecg", None) is not None:
        rt["fs_ecg"] = float(args.fs_ecg)
    if getattr(args, "eeg_channels", None) is not None:
        rt["eeg_channels"] = int(args.eeg_channels)

    # window/hop 可用 CLI 调
    if getattr(args, "window_sec", None) is not None:
        rt["window_sec"] = float(args.window_sec)
    if getattr(args, "hop_sec", None) is not None:
        rt["hop_sec"] = float(args.hop_sec)

    cfg["runtime"] = rt
    return cfg


# -------------------------
# Core realtime inference loop
# -------------------------
def run_realtime_inference(
    cfg_path: str = "config.yaml",
    model_path: str = "emotion_model.onnx",
    on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_event: Optional[threading.Event] = None,
    cfg_override: Optional[Dict[str, Any]] = None,
):
    cfg = load_cfg(cfg_path)
    if cfg_override is not None:
        # 只覆盖 runtime，其他保持 yaml 原样（preprocess/labels 等）
        cfg["runtime"] = cfg_override.get("runtime", cfg.get("runtime", {}))

    rt_cfg = cfg["runtime"]
    pp = cfg["preprocess"]

    # === source ===
    source = build_source_auto(rt_cfg)
    auto = AutoState()

    providers = pick_providers()
    sess = ort.InferenceSession(model_path, providers=providers)

    # === 从模型里自动推断 EEG/ECG 期望通道数 ===
    model_inputs = [i.name for i in sess.get_inputs()]
    expected_C = infer_expected_channels_from_onnx(sess)

    def _pick_name(keys, token: str) -> Optional[str]:
        for k in keys:
            if token.lower() in k.lower():
                return k
        return None

    eeg_name = _pick_name(model_inputs, "eeg") or ("eeg" if "eeg" in model_inputs else model_inputs[0])
    if len(model_inputs) > 1:
        ecg_name = _pick_name(model_inputs, "ecg") or ("ecg" if "ecg" in model_inputs else model_inputs[1])
    else:
        ecg_name = _pick_name(model_inputs, "ecg") or model_inputs[0]

    EEG_EXPECTED = expected_C.get(eeg_name, None)
    ECG_EXPECTED = expected_C.get(ecg_name, None)

    eeg_buf_ch = EEG_EXPECTED if (EEG_EXPECTED is not None and EEG_EXPECTED > 0) else 64
    ecg_buf_ch = ECG_EXPECTED if (ECG_EXPECTED is not None and ECG_EXPECTED > 0) else 4

    eeg_buf = RingBuffer(max_len=1024 * 8, ch=eeg_buf_ch)
    ecg_buf = RingBuffer(max_len=1024 * 8, ch=ecg_buf_ch)

    print(f"[Model] inputs={model_inputs} | eeg_name={eeg_name} C={EEG_EXPECTED} | ecg_name={ecg_name} C={ECG_EXPECTED}")
    print(f"[Runtime] source={rt_cfg.get('source')} udp={rt_cfg.get('udp_host')}:{rt_cfg.get('udp_port')} "
          f"window={rt_cfg.get('window_sec')} hop={rt_cfg.get('hop_sec')}")

    last_infer = 0.0

    # 真实源常见：短时间没数据，别让线程空转刷屏
    last_chunk_at = time.time()

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        chunk = source.read_chunk()
        if chunk is None:
            # 如果超过 2 秒没数据，轻提示一次（不刷屏）
            now = time.time()
            if now - last_chunk_at > 2.0:
                last_chunk_at = now
                print("[SRC] waiting for real data... (no chunk)")
            time.sleep(0.002)
            continue

        last_chunk_at = time.time()
        auto.update_from_chunk(chunk)

        # --- fs/channel fallback：真实 UDP 很可能不带这些字段 ---
        if auto.fs_eeg is None:
            v = chunk.get("fs_eeg", rt_cfg.get("fs_eeg"))
            if v is not None:
                auto.fs_eeg = float(v)

        if auto.fs_ecg is None:
            v = chunk.get("fs_ecg", rt_cfg.get("fs_ecg"))
            if v is not None:
                auto.fs_ecg = float(v)

        if auto.eeg_channels is None:
            v = chunk.get("eeg_channels", rt_cfg.get("eeg_channels"))
            if v is not None:
                auto.eeg_channels = int(v)

        # debug：低频输出
        _dbg = getattr(run_realtime_inference, "_dbg", 0) + 1
        run_realtime_inference._dbg = _dbg
        if _dbg % 250 == 0:
            print("[DBG] fs_eeg/fs_ecg:", auto.fs_eeg, auto.fs_ecg, "eeg_channels:", auto.eeg_channels,
                  "chunk keys:", list(chunk.keys()))

        fs_eeg = auto.fs_eeg
        fs_ecg = auto.fs_ecg

        eeg = chunk.get("eeg")
        ecg = chunk.get("ecg")
        t = chunk.get("t") if chunk.get("t") is not None else chunk.get("arrival_t")

        # === 转为 [T,C] 并对齐通道到模型期望 ===
        if eeg is not None:
            eeg = np.asarray(eeg, dtype=np.float32)
            if eeg.ndim == 1:
                eeg = eeg[:, None]
            eeg = ensure_channels(eeg, EEG_EXPECTED)
            if eeg is not None and eeg_buf.ch != eeg.shape[1]:
                eeg_buf = RingBuffer(max_len=eeg_buf.max_len, ch=eeg.shape[1])

        if ecg is not None:
            ecg = np.asarray(ecg, dtype=np.float32)
            if ecg.ndim == 1:
                ecg = ecg[:, None]
            ecg = ensure_channels(ecg, ECG_EXPECTED)
            if ecg is not None and ecg_buf.ch != ecg.shape[1]:
                ecg_buf = RingBuffer(max_len=ecg_buf.max_len, ch=ecg.shape[1])

        eeg_buf.push(t, eeg)
        ecg_buf.push(t, ecg)

        if fs_eeg is None and fs_ecg is None:
            continue

        now = time.time()
        if now - last_infer < float(rt_cfg["hop_sec"]):
            continue
        last_infer = now

        win_sec = float(rt_cfg["window_sec"])

        eeg_win = None
        ecg_win = None
        if fs_eeg:
            L = int(round(win_sec * fs_eeg))
            _, eeg_win = eeg_buf.get_last(L)
        if fs_ecg:
            L2 = int(round(win_sec * fs_ecg))
            _, ecg_win = ecg_buf.get_last(L2)

        if eeg_win is None and ecg_win is None:
            continue

        notch = pp["eeg"].get("notch", None)
        if notch == "auto" and fs_eeg:
            notch = pick_notch_freq(fs_eeg)

        eeg_p = preprocess_eeg(
            eeg_win,
            fs_eeg,
            bandpass=tuple(pp["eeg"]["bandpass"]),
            notch=notch if isinstance(notch, (int, float)) else None,
            normalize=pp["eeg"]["normalize"],
        ) if eeg_win is not None else None

        ecg_p = preprocess_ecg(
            ecg_win,
            fs_ecg,
            bandpass=tuple(pp["ecg"]["bandpass"]),
            normalize=pp["ecg"]["normalize"],
        ) if ecg_win is not None else None

        inputs: Dict[str, np.ndarray] = {}
        inputs[eeg_name] = to_model_tensor_tc(eeg_p, EEG_EXPECTED, fallback_T=1)
        inputs[ecg_name] = to_model_tensor_tc(ecg_p, ECG_EXPECTED, fallback_T=1)

        t0 = time.time()
        try:
            logits = sess.run(None, inputs)[0]  # [1,4]
        except Exception as e:
            shapes = {k: tuple(v.shape) for k, v in inputs.items()}
            print("[ERR] onnxruntime failed:", repr(e))
            print("[ERR] input shapes:", shapes)
            continue
        infer_cost = time.time() - t0

        prob = softmax(logits[0])
        pred_idx = int(np.argmax(prob))
        conf = float(prob[pred_idx])

        label_str = decode_label(prob, cfg["labels"]) if "labels" in cfg else str(pred_idx)
        valence, arousal, semantic = decode_va_semantic_from_idx(pred_idx)

        payload = {
            "label_idx": pred_idx,
            "label": label_str,
            "valence": valence,
            "arousal": arousal,
            "semantic": semantic,
            "confidence": conf,
            "pmax": float(prob.max()),
            "timestamp": time.time(),
            "infer_cost": float(infer_cost),
            "fs_eeg": fs_eeg,
            "fs_ecg": fs_ecg,
            "eeg_ch": int(inputs[eeg_name].shape[1]),
            "ecg_ch": int(inputs[ecg_name].shape[1]),
            "providers": providers,
            "source": rt_cfg.get("source", "auto"),
        }

        print(
            f"[RT] V={valence} A={arousal} | {semantic} | "
            f"conf={conf:.3f} infer={infer_cost*1000:.1f}ms "
            f"fs_eeg={fs_eeg} fs_ecg={fs_ecg} eegC={payload['eeg_ch']} ecgC={payload['ecg_ch']}"
        )

        if on_result is not None:
            try:
                on_result(payload)
            except Exception:
                pass


# -------------------------
# Dashboard (FastAPI + WebSocket) in SAME FILE
# -------------------------
DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Emotion Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 16px; padding: 16px; max-width: 720px; }
    .row { display:flex; justify-content:space-between; align-items:center; padding: 10px 0; border-bottom: 1px solid #f1f1f1; gap: 12px; }
    .row:last-child { border-bottom: none; }
    .k { color:#666; min-width: 110px; }
    .v { font-weight:800; text-align:right; flex: 1; }
    .big { font-size: 30px; }
    .sub { color:#777; font-size: 13px; font-weight: 600; }
    .pill { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #ddd; font-size:12px; font-weight: 700; }
    .pill.online { border-color:#b7e4c7; }
    .pill.offline { border-color:#f6c177; }
    .pill.stale { border-color:#ffd166; }
    .pill.error { border-color:#ef476f; }
    .barWrap { width: 260px; height: 12px; border-radius: 999px; border: 1px solid #ddd; overflow:hidden; margin-left: auto; }
    .bar { height: 100%; width: 0%; background: #111; transition: width 120ms linear; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  </style>
</head>
<body>
  <h2>Real-time Emotion (EEG + ECG)</h2>
  <div class="card">
    <div class="row"><div class="k">Valence</div><div class="v big" id="valence">—</div></div>
    <div class="row"><div class="k">Arousal</div><div class="v big" id="arousal">—</div></div>
    <div class="row"><div class="k">Semantic</div><div class="v" id="semantic">Waiting...</div></div>

    <div class="row">
      <div class="k">Confidence</div>
      <div class="v">
        <span class="mono" id="conf">0.000</span>
        <div class="barWrap" title="confidence bar">
          <div class="bar" id="confBar"></div>
        </div>
      </div>
    </div>

    <div class="row"><div class="k">Label</div><div class="v mono" id="label">—</div></div>
    <div class="row"><div class="k">Updated</div><div class="v mono" id="ts">—</div></div>

    <div class="row"><div class="k">Meta</div><div class="v sub" id="meta">—</div></div>
    <div class="row"><div class="k">Providers</div><div class="v sub" id="prov">—</div></div>

    <div class="row"><div class="k">Latency</div><div class="v mono" id="lat">—</div></div>
    <div class="row"><div class="k">Infer(ms)</div><div class="v mono" id="infer">—</div></div>

    <div class="row">
      <div class="k">Status</div>
      <div class="v"><span class="pill offline" id="status">offline</span></div>
    </div>
  </div>

  <script>
    const STALE_SEC = 2.0;
    const PING_SEC  = 15.0;

    function fmtTime(t) {
      if (!t) return "—";
      const d = new Date(t * 1000);
      return d.toLocaleTimeString();
    }
    function clamp01(x){ return Math.max(0, Math.min(1, x)); }

    const statusEl = document.getElementById("status");
    const confEl   = document.getElementById("conf");
    const confBar  = document.getElementById("confBar");

    let ws = null;
    let retry = 0;
    let lastMsgAt = 0;

    function setStatus(text, cls) {
      statusEl.textContent = text;
      statusEl.className = "pill " + cls;
    }

    function connect() {
      const url = `ws://${location.host}/ws`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        retry = 0;
        setStatus("online", "online");
        try { ws.send("hi"); } catch(e){}
      };

      ws.onclose = () => {
        setStatus("offline", "offline");
        scheduleReconnect();
      };

      ws.onerror = () => {
        setStatus("error", "error");
        try { ws.close(); } catch(e){}
      };

      ws.onmessage = (ev) => {
        lastMsgAt = Date.now();
        const x = JSON.parse(ev.data);

        document.getElementById("valence").textContent  = x.valence ?? "—";
        document.getElementById("arousal").textContent  = x.arousal ?? "—";
        document.getElementById("semantic").textContent = x.semantic ?? "—";

        const c = Number(x.confidence ?? 0);
        confEl.textContent = c.toFixed(3);
        confBar.style.width = (clamp01(c) * 100).toFixed(1) + "%";

        document.getElementById("label").textContent = x.label ?? x.label_idx ?? "—";
        document.getElementById("ts").textContent    = fmtTime(x.timestamp);

        document.getElementById("meta").textContent =
          `EEG: ${x.eeg_ch ?? "?"}ch@${x.fs_eeg ?? "?"}Hz | ECG: ${x.ecg_ch ?? "?"}ch@${x.fs_ecg ?? "?"}Hz | SRC: ${x.source ?? "?"}`;

        document.getElementById("prov").textContent = (x.providers ?? []).join(", ");

        if (x.timestamp) {
          const lat = (Date.now()/1000 - Number(x.timestamp));
          document.getElementById("lat").textContent = lat.toFixed(3) + " s";
        } else {
          document.getElementById("lat").textContent = "—";
        }

        const inferMs = Number(x.infer_cost ?? 0) * 1000;
        document.getElementById("infer").textContent = inferMs > 0 ? inferMs.toFixed(1) : "—";

        setStatus("online", "online");
      };
    }

    function scheduleReconnect() {
      retry += 1;
      const delay = Math.min(8000, 250 * Math.pow(2, retry));
      setTimeout(() => {
        try { connect(); } catch(e) { scheduleReconnect(); }
      }, delay);
    }

    setInterval(() => {
      if (!ws) return;
      const now = Date.now();
      if (ws.readyState === WebSocket.OPEN && lastMsgAt > 0) {
        const dt = (now - lastMsgAt) / 1000;
        if (dt > STALE_SEC) setStatus("stale", "stale");
      }
    }, 200);

    setInterval(() => {
      if (!ws) return;
      if (ws.readyState === WebSocket.OPEN) {
        try { ws.send("ping"); } catch(e){}
      }
    }, PING_SEC * 1000);

    connect();
  </script>
</body>
</html>
"""


def run_dashboard(host="0.0.0.0", port=8000, cfg_path="config.yaml", model_path="emotion_model.onnx", cfg_override=None):
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    import uvicorn

    app = FastAPI()
    latest: Dict[str, Any] = {
        "valence": "—",
        "arousal": "—",
        "semantic": "Waiting for signal...",
        "confidence": 0.0,
        "label": "—",
        "label_idx": None,
        "timestamp": 0.0,
        "infer_cost": 0.0,
        "fs_eeg": None,
        "fs_ecg": None,
        "eeg_ch": None,
        "ecg_ch": None,
        "providers": [],
        "source": "auto",
    }

    latest_lock = threading.Lock()
    clients = set()
    stop_event = threading.Event()

    def on_result(payload: Dict[str, Any]):
        with latest_lock:
            latest.update(payload)

    async def broadcaster():
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
                with latest_lock:
                    msg = json.dumps(latest, ensure_ascii=False)

                dead = []
                for ws in list(clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    clients.discard(ws)
        except asyncio.CancelledError:
            pass

    @app.on_event("startup")
    async def startup_event():
        def _infer():
            run_realtime_inference(
                cfg_path=cfg_path,
                model_path=model_path,
                on_result=on_result,
                stop_event=stop_event,
                cfg_override=cfg_override,
            )

        th = threading.Thread(target=_infer, daemon=True)
        th.start()
        asyncio.create_task(broadcaster())

    @app.on_event("shutdown")
    async def shutdown_event():
        stop_event.set()

    @app.get("/")
    def index():
        return HTMLResponse(DASHBOARD_HTML)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except Exception:
            pass
        finally:
            clients.discard(ws)

    uvicorn.run(app, host=host, port=port, log_level="info")


# -------------------------
# Entry
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="config.yaml")
    p.add_argument("--model", default="emotion_model.onnx")
    p.add_argument("--dashboard", action="store_true", help="start web dashboard (FastAPI+WebSocket)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    # 真实源参数（Nano 部署推荐走 UDP）
    p.add_argument("--source", default=None, help="override runtime.source (udp/sim/auto/...)")
    p.add_argument("--udp_host", default=None, help="override runtime.udp_host (bind host)")
    p.add_argument("--udp_port", type=int, default=None, help="override runtime.udp_port")

    # 真实源可能不带 fs/ch：给兜底
    p.add_argument("--fs_eeg", type=float, default=None, help="fallback fs_eeg if chunk has no fs_eeg")
    p.add_argument("--fs_ecg", type=float, default=None, help="fallback fs_ecg if chunk has no fs_ecg")
    p.add_argument("--eeg_channels", type=int, default=None, help="fallback eeg_channels if chunk has no eeg_channels")

    # 可调窗口与步长（部署常用）
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)

    args = p.parse_args()

    cfg0 = load_cfg(args.cfg)
    cfg1 = apply_runtime_overrides(cfg0, args)

    # 只传 runtime 覆盖（preprocess/labels 仍走 yaml）
    cfg_override = {"runtime": cfg1.get("runtime", {})}

    if args.dashboard:
        run_dashboard(host=args.host, port=args.port, cfg_path=args.cfg, model_path=args.model, cfg_override=cfg_override)
    else:
        run_realtime_inference(cfg_path=args.cfg, model_path=args.model, cfg_override=cfg_override)


if __name__ == "__main__":
    main()

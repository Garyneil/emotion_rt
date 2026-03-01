import json
import time
import socket
import numpy as np

class BaseSource:
    name = "base"
    def probe(self) -> bool:
        return False
    def read_chunk(self) -> dict:
        raise NotImplementedError

class LSLSource(BaseSource):
    name = "lsl"
    def __init__(self, stream_name=None):
        self.stream_name = stream_name
        self.inlet = None

    def probe(self) -> bool:
        try:
            from pylsl import resolve_stream, StreamInlet
            streams = resolve_stream('type', 'EEG', timeout=0.5)
            if not streams:
                return False
            self.inlet = StreamInlet(streams[0])
            return True
        except Exception:
            return False

    def read_chunk(self) -> dict:
        t_list, x_list, arrival = [], [], []
        t0 = time.time()
        for _ in range(64):
            sample, ts = self.inlet.pull_sample(timeout=0.01)
            if sample is None:
                break
            t_list.append(ts)
            x_list.append(sample)
            arrival.append(time.time() - t0 + t0)
        eeg = np.asarray(x_list, dtype=np.float32) if x_list else None
        if eeg is not None and eeg.ndim == 2:
            return {"t": np.asarray(t_list, dtype=np.float64),
                    "eeg": eeg, "ecg": None,
                    "arrival_t": np.asarray(arrival, dtype=np.float64)}
        return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

class UDPSource(BaseSource):
    name = "udp"
    def __init__(self, host="0.0.0.0", port=9000, buf=65535):
        self.host, self.port, self.buf = host, port, buf
        self.sock = None

    def probe(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.2)
            return True
        except Exception:
            return False

    def read_chunk(self) -> dict:
        try:
            data, _ = self.sock.recvfrom(self.buf)
            now = time.time()
            obj = json.loads(data.decode("utf-8"))
            eeg = np.asarray(obj.get("eeg"), dtype=np.float32) if obj.get("eeg") is not None else None
            ecg = np.asarray(obj.get("ecg"), dtype=np.float32) if obj.get("ecg") is not None else None
            t = np.asarray(obj.get("t"), dtype=np.float64) if obj.get("t") is not None else None
            arrival_t = np.full((len(t),), now, dtype=np.float64) if t is not None else None
            return {"t": t, "eeg": eeg, "ecg": ecg, "arrival_t": arrival_t}
        except Exception:
            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

class SerialSource(BaseSource):
    name = "serial"
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.port, self.baud = port, baud
        self.ser = None

    def probe(self) -> bool:
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2)
            return True
        except Exception:
            return False

    def read_chunk(self) -> dict:
        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}
            obj = json.loads(line)
            now = time.time()
            t = np.asarray([obj.get("t", now)], dtype=np.float64)
            eeg = np.asarray([obj["eeg"]], dtype=np.float32) if "eeg" in obj else None
            ecg = np.asarray([obj["ecg"]], dtype=np.float32) if "ecg" in obj else None
            arrival_t = np.asarray([now], dtype=np.float64)
            return {"t": t, "eeg": eeg, "ecg": ecg, "arrival_t": arrival_t}
        except Exception:
            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

# def build_source_auto(cfg: dict) -> BaseSource:
#     source_name = cfg.get("source", "auto")
#     source_args = cfg.get("source_args", {}) or {}
#
#     # ===== 1️⃣ 显式指定 source（不走 auto）=====
#     if source_name == "serial12hex":
#         print("[Source] using Serial12HexSource")
#         return Serial12HexSource(**source_args)
#
#     if source_name == "udp":
#         print("[Source] using UDPSource")
#         return UDPSource(**source_args)
#
#     if source_name == "serial":
#         print("[Source] using SerialSource")
#         return SerialSource(**source_args)
#
#     if source_name == "lsl":
#         print("[Source] using LSLSource")
#         return LSLSource()
#
#     # ===== 2️ auto 探测模式=====
#     candidates = [
#         LSLSource(),
#         UDPSource(**source_args),
#         SerialSource(**source_args),
#     ]
#
#     for s in candidates:
#         if s.probe():
#             print(f"[AutoSource] selected: {s.name}")
#             return s
#
#     raise RuntimeError(
#         "No available source found. "
#         "Please specify runtime.source and runtime.source_args."
#     )

#测试
def build_source_auto(cfg: dict) -> BaseSource:
    source_name = cfg.get("source", "auto")
    source_args = cfg.get("source_args", {}) or {}

    # ✅ 手动指定：sim
    if source_name == "sim":
        s = SimSource(**source_args)
        print(f"[Source] selected: {s.name}")
        return s

    # （可选）你之前做的 serial12hex 也建议在这儿手动分支
    # if source_name == "serial12hex":
    #     s = Serial12HexSource(**source_args)
    #     print(f"[Source] selected: {s.name}")
    #     return s

    # ✅ 保留原 auto 探测逻辑
    candidates = [
        LSLSource(),
        UDPSource(**source_args) if source_args else UDPSource(),
        SerialSource(**source_args) if source_args else SerialSource(),
    ]
    for s in candidates:
        if s.probe():
            print(f"[AutoSource] selected: {s.name}")
            return s
    raise RuntimeError("No available source found (lsl/udp/serial). Provide runtime.source + source_args.")


class Serial12HexSource:
    def __init__(
        self,
        port="/dev/ttyUSB0",
        baudrate=115200,
        head=0xFF,
        tail=0xFE,
        fs=250.0,
        scale_eeg=1.0,
        scale_ecg=1.0,
        timeout=0.2,
    ):
        import serial
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.head = head & 0xFF
        self.tail = tail & 0xFF
        self.fs = float(fs)
        self.scale_eeg = float(scale_eeg)
        self.scale_ecg = float(scale_ecg)

    def _sync_to_head(self):
        while True:
            b = self.ser.read(1)
            if not b:
                return False
            if b[0] == self.head:
                return True

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def read_chunk(self):
        if not self._sync_to_head():
            return None

        payload = self._read_exact(13)  # 12 data + 1 tail
        if len(payload) != 13:
            return None

        data = payload[:12]
        tail = payload[12]
        if tail != self.tail:
            return None

        vals = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

        eeg = (vals[:8] * self.scale_eeg)[None, :]
        ecg = (vals[8:] * self.scale_ecg)[None, :]
        t = np.array([time.time()], dtype=np.float64)

        return {
            "t": t,
            "eeg": eeg,
            "ecg": ecg,
            "fs_eeg": self.fs,
            "fs_ecg": self.fs,
            "eeg_channels": 8,
        }

#测试
class SimSource(BaseSource):
    name = "sim"

    def __init__(self, fs=250.0, eeg_channels=8, ecg_channels=4, noise=5.0, seed=0):
        self.fs = float(fs)
        self.eeg_channels = int(eeg_channels)
        self.ecg_channels = int(ecg_channels)
        self.noise = float(noise)
        self.rng = np.random.default_rng(int(seed))
        self._t0 = time.time()
        self._n = 0

    def probe(self) -> bool:
        # 模拟源永远可用
        return True

    def read_chunk(self):
        # 每次吐 1 个采样点（[1, C]），和你串口源一致
        t = np.array([self._t0 + self._n / self.fs], dtype=np.float64)
        self._n += 1

        # 你导师协议是 0~255 的“十进制值”，我们也模拟成这个范围
        eeg = self.rng.integers(0, 256, size=(1, self.eeg_channels), dtype=np.int32).astype(np.float32)
        ecg = self.rng.integers(0, 256, size=(1, self.ecg_channels), dtype=np.int32).astype(np.float32)

        # 加点噪声让它更像真实波动（可选）
        if self.noise > 0:
            eeg += self.rng.normal(0, self.noise, size=eeg.shape).astype(np.float32)
            ecg += self.rng.normal(0, self.noise, size=ecg.shape).astype(np.float32)

        return {
            "t": t,
            "eeg": eeg,
            "ecg": ecg,
            "fs_eeg": self.fs,
            "fs_ecg": self.fs,
            "eeg_channels": self.eeg_channels,
        }

    def close(self):
        pass
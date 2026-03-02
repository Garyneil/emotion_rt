import json
import time
import socket
import numpy as np
from typing import Any, Dict, Optional


# ---------------------------
# Base class
# ---------------------------
class BaseSource:
    name = "base"

    def probe(self) -> bool:
        return False

    def read_chunk(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self):
        pass


# ---------------------------
# LSL
# ---------------------------
class LSLSource(BaseSource):
    name = "lsl"

    def __init__(self, stream_name: Optional[str] = None):
        self.stream_name = stream_name
        self.inlet = None

    def probe(self) -> bool:
        try:
            from pylsl import resolve_stream, StreamInlet
            streams = resolve_stream("type", "EEG", timeout=0.5)
            if not streams:
                return False
            self.inlet = StreamInlet(streams[0])
            return True
        except Exception:
            return False

    def read_chunk(self) -> Dict[str, Any]:
        try:
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
                t = np.asarray(t_list, dtype=np.float64)
                arrival_t = np.asarray(arrival, dtype=np.float64)
                return {"t": t, "eeg": eeg, "ecg": None, "arrival_t": arrival_t}

            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}
        except Exception:
            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

    def close(self):
        try:
            self.inlet = None
        except Exception:
            pass


# ---------------------------
# UDP (JSON payload)
# ---------------------------
class UDPSource(BaseSource):
    name = "udp"

    def __init__(self, host: str = "0.0.0.0", port: int = 9000, buf: int = 65535):
        self.host, self.port, self.buf = host, int(port), int(buf)
        self.sock = None

    def probe(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(0.2)
            return True
        except Exception:
            return False

    def read_chunk(self) -> Dict[str, Any]:
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

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None


# ---------------------------
# Serial (line-based JSON)
# ---------------------------
class SerialSource(BaseSource):
    name = "serial"

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, timeout: float = 0.2):
        self.port = port
        self.baud = int(baud)
        self.timeout = float(timeout)
        self.ser = None

    def probe(self) -> bool:
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            return True
        except Exception:
            return False

    def read_chunk(self) -> Dict[str, Any]:
        try:
            line = self.ser.readline()
            if not line:
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

            s = line.decode("utf-8", errors="ignore").strip()
            if not s:
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

            obj = json.loads(s)
            now = time.time()

            t = np.asarray([obj.get("t", now)], dtype=np.float64)
            eeg = np.asarray([obj["eeg"]], dtype=np.float32) if "eeg" in obj else None
            ecg = np.asarray([obj["ecg"]], dtype=np.float32) if "ecg" in obj else None
            arrival_t = np.asarray([now], dtype=np.float64)

            return {"t": t, "eeg": eeg, "ecg": ecg, "arrival_t": arrival_t}
        except Exception:
            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None


# ---------------------------
# Serial12Hex (0xFF + 12 bytes + 0xFE)
# ---------------------------
class Serial12HexSource(BaseSource):
    name = "serial12hex"

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        head: int = 0xFF,
        tail: int = 0xFE,
        fs: float = 250.0,
        scale_eeg: float = 1.0,
        scale_ecg: float = 1.0,
        timeout: float = 0.2,
        eeg_bytes: int = 8,   # 默认 12 字节里前 8 字节当 EEG
        ecg_bytes: int = 4,   # 默认后 4 字节当 ECG
    ):
        import serial

        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)

        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)

        self.head = head & 0xFF
        self.tail = tail & 0xFF
        self.fs = float(fs)
        self.scale_eeg = float(scale_eeg)
        self.scale_ecg = float(scale_ecg)

        self.eeg_bytes = int(eeg_bytes)
        self.ecg_bytes = int(ecg_bytes)
        if self.eeg_bytes + self.ecg_bytes != 12:
            raise ValueError(f"eeg_bytes + ecg_bytes must be 12, got {self.eeg_bytes}+{self.ecg_bytes}")

    def probe(self) -> bool:
        # 端口能打开就算可用；读不到数据不在 probe 阶段判死
        try:
            return self.ser is not None and self.ser.is_open
        except Exception:
            return False

    def _sync_to_head(self) -> bool:
        while True:
            b = self.ser.read(1)
            if not b:
                return False
            if b[0] == self.head:
                return True

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf

    def read_chunk(self) -> Dict[str, Any]:
        try:
            if not self._sync_to_head():
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

            payload = self._read_exact(13)  # 12 data + 1 tail
            if len(payload) != 13:
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

            data = payload[:12]
            tail = payload[12]
            if tail != self.tail:
                return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

            vals = np.frombuffer(data, dtype=np.uint8).astype(np.float32)

            eeg_raw = vals[: self.eeg_bytes] * self.scale_eeg
            ecg_raw = vals[self.eeg_bytes : self.eeg_bytes + self.ecg_bytes] * self.scale_ecg

            # 统一成 [1, C] 形状
            eeg = eeg_raw[None, :]
            ecg = ecg_raw[None, :]

            now = time.time()
            t = np.asarray([now], dtype=np.float64)
            arrival_t = np.asarray([now], dtype=np.float64)

            return {"t": t, "eeg": eeg, "ecg": ecg, "arrival_t": arrival_t}
        except Exception:
            return {"t": None, "eeg": None, "ecg": None, "arrival_t": None}

    def close(self):
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None


# ---------------------------
# Sim source (optional)
# ---------------------------
class SimSource(BaseSource):
    name = "sim"

    def __init__(self, fs: float = 250.0, eeg_channels: int = 8, ecg_channels: int = 4, noise: float = 5.0, seed: int = 0):
        self.fs = float(fs)
        self.eeg_channels = int(eeg_channels)
        self.ecg_channels = int(ecg_channels)
        self.noise = float(noise)
        self.rng = np.random.default_rng(int(seed))
        self._t0 = time.time()
        self._n = 0

    def probe(self) -> bool:
        return True

    def read_chunk(self) -> Dict[str, Any]:
        t = np.asarray([self._t0 + self._n / self.fs], dtype=np.float64)
        self._n += 1

        eeg = self.rng.integers(0, 256, size=(1, self.eeg_channels), dtype=np.int32).astype(np.float32)
        ecg = self.rng.integers(0, 256, size=(1, self.ecg_channels), dtype=np.int32).astype(np.float32)

        if self.noise > 0:
            eeg += self.rng.normal(0, self.noise, size=eeg.shape).astype(np.float32)
            ecg += self.rng.normal(0, self.noise, size=ecg.shape).astype(np.float32)

        now = time.time()
        arrival_t = np.asarray([now], dtype=np.float64)
        return {"t": t, "eeg": eeg, "ecg": ecg, "arrival_t": arrival_t}


# ---------------------------
# Helper: filter kwargs by allowed keys
# ---------------------------
def _pick(d: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if k in allowed}


# ---------------------------
# Source builder
# ---------------------------
def build_source_auto(cfg: Dict[str, Any]) -> BaseSource:
    """
    Rules:
    1) If cfg['source'] is explicitly specified (not 'auto'), we STRICTLY use it.
    2) If cfg['source'] == 'auto' (or missing), we probe candidates in order.
    3) When passing source_args, we filter keys to avoid "unexpected keyword" errors.
    """
    source_name = (cfg.get("source") or "auto").lower()
    source_args = cfg.get("source_args", {}) or {}

    # ---- Explicit mode: do NOT probe other sources
    if source_name in ("serial12hex", "serial_12hex", "serial12"):
        args = _pick(source_args, {"port", "baudrate", "head", "tail", "fs", "scale_eeg", "scale_ecg", "timeout", "eeg_bytes", "ecg_bytes"})
        s = Serial12HexSource(**args)
        print(f"[Source] selected: {s.name}")
        return s

    if source_name == "udp":
        args = _pick(source_args, {"host", "port", "buf"})
        s = UDPSource(**args)
        if not s.probe():
            raise RuntimeError(f"UDPSource not available on {args.get('host','0.0.0.0')}:{args.get('port',9000)}")
        print(f"[Source] selected: {s.name}")
        return s

    if source_name == "serial":
        # allow both baud and baudrate from config
        args = dict(source_args)
        if "baudrate" in args and "baud" not in args:
            args["baud"] = args["baudrate"]
        args = _pick(args, {"port", "baud", "timeout"})
        s = SerialSource(**args)
        if not s.probe():
            raise RuntimeError(f"SerialSource not available on {args.get('port','/dev/ttyUSB0')}")
        print(f"[Source] selected: {s.name}")
        return s

    if source_name == "lsl":
        s = LSLSource()
        if not s.probe():
            raise RuntimeError("LSLSource not available (no EEG stream found)")
        print(f"[Source] selected: {s.name}")
        return s

    if source_name == "sim":
        args = _pick(source_args, {"fs", "eeg_channels", "ecg_channels", "noise", "seed"})
        s = SimSource(**args)
        print(f"[Source] selected: {s.name}")
        return s

    # ---- Auto probe mode
    if source_name not in ("auto", ""):
        raise ValueError(f"Unknown source '{source_name}'. Valid: auto/udp/serial/serial12hex/lsl/sim")

    candidates = [
        LSLSource(),
        UDPSource(**_pick(source_args, {"host", "port", "buf"})),
        Serial12HexSource(**_pick(source_args, {"port", "baudrate", "head", "tail", "fs", "scale_eeg", "scale_ecg", "timeout", "eeg_bytes", "ecg_bytes"})),
        SerialSource(**_pick({**source_args, **({"baud": source_args.get("baudrate")} if "baudrate" in source_args else {})}, {"port", "baud", "timeout"})),
    ]

    for s in candidates:
        if s.probe():
            print(f"[AutoSource] selected: {s.name}")
            return s

    raise RuntimeError("No available source found (lsl/udp/serial12hex/serial). Please set runtime.source + source_args.")
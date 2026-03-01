import numpy as np

def estimate_fs_from_timestamps(t: np.ndarray) -> float | None:
    if t is None or len(t) < 10:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if len(dt) < 10:
        return None
    med = float(np.median(dt))
    if med <= 0:
        return None
    fs = 1.0 / med
    if fs < 10 or fs > 5000:
        return None
    return fs

def estimate_fs_from_arrival_times(arrival: np.ndarray) -> float | None:
    return estimate_fs_from_timestamps(arrival)

def infer_channel_count(x: np.ndarray) -> int | None:
    if x is None or x.ndim != 2:
        return None
    n0, n1 = x.shape
    if n0 > n1 and n1 <= 512:
        return int(n1)
    if n1 > n0 and n0 <= 512:
        return int(n0)
    return None

def pick_notch_freq(fs: float) -> float:
    if fs and abs((fs / 50.0) - round(fs / 50.0)) < 0.05:
        return 50.0
    return 60.0

class AutoState:
    def __init__(self):
        self.fs_eeg = None
        self.fs_ecg = None
        self.eeg_channels = None
        self.eeg_channel_names = None

    def update_from_chunk(self, chunk: dict):
        t = chunk.get("t", None)
        arrival = chunk.get("arrival_t", None)

        if self.fs_eeg is None:
            self.fs_eeg = estimate_fs_from_timestamps(t) or estimate_fs_from_arrival_times(arrival)
        if self.fs_ecg is None:
            self.fs_ecg = estimate_fs_from_timestamps(t) or estimate_fs_from_arrival_times(arrival)

        if self.eeg_channels is None:
            eeg = chunk.get("eeg", None)
            if isinstance(eeg, np.ndarray):
                self.eeg_channels = infer_channel_count(eeg)

        if self.eeg_channel_names is None and self.eeg_channels:
            self.eeg_channel_names = [f"CH{i+1}" for i in range(self.eeg_channels)]

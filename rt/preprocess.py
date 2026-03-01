import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def _butter_bandpass(x, fs, lo, hi, order=4):
    if x is None:
        return None
    nyq = 0.5 * fs
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x, axis=0)

def _notch(x, fs, f0, q=30.0):
    if x is None:
        return None
    w0 = f0 / (fs / 2)
    if w0 <= 0 or w0 >= 1:
        return x
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, x, axis=0)

def robust_norm(x, eps=1e-6):
    if x is None:
        return None
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True) + eps
    return (x - med) / (1.4826 * mad)

def zscore(x, eps=1e-6):
    if x is None:
        return None
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True) + eps
    return (x - mu) / sd

def preprocess_eeg(eeg, fs, bandpass=(0.5, 45.0), notch=None, normalize="robust"):
    if eeg is None or fs is None:
        return eeg
    x = eeg.astype(np.float32)
    x = _butter_bandpass(x, fs, bandpass[0], bandpass[1])
    if notch:
        x = _notch(x, fs, float(notch))
    if normalize == "robust":
        x = robust_norm(x)
    elif normalize == "zscore":
        x = zscore(x)
    return x

def preprocess_ecg(ecg, fs, bandpass=(0.5, 40.0), normalize="zscore"):
    if ecg is None or fs is None:
        return ecg
    x = ecg.astype(np.float32)
    x = _butter_bandpass(x, fs, bandpass[0], bandpass[1])
    if normalize == "zscore":
        x = zscore(x)
    return x

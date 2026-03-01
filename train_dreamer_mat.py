import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

from rt.model import EmotionFusionNet

def _as_2d_time_channel(x: np.ndarray) -> np.ndarray:
    """
    将输入统一成 [T, C]（时间在前）
    DREAMER 里每段通常是 (T, C) 或 (C, T)，这里做鲁棒处理
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.ndim}D.")
    # 如果是 (C,T) 且 C<=64，转成 (T,C)
    if x.shape[0] <= 64 and x.shape[1] > x.shape[0]:
        x = x.T
    return x.astype(np.float32)

def _robust_z(x: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + eps
    return (x - mu) / sd

def _median_thr(vals: np.ndarray) -> float:
    return float(np.median(vals.astype(np.float32)))

def _to_quad_label(v: float, a: float, thr_v: float, thr_a: float) -> int:
    # 0..3: (valence_high << 1) | arousal_high
    vh = 1 if v >= thr_v else 0
    ah = 1 if a >= thr_a else 0
    return (vh << 1) | ah

class DREAMERMatWindowDataset(Dataset):
    """
    从 stimuli 切滑窗：
    - EEG: [T_eeg, 14]
    - ECG: [T_ecg, C_ecg] (通常 1 或 2)
    标签：ScoreValence/ScoreArousal（每段一个分数）
    """
    def __init__(self, mat_path: str, win_sec=3.0, hop_sec=1.0, use_baseline_subtract=True):
        m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        dreamer = m["DREAMER"]

        self.fs_eeg = int(dreamer.EEG_SamplingRate)
        self.fs_ecg = int(dreamer.ECG_SamplingRate)
        self.eeg_electrodes = list(np.atleast_1d(dreamer.EEG_Electrodes))
        self.data = list(np.atleast_1d(dreamer.Data))

        self.win_eeg = int(round(win_sec * self.fs_eeg))
        self.hop_eeg = int(round(hop_sec * self.fs_eeg))
        self.win_ecg = int(round(win_sec * self.fs_ecg))
        self.hop_ecg = int(round(hop_sec * self.fs_ecg))

        # 先收集全部 valence/arousal 分数，做中位数阈值（通用、稳）
        all_v, all_a = [], []
        for sbj in self.data:
            all_v.append(np.asarray(sbj.ScoreValence, dtype=np.float32))
            all_a.append(np.asarray(sbj.ScoreArousal, dtype=np.float32))
        all_v = np.concatenate(all_v)
        all_a = np.concatenate(all_a)

        self.thr_v = _median_thr(all_v)
        self.thr_a = _median_thr(all_a)

        print(f"[Info] fs_eeg={self.fs_eeg}, fs_ecg={self.fs_ecg}")
        print(f"[Info] EEG electrodes={len(self.eeg_electrodes)}")
        print(f"[Threshold] thr_v={self.thr_v:.3f}, thr_a={self.thr_a:.3f}")

        self.samples = []
        self.use_baseline_subtract = bool(use_baseline_subtract)

        # 生成样本索引：每个被试 18 段 stimuli
        for s_idx, sbj in enumerate(self.data):
            eeg_base_list = np.atleast_1d(sbj.EEG.baseline)   # (18,) object
            eeg_stim_list = np.atleast_1d(sbj.EEG.stimuli)
            ecg_base_list = np.atleast_1d(sbj.ECG.baseline)
            ecg_stim_list = np.atleast_1d(sbj.ECG.stimuli)

            v_scores = np.asarray(sbj.ScoreValence, dtype=np.float32)  # (18,)
            a_scores = np.asarray(sbj.ScoreArousal, dtype=np.float32)

            for vid in range(len(v_scores)):
                # 取一段
                eeg_stim = _as_2d_time_channel(eeg_stim_list[vid])
                ecg_stim = _as_2d_time_channel(ecg_stim_list[vid])

                # baseline subtract（强烈建议：减少个体差异/漂移）
                if self.use_baseline_subtract:
                    eeg_base = _as_2d_time_channel(eeg_base_list[vid])
                    ecg_base = _as_2d_time_channel(ecg_base_list[vid])
                    eeg_stim = eeg_stim - eeg_base.mean(axis=0, keepdims=True)
                    ecg_stim = ecg_stim - ecg_base.mean(axis=0, keepdims=True)

                # 归一化（对训练更稳）
                eeg_stim = _robust_z(eeg_stim)
                ecg_stim = _robust_z(ecg_stim)

                y = _to_quad_label(float(v_scores[vid]), float(a_scores[vid]), self.thr_v, self.thr_a)

                # 滑窗：EEG按 128Hz，ECG按 256Hz，按时间比例对齐窗口起点
                T_eeg = eeg_stim.shape[0]
                for start_eeg in range(0, max(1, T_eeg - self.win_eeg + 1), self.hop_eeg):
                    t0 = start_eeg / self.fs_eeg
                    start_ecg = int(round(t0 * self.fs_ecg))

                    if start_ecg + self.win_ecg > ecg_stim.shape[0]:
                        continue

                    self.samples.append((s_idx, vid, start_eeg, start_ecg, y))

        if len(self.samples) == 0:
            raise RuntimeError("No samples constructed. Check window/hop or mat segment lengths.")

        # 通道数（给模型/导出ONNX用）
        # 从第一条样本读一下真实shape
        s_idx, vid, se, sc, _ = self.samples[0]
        sbj0 = self.data[s_idx]
        eeg0 = _as_2d_time_channel(np.atleast_1d(sbj0.EEG.stimuli)[vid])
        ecg0 = _as_2d_time_channel(np.atleast_1d(sbj0.ECG.stimuli)[vid])
        self.eeg_ch = int(eeg0.shape[1])
        self.ecg_ch = int(ecg0.shape[1])

        print(f"[Shape] eeg_ch={self.eeg_ch}, ecg_ch={self.ecg_ch}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_idx, vid, start_eeg, start_ecg, y = self.samples[idx]
        sbj = self.data[s_idx]

        eeg_base_list = np.atleast_1d(sbj.EEG.baseline)
        eeg_stim_list = np.atleast_1d(sbj.EEG.stimuli)
        ecg_base_list = np.atleast_1d(sbj.ECG.baseline)
        ecg_stim_list = np.atleast_1d(sbj.ECG.stimuli)

        eeg = _as_2d_time_channel(eeg_stim_list[vid])  # [T,C]
        ecg = _as_2d_time_channel(ecg_stim_list[vid])

        if self.use_baseline_subtract:
            eeg_base = _as_2d_time_channel(eeg_base_list[vid])
            ecg_base = _as_2d_time_channel(ecg_base_list[vid])
            eeg = eeg - eeg_base.mean(axis=0, keepdims=True)
            ecg = ecg - ecg_base.mean(axis=0, keepdims=True)

        eeg = _robust_z(eeg)
        ecg = _robust_z(ecg)

        eeg_win = eeg[start_eeg:start_eeg + self.win_eeg, :]  # [win, C]
        ecg_win = ecg[start_ecg:start_ecg + self.win_ecg, :]  # [win, C]

        # 训练模型用 [B,C,T]，所以这里转成 [C,T]
        eeg_win = torch.tensor(eeg_win.T, dtype=torch.float32)
        ecg_win = torch.tensor(ecg_win.T, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return eeg_win, ecg_win, y

def main():
    mat_path = os.environ.get("DREAMER_MAT", r"E:\PycharmProjects\emotion_rt\DREAMER.mat")

    ds = DREAMERMatWindowDataset(mat_path, win_sec=3.0, hop_sec=1.0, use_baseline_subtract=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmotionFusionNet(eeg_ch=ds.eeg_ch, ecg_ch=ds.ecg_ch, num_classes=4).to(device)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    epochs = 8
    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for eeg, ecg, y in dl:
            eeg, ecg, y = eeg.to(device), ecg.to(device), y.to(device)
            logits = model(eeg, ecg)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * eeg.size(0)
            total += eeg.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()

        print(f"[Epoch {ep+1}/{epochs}] loss={loss_sum/total:.4f} acc={correct/total:.3f}")

    torch.save(
        {"state_dict": model.state_dict(),
         "eeg_ch": ds.eeg_ch, "ecg_ch": ds.ecg_ch,
         "fs_eeg": ds.fs_eeg, "fs_ecg": ds.fs_ecg,
         "thr_v": ds.thr_v, "thr_a": ds.thr_a},
        "emotion_model.pt"
    )
    print("Saved: emotion_model.pt (with meta)")

if __name__ == "__main__":
    main()

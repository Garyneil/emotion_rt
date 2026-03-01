import torch
import torch.nn as nn

class TinyCNN1D(nn.Module):
    def __init__(self, in_ch: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 5, padding=2),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width * 2, 5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = width * 2

    def forward(self, x):
        return self.net(x).squeeze(-1)

class EmotionFusionNet(nn.Module):
    def __init__(self, eeg_ch: int, ecg_ch: int, num_classes: int):
        super().__init__()
        self.eeg_enc = TinyCNN1D(eeg_ch, 32)
        self.ecg_enc = TinyCNN1D(ecg_ch, 16)
        dim = self.eeg_enc.out_dim + self.ecg_enc.out_dim
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg, ecg):
        ze = self.eeg_enc(eeg) if eeg is not None else None
        zc = self.ecg_enc(ecg) if ecg is not None else None
        if ze is None and zc is None:
            raise ValueError("Both EEG and ECG are None.")
        if ze is None:
            z = torch.cat([torch.zeros_like(zc), zc], dim=-1)
        elif zc is None:
            z = torch.cat([ze, torch.zeros_like(ze)], dim=-1)
        else:
            z = torch.cat([ze, zc], dim=-1)
        return self.head(z)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from rt.model import EmotionFusionNet

def binarize(v, thr):
    return 1 if v >= thr else 0

def va_to_quad(v, a, thr_v, thr_a):
    # (v,a) -> 4 classes
    return (binarize(v, thr_v) << 1) | binarize(a, thr_a)

def main():
    # Route B: public dataset pretrain example (DREAMER)
    from torcheeg.datasets import DREAMERDataset

    root = os.environ.get("DREAMER_ROOT", "./DREAMER")
    ds = DREAMERDataset(root_path=root, offline_transform=None, online_transform=None)

    sample0 = ds[0]
    print("[DREAMER] sample keys:", list(sample0.keys()))

    def pick_key(keys, candidates):
        for c in candidates:
            if c in keys:
                return c
        return None

    k = set(sample0.keys())
    k_eeg = pick_key(k, ["eeg", "EEG", "signal_eeg"])
    k_ecg = pick_key(k, ["ecg", "ECG", "signal_ecg"])
    k_val = pick_key(k, ["valence", "Valence", "label_valence"])
    k_aro = pick_key(k, ["arousal", "Arousal", "label_arousal"])

    if k_eeg is None or k_ecg is None or k_val is None or k_aro is None:
        raise KeyError("Cannot find expected keys in DREAMER sample. Print ds[0] and adjust key mapping.")

    eeg0 = torch.tensor(sample0[k_eeg], dtype=torch.float32)
    ecg0 = torch.tensor(sample0[k_ecg], dtype=torch.float32)

    # infer channel counts
    if eeg0.ndim == 2 and eeg0.shape[0] > eeg0.shape[1]:
        eeg_ch = int(eeg0.shape[1])
    elif eeg0.ndim == 2:
        eeg_ch = int(eeg0.shape[0])
    else:
        raise ValueError("Unexpected EEG shape.")

    if ecg0.ndim == 2:
        ecg_ch = int(min(ecg0.shape))
    else:
        ecg_ch = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmotionFusionNet(eeg_ch=eeg_ch, ecg_ch=ecg_ch, num_classes=4).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    # thresholds by median (quick & robust)
    vals, aros = [], []
    for i in range(min(len(ds), 500)):
        it = ds[i]
        vals.append(float(it[k_val]))
        aros.append(float(it[k_aro]))
    thr_v = float(torch.tensor(vals).median())
    thr_a = float(torch.tensor(aros).median())
    print(f"[Threshold] thr_v={thr_v:.4f} thr_a={thr_a:.4f}")

    model.train()
    epochs = 10
    for epoch in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for item in tqdm(ds, desc=f"epoch {epoch+1}/{epochs}"):
            eeg = torch.tensor(item[k_eeg], dtype=torch.float32)
            ecg = torch.tensor(item[k_ecg], dtype=torch.float32)

            # to [C,T]
            if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T
            if ecg.ndim == 2 and ecg.shape[0] > ecg.shape[1]:
                ecg = ecg.T
            if ecg.ndim == 1:
                ecg = ecg.unsqueeze(0)

            y = va_to_quad(float(item[k_val]), float(item[k_aro]), thr_v, thr_a)

            eeg = eeg.unsqueeze(0).to(device)
            ecg = ecg.unsqueeze(0).to(device)
            y = torch.tensor([y], dtype=torch.long, device=device)

            logits = model(eeg, ecg)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            total += 1
            correct += int(logits.argmax(dim=-1).item() == y.item())

        print(f"[Epoch {epoch+1}] loss={loss_sum/total:.4f} acc={correct/total:.3f}")

    torch.save(model.state_dict(), "emotion_model.pt")
    print("Saved: emotion_model.pt")

if __name__ == "__main__":
    main()

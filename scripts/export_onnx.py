import torch
from rt.model import EmotionFusionNet

def main():
    # Adjust eeg_ch/ecg_ch to match your trained model
    eeg_ch = 14
    ecg_ch = 1
    num_classes = 4

    model = EmotionFusionNet(eeg_ch=eeg_ch, ecg_ch=ecg_ch, num_classes=num_classes)
    sd = torch.load("emotion_model.pt", map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    eeg = torch.randn(1, eeg_ch, 384)
    ecg = torch.randn(1, ecg_ch, 384)

    torch.onnx.export(
        model,
        (eeg, ecg),
        "emotion_model.onnx",
        input_names=["eeg", "ecg"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "eeg": {0: "B", 2: "T"},
            "ecg": {0: "B", 2: "T"},
            "logits": {0: "B"},
        },
    )
    print("Exported: emotion_model.onnx")

if __name__ == "__main__":
    main()

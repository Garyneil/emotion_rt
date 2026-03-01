import numpy as np

def decode_label(prob: np.ndarray, label_cfg: dict) -> str:
    scheme = label_cfg.get("scheme", "quad_va")
    idx = int(np.argmax(prob))
    if scheme == "quad_va":
        mapping = {0: "LVLA", 1: "LVHA", 2: "HVLA", 3: "HVHA"}
        return mapping.get(idx, f"class_{idx}")
    return f"class_{idx}"


#“标签解码器”（Valence/Arousal + 情绪语义）

def decode_quad_label(label: int):
    """
    label: 0..3
    bit1: valence (0=Low, 1=High)
    bit0: arousal (0=Low, 1=High)
    """
    vh = (label >> 1) & 1
    ah = label & 1

    valence = "High" if vh == 1 else "Low"
    arousal = "High" if ah == 1 else "Low"

    # 语义映射（标准 VA 四象限）
    if vh == 0 and ah == 0:
        semantic = "Calm / Bored"      # 低效价低唤醒
    elif vh == 0 and ah == 1:
        semantic = "Stress / Anxiety"  # 低效价高唤醒
    elif vh == 1 and ah == 0:
        semantic = "Relaxed / Content" # 高效价低唤醒
    else:
        semantic = "Excited / Happy"   # 高效价高唤醒

    return valence, arousal, semantic

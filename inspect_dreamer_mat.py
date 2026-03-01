import os
import numpy as np

def _load_mat(path):
    # 兼容 v7.3 / 非 v7.3
    try:
        import h5py
        with h5py.File(path, "r") as f:
            print("[MAT] v7.3 (HDF5) detected.")
            print("[MAT] top keys:", list(f.keys()))
        return None, "v7.3"
    except Exception:
        from scipy.io import loadmat
        m = loadmat(path, struct_as_record=False, squeeze_me=True)
        print("[MAT] classic mat detected.")
        print("[MAT] top keys:", [k for k in m.keys() if not k.startswith("__")])
        return m, "classic"

def _describe(obj, name="root", depth=0, max_depth=3):
    ind = "  " * depth
    if depth > max_depth:
        return
    t = type(obj)
    if isinstance(obj, np.ndarray):
        print(f"{ind}{name}: ndarray shape={obj.shape} dtype={obj.dtype}")
        if obj.dtype == object and obj.size > 0:
            print(f"{ind}  (object array) first element type:", type(obj.flat[0]))
            _describe(obj.flat[0], name=name+"[0]", depth=depth+1, max_depth=max_depth)
    else:
        print(f"{ind}{name}: type={t}")

    # scipy mat_struct
    if hasattr(obj, "_fieldnames"):
        print(f"{ind}{name}: mat_struct fields={obj._fieldnames}")
        for fn in obj._fieldnames[:10]:
            _describe(getattr(obj, fn), name=f"{name}.{fn}", depth=depth+1, max_depth=max_depth)

def main():
    # 改成你的 mat 路径
    mat_path = os.environ.get("DREAMER_MAT", "DREAMER.mat")
    m, kind = _load_mat(mat_path)
    if kind == "v7.3":
        print("你的 mat 是 v7.3：我给你另一版读取脚本（h5py 专用）。")
        return

    # 通常你会看到 DREAMER 这个变量名
    key = None
    for k in m.keys():
        if not k.startswith("__"):
            if k.lower() == "dreamer":
                key = k
                break
    if key is None:
        key = [k for k in m.keys() if not k.startswith("__")][0]

    dreamer = m[key]
    print("\n[Inspect] Using key:", key)
    _describe(dreamer, "DREAMER", 0, 4)

    # 看 Data
    if hasattr(dreamer, "_fieldnames") and "Data" in dreamer._fieldnames:
        data = getattr(dreamer, "Data")
        print("\n[Inspect] DREAMER.Data")
        _describe(data, "Data", 0, 3)

if __name__ == "__main__":
    main()

import numpy as np

def fuse_series(motion, audio, pose=None, w_m=0.65, w_a=0.35, w_p=0.0, clip=(0,1)):
    # 長さ合わせ（最短に揃える）
    L = min(len(motion), len(audio), len(pose) if pose is not None else 10**9)
    m = motion[:L]; a = audio[:L]
    if pose is None:
        p = np.zeros(L, dtype=np.float32)
    else:
        p = pose[:L]
    x = w_m * m + w_a * a + w_p * p
    lo, hi = clip
    return np.clip(x, lo, hi)

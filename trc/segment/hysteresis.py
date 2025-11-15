import numpy as np
from typing import List, Tuple


def hysteresis_segment(
    x: np.ndarray,
    dt: float,
    low: float,
    high: float,
    min_play: float,
    min_gap: float,
    snap_peak_sec: float,
) -> List[Tuple[float, float]]:
    assert 0.0 <= low <= high <= 1.0
    L = len(x)
    if L == 0:
        return []

    on = False
    segs_idx: list[tuple[int, int]] = []
    start = 0

    for i, v in enumerate(x):
        if not on and v >= high:
            on = True
            start = i
        elif on and v < low:
            on = False
            segs_idx.append((start, i))

    if on:
        segs_idx.append((start, L - 1))

    # 最小長でフィルタ
    min_len = int(round(min_play / dt))
    segs_idx = [
        (s, e) for s, e in segs_idx
        if (e - s + 1) >= max(min_len, 1)
    ]

    # 短いギャップのマージ
    merged: list[list[int]] = []
    for s, e in segs_idx:
        if not merged:
            merged.append([s, e])
            continue
        ps, pe = merged[-1]
        gap_sec = (s - pe - 1) * dt
        if gap_sec < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    segs_idx = [(s, e) for s, e in merged]

    # 端点スナップ
    r = int(round(snap_peak_sec / dt)) if snap_peak_sec > 0 else 0

    def local_peak(i0: int, up: bool = True) -> int:
        if r <= 0:
            return i0
        lo = max(0, i0 - r)
        hi = min(L, i0 + r + 1)
        window = x[lo:hi]
        if window.size == 0:
            return i0
        rel = int(np.argmax(window) if up else np.argmin(window))
        return lo + rel

    snapped: list[tuple[int, int]] = []
    for s, e in segs_idx:
        s2 = local_peak(s, up=True)
        e2 = local_peak(e, up=False)
        if e2 <= s2:
            e2 = s2 + max(1, min_len)
        snapped.append((s2, e2))

    # 秒に変換
    return [(s * dt, e * dt) for s, e in snapped]

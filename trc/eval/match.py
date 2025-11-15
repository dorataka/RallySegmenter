from typing import List, Tuple

def match_segments(
    gt: List[Tuple[float, float]],
    pred: List[Tuple[float, float]],
    tol: float,
    iou_thr: float,
) -> tuple[int, int, int]:
    tp = 0
    used_pred = set()

    for g0, g1 in gt:
        g0_t = g0 - tol
        g1_t = g1 + tol
        best_iou = -1.0
        best_idx = -1

        for i, (p0, p1) in enumerate(pred):
            if i in used_pred:
                continue
            inter = max(0.0, min(g1_t, p1) - max(g0_t, p0))
            union = max(g1_t, p1) - min(g0_t, p0) + 1e-6
            iou = inter / union
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx >= 0:
            tp += 1
            used_pred.add(best_idx)

    return tp, len(gt), len(pred)

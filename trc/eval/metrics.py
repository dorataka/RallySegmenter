def precision_recall_f1(tp: int, gt_n: int, pred_n: int):
    p = tp / max(pred_n, 1) if pred_n > 0 else 0.0
    r = tp / max(gt_n, 1) if gt_n > 0 else 0.0
    if p + r == 0:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r)
    return p, r, f1

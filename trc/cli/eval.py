import argparse
import csv
import os

from trc.config import load_config
from trc.eval.match import match_segments
from trc.eval.metrics import precision_recall_f1


def _load_segments(path: str):
    segs = []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            s = float(row["start"])
            e = float(row["end"])
            segs.append((s, e))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT csv (start,end)")
    ap.add_argument("--pred", required=True, help="Pred csv (start,end)")
    ap.add_argument(
        "--config",
        default=os.path.join("configs", "default.yml"),
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    gt = _load_segments(args.gt)
    pred = _load_segments(args.pred)

    tp, n_gt, n_pred = match_segments(
        gt,
        pred,
        tol=cfg.eval.tolerance_sec,
        iou_thr=cfg.eval.iou_thr,
    )
    p, r, f1 = precision_recall_f1(tp, n_gt, n_pred)

    print(f"GT: {n_gt}, Pred: {n_pred}, TP: {tp}")
    print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")


if __name__ == "__main__":
    main()

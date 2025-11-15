import argparse
import csv
import json
import os
from typing import List, Tuple, Dict


def load_segments(path: str) -> List[Dict[str, float]]:
    """
    CSV からセグメントを読み込む。
    ヘッダ: start, end
    区切り: カンマ or タブ を自動判定。
    """
    segs: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        if "start" not in reader.fieldnames or "end" not in reader.fieldnames:
            raise ValueError(f"{path} に 'start' / 'end' ヘッダがありません")

        for row in reader:
            try:
                s = float(row["start"])
                e = float(row["end"])
            except (KeyError, ValueError):
                continue
            if e > s:
                segs.append({"start": s, "end": e})

    # start でソート
    segs.sort(key=lambda x: x["start"])
    return segs


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    1 次元区間の IoU (Intersection over Union) を計算する。
    """
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_segments(
    gt: List[Dict[str, float]],
    pred: List[Dict[str, float]],
    iou_thr: float = 0.5,
) -> Dict:
    """
    区間 IoU に基づいて 1:1 マッチングを行い、
    TP / FP / FN / Precision / Recall / F1 を計算する。
    """

    gt_intervals = [(g["start"], g["end"]) for g in gt]
    pred_intervals = [(p["start"], p["end"]) for p in pred]

    matched_gt = set()
    matches = []

    for j, p in enumerate(pred_intervals):
        best_iou = 0.0
        best_i = -1
        for i, g in enumerate(gt_intervals):
            if i in matched_gt:
                continue
            iou = interval_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_iou >= iou_thr and best_i >= 0:
            matched_gt.add(best_i)
            matches.append(
                {
                    "pred_index": j,
                    "gt_index": best_i,
                    "iou": best_iou,
                }
            )

    n_tp = len(matches)
    n_pred = len(pred_intervals)
    n_gt = len(gt_intervals)
    n_fp = n_pred - n_tp
    n_fn = n_gt - n_tp

    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall = n_tp / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # GT ごとの coverage もついでに計算（参考値）
    gt_coverages = []
    for i, g in enumerate(gt_intervals):
        g_len = g[1] - g[0]
        if g_len <= 0:
            gt_coverages.append(0.0)
            continue
        # その GT と重なる pred を全部見て「どれくらい覆われているか」を計算
        covered = 0.0
        for p in pred_intervals:
            inter = max(0.0, min(g[1], p[1]) - max(g[0], p[0]))
            covered += inter
        covered = min(covered, g_len)
        gt_coverages.append(covered / g_len)

    metrics = {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "iou_thr": iou_thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gt_coverages": gt_coverages,
        "matches": matches,
    }
    return metrics


def main():
    ap = argparse.ArgumentParser(description="ラリーセグメントの GT / 予測 の評価スクリプト")
    ap.add_argument("--gt", required=True, help="GT セグメント CSV (start,end)")
    ap.add_argument("--pred", required=True, help="予測セグメント CSV (start,end)")
    ap.add_argument(
        "--iou_thr",
        type=float,
        default=0.5,
        help="マッチ判定に使う IoU 閾値 (default: 0.5)",
    )
    ap.add_argument(
        "--stats",
        type=str,
        default="",
        help="評価結果を JSON で保存するパス（省略可）",
    )
    args = ap.parse_args()

    gt = load_segments(args.gt)
    pred = load_segments(args.pred)

    print(f"[info] GT segments   : {len(gt)}")
    print(f"[info] Pred segments : {len(pred)}")
    print(f"[info] IoU threshold : {args.iou_thr}")

    metrics = evaluate_segments(gt, pred, iou_thr=args.iou_thr)

    print("\n=== Summary ===")
    print(f"GT (n_gt)         : {metrics['n_gt']}")
    print(f"Pred (n_pred)     : {metrics['n_pred']}")
    print(f"TP                : {metrics['n_tp']}")
    print(f"FP                : {metrics['n_fp']}")
    print(f"FN                : {metrics['n_fn']}")
    print(f"Precision         : {metrics['precision']:.3f}")
    print(f"Recall            : {metrics['recall']:.3f}")
    print(f"F1                : {metrics['f1']:.3f}")

    if metrics["gt_coverages"]:
        avg_cov = sum(metrics["gt_coverages"]) / len(metrics["gt_coverages"])
        print(f"Avg GT coverage   : {avg_cov:.3f}")
        print("GT coverage (per segment):")
        for i, c in enumerate(metrics["gt_coverages"]):
            print(f"  GT#{i:02d}: {c:.3f}")

    print("\n=== Matches (pred -> gt) ===")
    for m in metrics["matches"]:
        pj = m["pred_index"]
        gi = m["gt_index"]
        iou = m["iou"]
        p = pred[pj]
        g = gt[gi]
        print(
            f"  pred#{pj:02d} [{p['start']:.3f}–{p['end']:.3f}]"
            f"  <->  gt#{gi:02d} [{g['start']:.3f}–{g['end']:.3f}]"
            f"  IoU={iou:.3f}"
        )

    if args.stats:
        os.makedirs(os.path.dirname(args.stats), exist_ok=True)
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[info] Saved stats JSON -> {os.path.abspath(args.stats)}")


if __name__ == "__main__":
    main()

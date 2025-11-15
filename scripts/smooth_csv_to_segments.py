# scripts/smooth_csv_to_segments.py
#
# 使い方:
#   python scripts/smooth_csv_to_segments.py stats/features_with_play_smooth.csv \
#       --out_csv stats/smooth_segments.csv \
#       --min_len 1.0
#
# やること:
#   - pred_label_smooth == 1 の連続区間を検出
#   - start,end 列を持つ CSV を出力（cut_segments.py 向け）

import argparse
import pandas as pd

def detect_segments(df, t_col="t", label_col="pred_label_smooth", min_len=1.0):
    t = df[t_col].values
    y = df[label_col].values

    segments = []
    start = None

    for i in range(len(y)):
        if y[i] == 1 and start is None:
            start = t[i]
        if (y[i] == 0 or i == len(y) - 1) and start is not None:
            end = t[i] if y[i] == 0 else t[i]
            if end - start >= min_len:
                segments.append((start, end))
            start = None

    return segments

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="features_with_play_smooth.csv")
    ap.add_argument("--out_csv", default="stats/smooth_segments.csv",
                    help="start,end を書き出すCSVパス")
    ap.add_argument("--min_len", type=float, default=1.0,
                    help="最低ラリー長 [秒]")
    args = ap.parse_args()

    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)

    if "pred_label_smooth" not in df.columns:
        raise ValueError("CSVに pred_label_smooth 列がありません。")

    segments = detect_segments(df, min_len=args.min_len)
    print(f"[info] found {len(segments)} segments")

    out_df = pd.DataFrame(segments, columns=["start", "end"])
    print(f"[save] {args.out_csv}")
    out_df.to_csv(args.out_csv, index=False)
    print("[done]")

if __name__ == "__main__":
    main()

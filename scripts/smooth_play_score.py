# scripts/smooth_play_score.py
#
# 使い方:
#   python scripts/smooth_play_score.py stats/features_with_play.csv \
#       --out_csv stats/features_with_play_smooth.csv \
#       --win_sec 0.5 \
#       --thr 0.715
#
# やること:
#   - features_with_play.csv を読み込む（p_play_raw 必須）
#   - 時間方向に win_sec 秒ぶんの移動平均で平滑化
#   - p_play_smooth を追加
#   - thr を指定した場合は pred_label_smooth も追加

import argparse
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="p_play_raw を含む CSV (features_with_play.csv)")
    ap.add_argument("--out_csv", default=None,
                    help="平滑化結果を書き出す CSV パス")
    ap.add_argument("--win_sec", type=float, default=0.5,
                    help="平滑化に使う窓の長さ [秒] (デフォルト: 0.5秒)")
    ap.add_argument("--thr", type=float, default=None,
                    help="0/1 ラベルを作るための閾値（指定したときのみ pred_label_smooth を追加）")
    args = ap.parse_args()

    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)

    if "t" not in df.columns:
        raise ValueError("入力CSVに 't' 列がありません。")
    if "p_play_raw" not in df.columns:
        raise ValueError("入力CSVに 'p_play_raw' 列がありません。")

    # 出力ファイル名の決定
    out_csv = args.out_csv
    if out_csv is None:
        if args.csv.lower().endswith(".csv"):
            out_csv = args.csv[:-4] + ".smooth.csv"
        else:
            out_csv = args.csv + ".smooth.csv"

    # サンプリング間隔 dt を推定
    dt = df["t"].diff().median()
    if pd.isna(dt) or dt <= 0:
        raise ValueError("t の差分から有効な dt が計算できませんでした。")

    win_sec = float(args.win_sec)
    win_samples = max(1, int(round(win_sec / dt)))
    print(f"[info] 推定サンプリング間隔 dt = {dt:.6f} 秒")
    print(f"[info] 平滑化窓: {win_sec:.3f} 秒 ≒ {win_samples} サンプル")

    s = df["p_play_raw"].astype(float)

    # center=True: 現在時刻を中心に前後 win_sec/2 くらいを見る感じになる
    df["p_play_smooth"] = s.rolling(
        window=win_samples,
        min_periods=1,
        center=True
    ).mean()

    # オプションで 0/1 ラベルも作る
    if args.thr is not None:
        thr = float(args.thr)
        print(f"[info] apply threshold thr={thr:.4f} → pred_label_smooth を追加")
        df["pred_label_smooth"] = (df["p_play_smooth"] >= thr).astype(int)

    print(f"[save] {out_csv}")
    df.to_csv(out_csv, index=False)
    print("[done]")

if __name__ == "__main__":
    main()

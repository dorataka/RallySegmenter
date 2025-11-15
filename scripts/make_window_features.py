# input:
#   t, motion, audio, pose, fused, label ...
# output:
#   上に加えて:
#   motion_mean_3s, motion_max_3s,
#   audio_mean_3s,  audio_max_3s,
#   pose_mean_3s,   pose_max_3s

import sys
import pandas as pd

def main(in_csv: str, out_csv: str | None = None):
    if out_csv is None:
        if in_csv.lower().endswith(".csv"):
            out_csv = in_csv[:-4] + ".windowed.csv"
        else:
            out_csv = in_csv + ".windowed.csv"

    print(f"[load] {in_csv}")
    df = pd.read_csv(in_csv)

    if "t" not in df.columns:
        raise ValueError("入力CSVに 't' 列がありません。")

    # サンプリング間隔 dt を推定（中央値を採用）
    dt = df["t"].diff().median()
    if pd.isna(dt) or dt <= 0:
        raise ValueError("t の差分から有効な dt が計算できませんでした。")

    # 3秒ぶんのサンプル数（前後1.5秒ずつ）
    samples_3s = max(1, int(round(3.0 / dt)))
    print(f"[info] 推定サンプリング間隔 dt = {dt:.6f} 秒 → 3秒あたり {samples_3s} サンプル")
    print("       （center=True なので、前後1.5秒ずつを使います）")

    base_feats = ["motion", "audio", "pose"]
    for name in base_feats:
        if name not in df.columns:
            print(f"[warn] 列 '{name}' がないのでスキップします。")
            continue

        s = df[name].astype(float)

        # center=True にして、窓の中心時刻に平均・最大を割り当てる
        mean_col = f"{name}_mean_3s"
        max_col  = f"{name}_max_3s"

        df[mean_col] = s.rolling(
            window=samples_3s,
            min_periods=1,
            center=True
        ).mean()

        df[max_col] = s.rolling(
            window=samples_3s,
            min_periods=1,
            center=True
        ).max()

        print(f"[feat] added {mean_col}, {max_col}")

    print(f"[save] {out_csv}")
    df.to_csv(out_csv, index=False)
    print("[done]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python make_window_features.py input.csv [output.csv]")
        raise SystemExit(1)
    in_csv = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else None
    main(in_csv, out_csv)

# scripts/make_features_windowed_from_smooth.py
"""
stats/features_with_play_smooth_input1〜4.csv を全部読み込んで縦結合し、
gt -> label にリネームして stats/features_windowed.csv に保存するスクリプト。
"""

import glob
import os

import pandas as pd


def main():
    # 必要ならパターンは適宜変えてOK
    pattern = os.path.join("stats", "features_with_play_smooth_input*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matched: {pattern}")

    print("[info] input files:")
    for f in files:
        print("  -", f)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"[load] {f}: shape={df.shape}")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[info] concatenated shape: {df_all.shape}")

    # ラベル列を gt -> label に揃える
    if "label" not in df_all.columns:
        if "gt" in df_all.columns:
            df_all = df_all.rename(columns={"gt": "label"})
            print("[info] rename column: gt -> label")
        else:
            raise KeyError("ラベル列 'gt' も 'label' も見つかりませんでした。")

    out_path = os.path.join("stats", "features_windowed.csv")
    os.makedirs("stats", exist_ok=True)
    df_all.to_csv(out_path, index=False)
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()

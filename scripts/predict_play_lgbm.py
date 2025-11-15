# scripts/predict_play_lgbm.py
#
# 使い方:
#   python scripts/predict_play_lgbm.py stats/features_windowed.csv \
#       --model stats/play_lgbm_model.txt \
#       --out_csv stats/features_with_play.csv \
#       --thr 0.72

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="features_windowed.csv のパス")
    ap.add_argument("--model", default="stats/play_lgbm_model.txt",
                    help="学習済み LightGBM モデルのパス")
    ap.add_argument("--out_csv", default="stats/features_with_play.csv",
                    help="p_play_raw を付与して保存する CSV パス")
    ap.add_argument("--thr", type=float, default=None,
                    help="0/1 ラベルを作るための閾値（指定しなければ作らない）")
    args = ap.parse_args()

    print(f"[load] csv: {args.csv}")
    df = pd.read_csv(args.csv)

    print(f"[load] model: {args.model}")
    booster = lgb.Booster(model_file=args.model)

    candidate_feats = [
        "motion", "audio", "pose",
        "motion_mean_3s", "motion_max_3s",
        "audio_mean_3s",  "audio_max_3s",
        "pose_mean_3s",   "pose_max_3s",
    ]
    feature_cols = [c for c in candidate_feats if c in df.columns]
    if not feature_cols:
        raise ValueError("使用できる特徴量列が見つかりませんでした。")

    print(f"[info] use features: {feature_cols}")
    X = df[feature_cols].astype(float).values

    print("[predict] p_play_raw を計算中…")
    y_prob = booster.predict(X, num_iteration=booster.best_iteration)
    df["p_play_raw"] = y_prob

    if args.thr is not None:
        thr = float(args.thr)
        print(f"[info] apply threshold thr={thr:.4f} → pred_label を追加")
        df["pred_label"] = (df["p_play_raw"] >= thr).astype(int)

    print(f"[save] {args.out_csv}")
    df.to_csv(args.out_csv, index=False)
    print("[done]")

if __name__ == "__main__":
    main()

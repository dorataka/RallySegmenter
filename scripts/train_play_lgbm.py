# 使い方:
#   python scripts/train_play_lgbm.py stats/features_windowed.csv \
#       --model stats/play_lgbm_model.txt \
#       --pred_csv stats/play_pred_valid.csv
#
# やること:
#   - features_windowed.csv を読み込む
#   - LightGBM で「プレー中確率 p_play_raw(t)」を学習
#   - valid 部分で F1 が最大になる閾値を探索して表示
#   - モデルを保存（.txt）
#   - valid の予測結果を CSV に保存（デバッグ・可視化用）

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score

# LightGBM のバージョン差に対応した callback import
try:
    from lightgbm import early_stopping, log_evaluation
except ImportError:
    # 古いバージョン向けフォールバック
    early_stopping = lgb.early_stopping
    log_evaluation = lgb.log_evaluation


def find_best_threshold(y_true, y_prob, num_steps: int = 401):
    """
    0.0〜1.0 の間で閾値をスキャンして、F1 が最大となる閾値を探す。
    """
    best_thr = 0.5
    best_f1 = -1.0

    for thr in np.linspace(0.0, 1.0, num_steps):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="features_windowed.csv のパス")
    ap.add_argument("--model", default="stats/play_lgbm_model.txt",
                    help="LightGBM モデル保存先")
    ap.add_argument("--pred_csv", default="stats/play_pred_valid.csv",
                    help="valid 部分の予測を書き出す CSV パス")
    args = ap.parse_args()

    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)

    # 必須列の確認
    required = ["t", "label", "motion", "audio", "pose"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"入力CSVに必須列 '{col}' がありません。")

    # 特徴量に使う列を指定
    candidate_feats = [
        "motion", "audio", "pose",
        "motion_mean_3s", "motion_max_3s",
        "audio_mean_3s",  "audio_max_3s",
        "pose_mean_3s",   "pose_max_3s",
    ]
    feature_cols = [c for c in candidate_feats if c in df.columns]

    print(f"[info] use features: {feature_cols}")

    X = df[feature_cols].astype(float).values
    y = df["label"].astype(int).values

    n_samples = len(df)
    pos_ratio = (y == 1).mean()
    print(f"[info] samples: {n_samples}, positive ratio (label==1): {pos_ratio:.3f}")

    # 時系列として 先頭 80% を train, 後ろ 20% を valid にする
    split_idx = int(n_samples * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_valid, y_valid = X[split_idx:], y[split_idx:]
    t_valid = df["t"].values[split_idx:]

    print(f"[info] train size: {len(y_train)}, valid size: {len(y_valid)}")

    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_cols)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }

    print("[train] LightGBM で学習中…")
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            early_stopping(50),   # 50ラウンド改善しなければ打ち切り
            log_evaluation(50),   # 50ラウンドごとにログ表示
        ],
    )

    print(f"[save] model → {args.model}")
    booster.save_model(args.model)

    print("[predict] valid 部分の p_play_raw を計算")
    y_prob_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)

    best_thr, best_f1 = find_best_threshold(y_valid, y_prob_valid)
    print(f"[info] best threshold: {best_thr:.4f}, F1={best_f1:.4f}")

    y_pred_valid = (y_prob_valid >= best_thr).astype(int)
    print("---- classification report (valid, thr=best) ----")
    print(classification_report(y_valid, y_pred_valid, digits=4))

    # 予測結果を CSV に書き出し（後で可視化や解析に使える）
    out_df = pd.DataFrame({
        "t": t_valid,
        "label": y_valid,
        "p_play_raw": y_prob_valid,
        "pred_label": y_pred_valid,
    })
    print(f"[save] valid predictions → {args.pred_csv}")
    out_df.to_csv(args.pred_csv, index=False)

    # 特徴量重要度をちょっと表示
    print("---- feature importances ----")
    for name, imp in sorted(
        zip(feature_cols, booster.feature_importance()),
        key=lambda x: -x[1]
    ):
        print(f"{name:16s} {imp}")


if __name__ == "__main__":
    main()

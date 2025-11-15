# scripts/features_to_smooth_segments.py
#
# 使い方:
#   python scripts/features_to_smooth_segments.py stats/input.features.csv \
#       --model stats/play_lgbm_model.txt \
#       --win_sec 0.5 \
#       --thr 0.715 \
#       --min_len 1.0 \
#       --smooth_csv stats/features_with_play_smooth.csv \
#       --segments_csv stats/smooth_segments.csv
#
# やること:
#   1) input.features.csv → （一時ディレクトリ内の）windowed_tmp.csv
#   2) windowed_tmp.csv → （一時ディレクトリ内の）with_play_tmp.csv
#   3) with_play_tmp.csv → 最終成果物 smooth_csv（= features_with_play_smooth.csv）
#   4) smooth_csv → 最終成果物 segments_csv（= smooth_segments.csv）
#
# プロジェクト直下には「features_with_play_smooth.csv」と
# 「smooth_segments.csv」以外の CSV は増えないようにしている。

import argparse
import os
import sys
import subprocess
import tempfile
import shutil

def run(cmd):
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_csv", help="input.features.csv のパス")
    ap.add_argument("--model", default="stats/play_lgbm_model.txt",
                    help="学習済み LightGBM モデルのパス")
    ap.add_argument("--win_sec", type=float, default=0.5,
                    help="p_play_raw の平滑化窓 [秒]")
    ap.add_argument("--thr", type=float, default=0.715,
                    help="プレー/非プレー判定に使う閾値")
    ap.add_argument("--min_len", type=float, default=1.0,
                    help="ラリーとして採用する最短長 [秒]")
    ap.add_argument("--smooth_csv", default=None,
                    help="p_play_smooth 付き最終 CSV（デフォルト: input と同ディレクトリに features_with_play_smooth.csv）")
    ap.add_argument("--segments_csv", default=None,
                    help="start,end を書き出す CSV（デフォルト: input と同ディレクトリに smooth_segments.csv）")
    args = ap.parse_args()

    # パス周り
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scripts_dir = os.path.join(proj_root, "scripts")

    def sp(name: str) -> str:
        return os.path.join(scripts_dir, name)

    in_features = os.path.abspath(args.features_csv)
    base_dir = os.path.dirname(in_features) or "."

    # 最終成果物のパス
    smooth_csv = args.smooth_csv or os.path.join(base_dir, "features_with_play_smooth.csv")
    segments_csv = args.segments_csv or os.path.join(base_dir, "smooth_segments.csv")

    print("[info] features_csv       :", in_features)
    print("[info] smooth_csv (out)   :", smooth_csv)
    print("[info] segments_csv (out) :", segments_csv)
    print("[info] model              :", args.model)
    print("[info] win_sec / thr / min_len =", args.win_sec, args.thr, args.min_len)

    # 一時ディレクトリ作成（中間CSVはここに閉じ込める）
    tmp_dir = tempfile.mkdtemp(prefix="fts_tmp_")
    print("[info] use tmp_dir        :", tmp_dir)

    try:
        windowed_csv = os.path.join(tmp_dir, "windowed_tmp.csv")
        with_play_csv = os.path.join(tmp_dir, "with_play_tmp.csv")

        # 1) 3秒窓の特徴量を作成（中間ファイルは tmp_dir）
        run([
            sys.executable,
            sp("make_window_features.py"),
            in_features,
            windowed_csv,
        ])

        # 2) LightGBM モデルで p_play_raw を付与（中間ファイルは tmp_dir）
        run([
            sys.executable,
            sp("predict_play_lgbm.py"),
            windowed_csv,
            "--model", args.model,
            "--out_csv", with_play_csv,
            "--thr", str(args.thr),
        ])

        # 3) p_play_raw を平滑化して p_play_smooth を追加（← これは最終成果物）
        run([
            sys.executable,
            sp("smooth_play_score.py"),
            with_play_csv,
            "--out_csv", smooth_csv,
            "--win_sec", str(args.win_sec),
            "--thr", str(args.thr),
        ])

        # 4) pred_label_smooth から start,end CSV を生成（← これも最終成果物）
        run([
            sys.executable,
            sp("smooth_csv_to_segments.py"),
            smooth_csv,
            "--out_csv", segments_csv,
            "--min_len", str(args.min_len),
        ])

        print("[done] pipeline completed.")
        print("  input.features.csv        →", in_features)
        print("  features_with_play_smooth →", smooth_csv)
        print("  smooth_segments.csv       →", segments_csv)

    finally:
        # 一時ディレクトリを掃除したい場合はここで消す
        try:
            shutil.rmtree(tmp_dir)
            print("[cleanup] removed tmp_dir :", tmp_dir)
        except Exception as e:
            print("[warn] tmp_dir cleanup failed:", e)


if __name__ == "__main__":
    main()

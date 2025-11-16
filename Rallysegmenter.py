from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys
import json


def run(cmd: list[str]) -> None:
    """サブコマンドを実行する小さなユーティリティ"""
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_best_thr(meta_path: Path) -> float | None:
    """
    LightGBM 学習時に保存した best_thr を読み込むヘルパー。
    ファイルがない / キーがない場合は None を返す。
    """
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    val = meta.get("best_thr")
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="input.mp4 → rallies_smooth.mp4 を一気に作るラッパー"
    )
    parser.add_argument("input_mp4", help="入力動画 (例: input.mp4)")
    parser.add_argument(
        "-o", "--out",
        default="rallies_smooth.mp4",
        help="出力ハイライト動画のパス (デフォルト: rallies_smooth.mp4)",
    )
    parser.add_argument(
        "--workdir",
        default="stats/smooth_highlight",
        help="中間CSVを置く作業ディレクトリ (デフォルト: stats/smooth_highlight)",
    )
    parser.add_argument(
        "--model",
        default="stats/play_lgbm_model.txt",
        help="学習済み LGBM モデルのパス",
    )
    parser.add_argument(
        "--meta",
        default="stats/play_lgbm_meta.json",
        help="学習時に保存したメタ情報(JSON)のパス（best_thr など）",
    )
    # thr は「未指定(None)なら meta の best_thr を使う」方針
    parser.add_argument(
        "--thr",
        type=float,
        default=None,
        help="play判定のしきい値（未指定なら meta の best_thr を使用）",
    )
    parser.add_argument("--min-len", type=float, default=2.5, help="最短ラリー長 [sec]")
    parser.add_argument("--max-gap", type=float, default=0.4, help="ラリーをマージする最大ギャップ [sec]")
    parser.add_argument("--pad-head", type=float, default=0.20, help="ラリー前に足す余白 [sec]")
    parser.add_argument("--pad-tail", type=float, default=0.15, help="ラリー後に足す余白 [sec]")

    # ★ 追加：動画生成をスキップするフラグ
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="ラリー動画(rallies_smooth.mp4)の生成をスキップして、CSV出力で終了する",
    )

    args = parser.parse_args()

    in_mp4 = Path(args.input_mp4)
    out_mp4 = Path(args.out)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # 中間ファイルのパスをここで一元管理
    features_input = workdir / "features_input.csv"
    features_window = workdir / "features_windowed.csv"
    pred_csv = workdir / "play_lgbm_pred.csv"
    smooth_features = workdir / "features_with_play_smooth.csv"
    smooth_segments = workdir / "smooth_segments.csv"

    # ---- しきい値の決定ロジック ----
    meta_thr = load_best_thr(Path(args.meta))
    if args.thr is not None:
        thr = args.thr
        print(f"[info] use thr from CLI: {thr}")
    elif meta_thr is not None:
        thr = meta_thr
        print(f"[info] use thr from meta: {thr} ({args.meta})")
    else:
        thr = 0.5
        print(f"[warn] no thr specified and no meta found; fallback thr={thr}")

    py = sys.executable  # venv の python を使う

    # ------------------------------------------------------------
    # 1) input.mp4 → features_input.csv
    #    segment CLI に --dump_features させる（＝ dump_feature 相当）
    # ------------------------------------------------------------
    run([
        py,
        "-m", "trc.cli.segment",        # trc/cli/segment.py をモジュール実行
        str(in_mp4),
        "--config", "configs/default.yml",
        "--dump_features", str(features_input),
        # 必要ならここに --pred_csv や --overlay も足せる
    ])

    # ------------------------------------------------------------
    # 2) features_input.csv → features_windowed.csv
    # ------------------------------------------------------------
    run([
        py,
        "scripts/make_window_features.py",
        str(features_input),
        str(features_window),
    ])

    # ------------------------------------------------------------
    # 3) LightGBM で p_play_raw を推論 → play_lgbm_pred.csv
    # ------------------------------------------------------------
    run([
        py,
        "scripts/predict_play_lgbm.py",
        "--model",
        str(args.model),
        "--out_csv",
        str(pred_csv),
        str(features_window),  # 入力CSVは位置引数で最後に渡す
    ])

    # ------------------------------------------------------------
    # 4) features_windowed.csv → 平滑化スコア & セグメント
    #     （features_to_smooth_segments.py に全部やらせる）
    # ------------------------------------------------------------
    run([
        py,
        "scripts/features_to_smooth_segments.py",
        "--model", str(args.model),                  # LGBMモデルを使って p_play を計算
        "--thr", str(thr),                           # しきい値（meta or CLI）
        "--min_len", str(args.min_len),              # 最短ラリー長
        "--smooth_csv", str(smooth_features),        # features_with_play_smooth.csv
        "--segments_csv", str(smooth_segments),      # smooth_segments.csv
        str(features_window),                        # ★ 最後に位置引数として features_csv を渡す
    ])

    # ------------------------------------------------------------
    # 5) segments + input.mp4 → rallies_smooth.mp4
    #    ※ 重いので、--no-video のときはスキップ
    # ------------------------------------------------------------
    if not args.no_video:
        run([
            py,
            "cut_segments.py",
            str(in_mp4),          # input.mp4
            str(smooth_segments), # pred_segments.csv 的なCSV (ここでは smooth_segments.csv)
            str(out_mp4),         # output.mp4
        ])

        print()
        print("✅ 完了:", out_mp4)
    else:
        print()
        print("✅ 動画出力をスキップしました (--no-video)")
        print(f"    features_with_play_smooth.csv: {smooth_features}")
        print(f"    smooth_segments.csv:           {smooth_segments}")


if __name__ == "__main__":
    main()

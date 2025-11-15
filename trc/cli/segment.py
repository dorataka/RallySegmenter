import argparse
import csv
import os

import numpy as np

from trc.config import load_config
from trc.features.motion import motion_series
from trc.features.audio import audio_series
from trc.features.pose import yolo_pose_motion
from trc.fuse import fuse_series
from trc.segment.hysteresis import hysteresis_segment


def _dump_features_csv(
    path: str,
    grid: np.ndarray,
    motion: np.ndarray,
    audio: np.ndarray,
    fused: np.ndarray,
    pose: np.ndarray | None = None,
):
    """
    grid 上の各種 series を 1 行ずつ CSV に書き出す。
    カラム: t, motion, audio, pose, fused
    """
    n = len(grid)
    m = len(motion)
    a = len(audio)
    f = len(fused)
    p = len(pose) if pose is not None else 0

    L = min(n, m, a, f, p if p > 0 else 10**9)

    t_arr = grid[:L]
    m_arr = motion[:L]
    a_arr = audio[:L]
    f_arr = fused[:L]
    if pose is not None and p > 0:
        p_arr = pose[:L]
    else:
        p_arr = np.zeros(L, dtype=np.float32)

    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "motion", "audio", "pose", "fused"])
        for t, mv, av, pv, fv in zip(t_arr, m_arr, a_arr, p_arr, f_arr):
            writer.writerow([f"{t:.3f}", f"{mv:.6f}", f"{av:.6f}", f"{pv:.6f}", f"{fv:.6f}"])

    print(f"[dump] features -> {os.path.abspath(path)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_mp4")
    ap.add_argument(
        "--config",
        default=os.path.join("configs", "default.yml"),
        help="設定ファイル（YAML）のパス",
    )
    ap.add_argument(
        "--pred_csv",
        default=os.path.join("runs", "pred_segments.csv"),
        help="予測セグメントを書き出す CSV パス",
    )
    ap.add_argument(
        "--overlay",
        type=str,
        default="",
        help="HUD を重ねたデバッグ動画の出力パス（空なら出さない）",
    )
    ap.add_argument(
        "--dump_features",
        type=str,
        default="",
        help="grid と特徴量 (motion/audio/pose/fused) を CSV で保存するパス",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    os.makedirs(os.path.dirname(args.pred_csv), exist_ok=True)

    # --- motion series ---
    mot, dt_m = motion_series(
        args.input_mp4,
        step=cfg.io.step,
        scale=cfg.io.scale,
        method=cfg.motion.method,
        smooth=cfg.motion.smooth,
        norm=cfg.motion.norm,
    )

    # --- audio series ---
    aud, dt_a = audio_series(
        args.input_mp4,
        method=cfg.audio.method,
        rms_win=cfg.audio.rms_win,
        hop=cfg.audio.hop,
        smooth=cfg.audio.smooth,
        bandpass=cfg.audio.bandpass,
    )

    if len(mot) == 0 or len(aud) == 0:
        print("Empty motion or audio series, no segments.")
        return

    # ---- ここから「時間軸ベースで揃える」 ----

    # それぞれの時間軸（秒）
    t_m = np.arange(len(mot), dtype=np.float32) * float(dt_m)
    t_a = np.arange(len(aud), dtype=np.float32) * float(dt_a)

    # グリッドの時間解像度は細かい方に合わせる
    dt = float(min(dt_m, dt_a))

    # 両方が定義されている範囲だけを使う（物理的に正しい範囲）
    T_end = float(min(t_m[-1], t_a[-1]))
    if T_end <= 0:
        print("Non-positive duration, no segments.")
        return

    grid = np.arange(0.0, T_end, dt, dtype=np.float32)

    # motion / audio を共通 grid に線形補間
    mot_on_grid = np.interp(grid, t_m, mot).astype(np.float32)
    aud_on_grid = np.interp(grid, t_a, aud).astype(np.float32)

    # デバッグ用に長さを一応表示
    print(f"[debug] len(mot_raw)={len(mot)}, dt_m={dt_m:.4f}, T_m={t_m[-1]:.2f}s")
    print(f"[debug] len(aud_raw)={len(aud)}, dt_a={dt_a:.4f}, T_a={t_a[-1]:.2f}s")
    print(f"[debug] len(grid)={len(grid)}, dt={dt:.4f}, T_end={T_end:.2f}s")

    # --- Pose motion (grid) ---
    pose = None
    if cfg.pose.enabled and (cfg.pose.weight > 0.0 or cfg.fuse.pose_weight > 0.0):
        print("[step] pose motion …")
        pm = yolo_pose_motion(
            args.input_mp4,
            grid,
            dt,
            step=cfg.pose.step,
            min_box_h=cfg.pose.min_box_h,
            kps_min_conf=cfg.pose.kps_min_conf,
            topk=cfg.pose.topk,
            smooth_k=cfg.pose.smooth_k,
            speed_low=cfg.pose.speed_low,
            speed_high=cfg.pose.speed_high,
            debug=False,
        )
        if pm is not None:
            # yolo_pose_motion は len(grid) の配列を返す想定
            pose = pm.astype(np.float32)
        else:
            pose = np.zeros_like(grid, dtype=np.float32)
    else:
        pose = None

    # --- fuse motion + audio (+ pose) ---
    fused = fuse_series(
        mot_on_grid,
        aud_on_grid,
        pose,
        w_m=cfg.fuse.motion_weight,
        # audio の重みは config 側の定義に合わせてどちらか使ってください
        # w_a=cfg.audio.weight,  # audio セクションに weight がある場合はこちら
        w_a=cfg.fuse.audio_weight,  # fuse セクションで重みを持っているならこちら
        w_p=cfg.fuse.pose_weight,
        clip=cfg.fuse.clip_range,
    )

    # --- segmentation ---
    segs = hysteresis_segment(
        fused,
        dt,
        cfg.segment.hyst_low,
        cfg.segment.hyst_high,
        cfg.segment.min_play,
        cfg.segment.min_gap,
        cfg.segment.snap_peak_sec,
    )

    # --- features CSV 出力（任意） ---
    if args.dump_features:
        _dump_features_csv(
            args.dump_features,
            grid=grid,
            motion=mot_on_grid,
            audio=aud_on_grid,
            fused=fused,
            pose=pose,
        )

    # --- overlay 出力（任意） ---
    if args.overlay:
        from trc.overlay import render_debug_overlay

        v_i = mot_on_grid    # grid 上の motion
        a_i = aud_on_grid    # grid 上の audio
        feat = fused         # grid 上の fused
        pose_series = pose if pose is not None else np.zeros_like(grid, dtype=np.float32)

        params = dict(
            # fuse weights（表示用）
            motion_weight=float(getattr(cfg.fuse, "motion_weight", 0.5)),
            audio_weight=float(getattr(cfg.fuse, "audio_weight", 0.2)),
            pose_weight=float(getattr(cfg.fuse, "pose_weight", 0.3)),
            # segment
            hyst_low=cfg.segment.hyst_low,
            hyst_high=cfg.segment.hyst_high,
            min_play=cfg.segment.min_play,
            min_gap=cfg.segment.min_gap,
            snap_peak_sec=cfg.segment.snap_peak_sec,
            # pose
            pose_enabled=bool(getattr(cfg.pose, "enabled", True)),
            pose_step=int(getattr(cfg.pose, "step", 3)),
            pose_smooth_k=int(getattr(cfg.pose, "smooth_k", 5)),
        )

        render_debug_overlay(
            args.input_mp4,
            args.overlay,
            grid,
            v_i,
            a_i,
            feat,
            pose_series,  # overlay 側では「walk_mask 引数」を pose series として使う
            params,
            step=2,   # 2フレームに1枚描画（軽量化）
        )
        print(f"[overlay] saved -> {os.path.abspath(args.overlay)}")

    # --- セグメント CSV 出力 ---
    with open(args.pred_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["start", "end"])
        for s, e in segs:
            wr.writerow([f"{s:.3f}", f"{e:.3f}"])

    print(f"Wrote {len(segs)} segments to {args.pred_csv}")


if __name__ == "__main__":
    main()

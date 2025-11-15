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

    # 長さを揃える
    L = min(len(mot), len(aud))
    mot = mot[:L]
    aud = aud[:L]
    dt = min(dt_m, dt_a)

    # --- Pose motion (grid) ---
    pose = None
    if cfg.pose.enabled and (cfg.pose.weight > 0.0 or cfg.fuse.pose_weight > 0.0):
        # motion/audio と同じ時間解像度の grid
        grid = np.arange(0.0, L * dt, dt, dtype=np.float32)
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
            pose = pm[:L]
        else:
            pose = np.zeros(L, dtype=np.float32)
    else:
        pose = None

    # --- fuse motion + audio (+ pose) ---
    fused = fuse_series(
        mot,
        aud,
        pose,
        w_m=cfg.fuse.motion_weight,
        # audio の重みは config 側の定義に合わせてどちらか使ってください
        # cfg.audio.weight があるなら下の行を使う:
        # w_a=cfg.audio.weight,
        # fuse.audio_weight で持っているならこちら:
        w_a=cfg.fuse.audio_weight,
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

    # --- overlay 出力（任意） ---
    if args.overlay:
        from trc.overlay import render_debug_overlay

        L = len(fused)
        grid = np.arange(0.0, L * dt, dt, dtype=np.float32)

        v_i = mot        # grid 上の motion
        a_i = aud        # grid 上の audio
        feat = fused     # grid 上の fused
        walk_mask = np.zeros_like(grid, dtype=np.float32)  # walk_mask はまだゼロ埋め

        params = dict(
            min_rally=cfg.segment.min_play,
            max_gap=cfg.segment.min_gap,
            penalty=0.0,
            pad_head=0.0,
            pad_tail=0.0,
            prune_min_hits=0,
            prune_min_len=0.0,
            pose_nonplay=getattr(cfg.pose, "enabled", False),
            pose_gait_thr=getattr(cfg.pose, "kps_min_conf", 0.2),
            pose_min_walk=0.0,
            pose_topk=getattr(cfg.pose, "topk", 1),
        )

        render_debug_overlay(
            args.input_mp4,
            args.overlay,
            grid,
            v_i,
            a_i,
            feat,
            walk_mask,
            params,
            step=2,   # 2フレームに1枚描画（軽量化）
        )
        print(f"[overlay] saved -> {os.path.abspath(args.overlay)}")

    # --- CSV 出力 ---
    with open(args.pred_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["start", "end"])
        for s, e in segs:
            wr.writerow([f"{s:.3f}", f"{e:.3f}"])

    print(f"Wrote {len(segs)} segments to {args.pred_csv}")


if __name__ == "__main__":
    main()

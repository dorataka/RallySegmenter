from __future__ import annotations
import os
import cv2
import numpy as np
from tqdm import tqdm

__all__ = ["render_debug_overlay"]

# ---- 内部ユーティリティ ----
def _interp_on_grid(grid: np.ndarray, series: np.ndarray, t: float) -> float:
    if grid.size == 0 or series.size == 0:
        return 0.0
    t = float(np.clip(t, grid[0], grid[-1]))
    return float(np.interp(t, grid, series))

def _draw_hbar(img, x, y, w, h, val, label):
    # val in [0,1]
    val = float(np.clip(val, 0.0, 1.0))

    # 背景 & 枠
    cv2.rectangle(img, (x, y), (x+w, y+h), (60, 60, 60), 1)
    cv2.rectangle(img, (x, y), (x+int(w*val), y+h), (200, 200, 200), -1)

    # テキストをバー内に配置（下端から少し上）
    text = f"{label}: {val:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    tx = x + 6                       # 左から少し内側
    ty = y + h - 5                   # バーの下縁より5px上（=文字の下端）
    # 文字が読みにくい場面向けに薄い影を入れる（任意）
    cv2.putText(img, text, (tx+1, ty+1), font, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty),     font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

def _draw_params_block(img, lines, x=10, y=10, pad=8):
    # サイズ見積り
    w = 0
    h = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        w = max(w, tw)
        h += th + 6
    W = w + pad * 2
    H = h + pad * 2
    # 背景パネル
    cv2.rectangle(img, (x, y), (x + W, y + H), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + W, y + H), (90, 90, 90), 1)
    # テキスト
    cy = y + pad + 14
    for line in lines:
        cv2.putText(
            img,
            line,
            (x + pad, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cy += 20

def _draw_walkmask_value(vis, t_sec, grid, walk_mask, x=20, y=120):
    """
    vis: BGRフレーム
    t_sec: 現在フレームの絶対秒（frame_idx / fps）
    grid: walk_mask と同じ長さの時間配列
    walk_mask: 長さ len(grid) の 0..1 配列
    """
    if (walk_mask is None) or (grid is None):
        return
    if len(walk_mask) == 0 or len(grid) != len(walk_mask):
        return

    val = float(np.interp(t_sec, grid, walk_mask))  # 現在時刻に対応する値
    cv2.putText(vis, f"WalkMask={val:.2f}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # 0..1 の横バー
    bar_w, bar_h = 180, 8
    x0, y0 = x, y + 10
    cv2.rectangle(vis, (x0, y0), (x0+bar_w, y0+bar_h), (0,80,0), 1)
    cv2.rectangle(vis, (x0, y0), (x0+int(bar_w*max(0.0, min(1.0, val))), y0+bar_h), (0,255,0), -1)

# ---- メイン ----
def render_debug_overlay(
    in_mp4: str,
    out_mp4: str,
    grid: np.ndarray,
    v_i: np.ndarray,
    a_i: np.ndarray,
    feat: np.ndarray,
    walk_mask: np.ndarray | None,
    params: dict,
    step: int = 2,
):
    """
    入力動画 in_mp4 にHUDを重ねて out_mp4 へ出力。
    grid 上にある各種系列（v_i, a_i, feat, walk_mask）を現在時刻へ線形補間して下部バー表示。
    step: 何フレームごとに1枚描画するか（大きいほど軽く、出力fpsは低くなる）
    """
    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for overlay")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(step))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    pbar = tqdm(total=total_frames, desc="[overlay]", unit="f") 

    # 出力先ディレクトリがなければ作成
    out_dir = os.path.dirname(os.path.abspath(out_mp4))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    vw = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(10.0, fps / step),  # 表示用に最低10fps
        (W, H),
    )
    if not vw.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter for overlay")

    # paramsの安全な取り出し（欠けても落ちないように）
    get = params.get
    min_rally = float(get("min_rally", 2.0))
    max_gap   = float(get("max_gap", 0.6))
    penalty   = float(get("penalty", 12.0))
    pad_head  = float(get("pad_head", 0.20))
    pad_tail  = float(get("pad_tail", 0.35))
    prune_min_hits = int(get("prune_min_hits", 3))
    prune_min_len  = float(get("prune_min_len", 1.8))
    pose_nonplay   = bool(get("pose_nonplay", False))
    pose_gait_thr  = float(get("pose_gait_thr", 0.35))
    pose_min_walk  = float(get("pose_min_walk", 1.0))
    pose_topk      = int(get("pose_topk", 2))

    idx = 0
    zeros_like_grid = np.zeros_like(grid, dtype=np.float32)

    while True:
        ret = cap.grab()
        if not ret:
            break

        pbar.update(1)

        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / fps

            # 値の取得（Noneガード込み）
            mv = _interp_on_grid(grid, v_i if v_i is not None else zeros_like_grid, t)
            av = _interp_on_grid(grid, a_i if a_i is not None else zeros_like_grid, t)
            fv = _interp_on_grid(grid, feat if feat is not None else zeros_like_grid, t)
            wm_series = walk_mask if walk_mask is not None else zeros_like_grid
            wm = _interp_on_grid(grid, wm_series, t)

            # 下部に水平バー（高さが小さい映像でもはみ出ないように計算）
            bars = 4
            bar_h = max(12, min(24, H // 40))
            gap_v = bar_h + 6
            total_h = bars * gap_v + 20
            y0 = max(0, H - total_h)
            x0 = 14
            w = max(40, W - 28)
            _draw_hbar(frame, x0, y0 + 0 * gap_v, w, bar_h, mv, "Motion")
            _draw_hbar(frame, x0, y0 + 1 * gap_v, w, bar_h, av, "Audio")
            _draw_hbar(frame, x0, y0 + 2 * gap_v, w, bar_h, fv, "Fused")
            _draw_hbar(frame, x0, y0 + 3 * gap_v, w, bar_h, wm, "WalkMask")

            # 左上にパラメータ群（固定）
            lines = [
                f"t={t:7.2f}s  fps={fps:.2f}",
                f"min_rally={min_rally:.2f}  max_gap={max_gap:.2f}",
                f"penalty={penalty:.2f}  pad_head={pad_head:.2f}  pad_tail={pad_tail:.2f}",
                f"prune(min_hits={prune_min_hits}, min_len={prune_min_len:.2f})",
                f"pose_nonplay={pose_nonplay} gait_thr={pose_gait_thr:.2f} "
                f"min_walk={pose_min_walk:.2f} topK={pose_topk}",
            ]
            _draw_params_block(frame, lines, x=10, y=10)

            vw.write(frame)

        idx += 1

    cap.release()
    vw.release()
    pbar.close() 
    return out_mp4

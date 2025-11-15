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
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    cv2.rectangle(img, (x, y), (x + int(w * val), y + h), (200, 200, 200), -1)

    # テキストをバー内に配置（下端から少し上）
    text = f"{label}: {val:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    tx = x + 6                      # 左から少し内側
    ty = y + h - 5                  # バーの下縁より5px上（=文字の下端）
    # 読みやすさ確保のための軽い影
    cv2.putText(img, text, (tx + 1, ty + 1), font, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, font_scale,
                (240, 240, 240), thickness, cv2.LINE_AA)


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


# ---- メイン ----
def render_debug_overlay(
    in_mp4: str,
    out_mp4: str,
    grid: np.ndarray,
    v_i: np.ndarray,          # motion series
    a_i: np.ndarray,          # audio series
    feat: np.ndarray,         # fused score
    walk_mask: np.ndarray | None,  # ← ここを pose series として再利用
    params: dict,
    step: int = 2,
):
    """
    入力動画 in_mp4 に HUD を重ねて out_mp4 へ出力する。
    grid 上にある各種系列（motion, audio, pose, fused）を現在時刻へ線形補間して
    下部バー表示する。

    Parameters
    ----------
    in_mp4 : str
        入力動画パス
    out_mp4 : str
        出力動画パス
    grid : np.ndarray
        時間軸（秒）の等間隔配列
    v_i : np.ndarray
        motion 系列（0〜1 を想定）
    a_i : np.ndarray
        audio 系列（0〜1 を想定）
    feat : np.ndarray
        fused 系列（0〜1 を想定）
    walk_mask : np.ndarray | None
        現在は walk ではなく「pose 系列」として扱う（0〜1）
    params : dict
        デバッグ用のパラメータ表示に用いる辞書
    step : int
        何フレームごとに 1 枚描画するか（大きいほど軽い）
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
        max(10.0, fps / step),  # 表示用に最低 10fps
        (W, H),
    )
    if not vw.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter for overlay")

    # params の安全な取り出し
    params = params or {}
    get = params.get

    # fuse の重み（default.yml ベースでフォールバック）
    motion_w = float(get("motion_weight", get("fuse_motion_weight", 0.50)))
    audio_w  = float(get("audio_weight",  get("fuse_audio_weight",  0.20)))
    pose_w   = float(get("pose_weight",   get("fuse_pose_weight",   0.30)))

    # segment 関連
    hyst_low  = float(get("hyst_low",  0.30))
    hyst_high = float(get("hyst_high", 0.38))
    min_play  = float(get("min_play",  get("min_rally", 1.4)))
    min_gap   = float(get("min_gap",   get("max_gap",   0.8)))
    snap_peak = float(get("snap_peak_sec", 0.3))

    # pose 関連
    pose_enabled  = bool(get("pose_enabled", True))
    pose_step     = int(get("pose_step", 3))
    pose_smooth_k = int(get("pose_smooth_k", 5))

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

            # 値の取得（None ガード込み）
            mv = _interp_on_grid(grid, v_i if v_i is not None else zeros_like_grid, t)
            av = _interp_on_grid(grid, a_i if a_i is not None else zeros_like_grid, t)
            fv = _interp_on_grid(grid, feat if feat is not None else zeros_like_grid, t)
            pose_series = walk_mask if walk_mask is not None else zeros_like_grid
            pv = _interp_on_grid(grid, pose_series, t)

            # 下部に水平バーを 4 本（Motion / Audio / Pose / Fused）
            bars = 4
            bar_h = max(12, min(24, H // 40))
            gap_v = bar_h + 6
            total_h = bars * gap_v + 20
            y0 = max(0, H - total_h)
            x0 = 14
            w = max(40, W - 28)

            _draw_hbar(frame, x0, y0 + 0 * gap_v, w, bar_h, mv, "Motion")
            _draw_hbar(frame, x0, y0 + 1 * gap_v, w, bar_h, av, "Audio")
            _draw_hbar(frame, x0, y0 + 2 * gap_v, w, bar_h, pv, "Pose")
            _draw_hbar(frame, x0, y0 + 3 * gap_v, w, bar_h, fv, "Fused")

            # 左上にパラメータ群（固定）
            lines = [
                f"t={t:7.2f}s  fps={fps:.2f}",
                f"fuse: motion={motion_w:.2f}, audio={audio_w:.2f}, pose={pose_w:.2f}",
                f"segment: hyst=({hyst_low:.2f},{hyst_high:.2f}) "
                f"min_play={min_play:.2f}  min_gap={min_gap:.2f}",
                f"snap_peak={snap_peak:.2f}s",
                f"pose: enabled={pose_enabled} step={pose_step} smooth_k={pose_smooth_k}",
            ]
            _draw_params_block(frame, lines, x=10, y=10)

            vw.write(frame)

        idx += 1

    cap.release()
    vw.release()
    pbar.close()
    return out_mp4

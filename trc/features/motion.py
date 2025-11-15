import cv2
import numpy as np
from tqdm import tqdm

def robust_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    p1, p99 = np.percentile(x, [1, 99])
    denom = max(p99 - p1, 1e-6)
    z = (x - p1) / denom
    return np.clip(z, 0.0, 1.0)


def motion_series(
    video_path: str,
    step: int = 1,
    scale: float = 0.5,
    method: str = "flow",
    smooth: int = 7,
    norm: str = "robust",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0  # ★ 全フレーム数

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return np.zeros(0, dtype=np.float32), 1.0 / fps

    H = int(frame.shape[0] * scale)
    W = int(frame.shape[1] * scale)
    prev = cv2.resize(frame, (W, H))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    vals = []

    # ★ プログレスバーを用意（最初の1フレームはすでに読んでいるので initial=1）
    pbar = tqdm(
        total=total_frames if total_frames > 0 else None,
        initial=1,
        desc="[motion]",
        unit="f",
    )

    while True:
        for _ in range(step):
            ok, frame = cap.read()
            if not ok:
                break
            pbar.update(1)  # ★ フレームを読むたびに進捗を1進める
        if not ok:
            break

        frame = cv2.resize(frame, (W, H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == "flow":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.linalg.norm(flow, axis=2).mean()
            vals.append(mag)
        else:
            diff = cv2.absdiff(prev_gray, gray)
            vals.append(float(diff.mean()))

        prev_gray = gray

    pbar.close()   # ★ 終了時に閉じる
    cap.release()

    x = np.asarray(vals, dtype=np.float32)
    if x.size and smooth > 1:
        k = max(1, smooth | 1)  # 奇数に
        x = cv2.blur(x[:, None], (k, 1)).ravel()

    if x.size:
        if norm == "robust":
            x = robust_norm(x)
        else:
            x = (x - x.min()) / max(x.ptp(), 1e-6)

    dt = step / fps
    return x, dt

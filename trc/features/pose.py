import os
import json
import cv2
import numpy as np

__all__ = ["yolo_pose_motion"]

ROI_JSON = "court_roi.json"


# ---- ROI を管理するユーティリティ（near / far の2枚） ----

def _load_or_ask_rois(frame):
    """
    frame: 最初のフレーム (H,W,3)

    - すでに court_roi.json があり、解像度が一致すれば near / far の 2 ROI を使う
    - なければ cv2.selectROI でユーザーに
        1) 手前プレイヤー用 ROI
        2) 奥側プレイヤー用 ROI
      を順番に囲ってもらい、保存する

    戻り値: (near_roi, far_roi)
        near_roi, far_roi はそれぞれ (x, y, w, h) or None
    """
    h, w = frame.shape[:2]

    # 既存 ROI があれば使う
    if os.path.exists(ROI_JSON):
        try:
            with open(ROI_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("width") == w and data.get("height") == h:
                near_roi = data.get("near_roi", None)
                far_roi = data.get("far_roi", None)
                if near_roi and far_roi:
                    print(f"[roi] loaded near/far ROI from {ROI_JSON}")
                    return (
                        tuple(int(v) for v in near_roi),
                        tuple(int(v) for v in far_roi),
                    )
        except Exception as e:
            print(f"[roi] failed to load {ROI_JSON}: {e}")

    # なければユーザーに 2 回選んでもらう
    print("[roi] No near/far ROI found. Showing a frame to select them …")

    # near ROI（手前プレイヤー）
    tmp = frame.copy()
    cv2.putText(
        tmp,
        "Drag NEAR PLAYER region (court near side), ENTER/SPACE to confirm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    r_near = cv2.selectROI("Select NEAR ROI", tmp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select NEAR ROI")
    x_n, y_n, w_n, h_n = map(int, r_near)
    if w_n <= 0 or h_n <= 0:
        print("[roi] NEAR ROI not selected, continue without ROI.")
        near_roi = None
    else:
        near_roi = (x_n, y_n, w_n, h_n)
        print(f"[roi] selected NEAR ROI: {near_roi}")

    # far ROI（奥側プレイヤー）
    tmp = frame.copy()
    cv2.putText(
        tmp,
        "Drag FAR PLAYER region (court far side), ENTER/SPACE to confirm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    r_far = cv2.selectROI("Select FAR ROI", tmp, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select FAR ROI")
    x_f, y_f, w_f, h_f = map(int, r_far)
    if w_f <= 0 or h_f <= 0:
        print("[roi] FAR ROI not selected, continue without FAR ROI.")
        far_roi = None
    else:
        far_roi = (x_f, y_f, w_f, h_f)
        print(f"[roi] selected FAR ROI: {far_roi}")

    # 保存
    try:
        data = {
            "width": w,
            "height": h,
            "near_roi": list(near_roi) if near_roi else None,
            "far_roi": list(far_roi) if far_roi else None,
        }
        with open(ROI_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[roi] saved near/far ROI to {ROI_JSON}")
    except Exception as e:
        print(f"[roi] failed to save {ROI_JSON}: {e}")

    return near_roi, far_roi


# ---- 簡易マルチオブジェクトトラッカー（IOU ベース） ----

class _Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1,y1,x2,y2]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0


class SimpleTracker:
    """
    SORT 風の簡易トラッカー。
    Kalman は省略して IOU ベースの貪欲マッチングだけにした軽量版。
    """

    def __init__(self, iou_thr=0.3, max_age=30, min_hits=3):
        self.iou_thr = iou_thr
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 1

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        inter = w * h
        if inter <= 0:
            return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    def update(self, dets):
        """
        dets: ndarray (N,5) [x1,y1,x2,y2,score]
        戻り値: list of (x1,y1,x2,y2,track_id,det_idx)
        """
        dets = np.asarray(dets, dtype=np.float32)
        n_det = len(dets)

        # 既存 track がなければ全部新規
        if len(self.tracks) == 0:
            for j in range(n_det):
                box = dets[j, :4]
                tr = _Track(self._next_id, box)
                self._next_id += 1
                self.tracks.append(tr)
            return [
                (tr.bbox[0], tr.bbox[1], tr.bbox[2], tr.bbox[3], tr.id, idx)
                for idx, tr in enumerate(self.tracks)
                if tr.hits >= self.min_hits
            ]

        # IOU 行列
        track_boxes = np.array([t.bbox for t in self.tracks], dtype=np.float32)
        iou_mat = np.zeros((len(self.tracks), n_det), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j in range(n_det):
                iou_mat[i, j] = self._iou(tb, dets[j, :4])

        matched_tr = set()
        matched_det = set()
        matches = []

        # 貪欲マッチング
        while True:
            if iou_mat.size == 0:
                break
            i, j = divmod(iou_mat.argmax(), iou_mat.shape[1])
            if iou_mat[i, j] < self.iou_thr:
                break
            if i in matched_tr or j in matched_det:
                iou_mat[i, j] = -1.0
                continue
            matched_tr.add(i)
            matched_det.add(j)
            matches.append((i, j))
            iou_mat[i, :] = -1.0
            iou_mat[:, j] = -1.0

        # マッチした track を更新
        for i, j in matches:
            tr = self.tracks[i]
            tr.bbox = dets[j, :4]
            tr.age = 0
            tr.time_since_update = 0
            tr.hits += 1

        # マッチしなかった track を age++
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_tr:
                tr.age += 1
                tr.time_since_update += 1

        # 新しい det から track 生成
        for j in range(n_det):
            if j in matched_det:
                continue
            box = dets[j, :4]
            tr = _Track(self._next_id, box)
            self._next_id += 1
            self.tracks.append(tr)

        # 古くなった track を削除
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # 出力: hits が一定以上の track だけ返す
        out = []
        for tr in self.tracks:
            if tr.hits < self.min_hits:
                continue
            best_j = -1
            best_iou = 0.0
            for j in range(n_det):
                iou = self._iou(tr.bbox, dets[j, :4])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0:
                out.append(
                    (tr.bbox[0], tr.bbox[1], tr.bbox[2], tr.bbox[3], tr.id, best_j)
                )

        return out


# ---- Pose Motion with tracking + near/far 2ROI ----

def yolo_pose_motion(
    in_mp4,
    grid,
    grid_dt,
    *,
    step=3,
    min_box_h=32,
    kps_min_conf=0.2,
    topk=2,          # シグネチャ維持のため残すが内部では使っていない
    smooth_k=5,
    speed_low=0.04,
    speed_high=0.18,
    debug=False,
):
    """
    YOLOv8-Pose + 簡易トラッキング + near/far の 2 ROI で
    「プレイヤーの動き量」シリーズを返す。

    - 最初のフレームを表示して
        1) 手前プレイヤーが存在しうる範囲（near）
        2) 奥側プレイヤーが存在しうる範囲（far）
      を別々に囲ってもらう（長方形で OK）
    - bbox 中心が near ROI / far ROI のどちらかに入る人物だけを対象とし、
      それ以外（隣コート・観客等）は無視する。
    - near をメイン（w_near=0.8）、far をサブ（w_far=0.2）として融合。
      far がほとんど検出されない動画では自動で w_far=0.0 に落とす。

    戻り値:
        pose_series: np.ndarray shape (len(grid),), 値は [0,1] に正規化された動き量
        失敗した場合は None
    """
    try:
        from ultralytics import YOLO
    except Exception:
        return None

    cap = cv2.VideoCapture(in_mp4)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or (1.0 / grid_dt)
    fps = float(fps)

    # 最初のフレームで near/far ROI を決める
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return None

    H, W = frame0.shape[:2]
    near_roi, far_roi = _load_or_ask_rois(frame0)  # それぞれ (x,y,w,h) or None

    # 再スタート
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # モデル読み込み（pose 用）
    try:
        model = YOLO("yolov8n-pose.pt")  # 軽量 pose モデル。必要なら変更。
    except Exception:
        cap.release()
        return None

    tracker = SimpleTracker(
        iou_thr=0.3, max_age=int(fps * 1.0), min_hits=3
    )

    n_grid = len(grid)
    near_series = np.zeros(n_grid, dtype=np.float32)
    far_series = np.zeros(n_grid, dtype=np.float32)

    # track_id -> 前フレームの keypoints
    prev_kps = {}

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps
        if t < grid[0] - grid_dt or t > grid[-1] + grid_dt:
            frame_idx += 1
            continue

        # YOLO-Pose
        res = model(frame, verbose=False)
        if len(res) == 0:
            frame_idx += 1
            continue
        r0 = res[0]

        if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
            frame_idx += 1
            continue

        kpts = r0.keypoints.data.cpu().numpy()    # (N, K, 3)
        boxes_xyxy = r0.boxes.xyxy.cpu().numpy()  # (N,4)
        scores = r0.boxes.conf.cpu().numpy()      # (N,)

        if boxes_xyxy.size == 0:
            frame_idx += 1
            continue

        dets = np.hstack([boxes_xyxy, scores[:, None]])  # (N,5)

        # tracking
        tracks = tracker.update(dets)  # list of (x1,y1,x2,y2,track_id,det_idx)
        if not tracks:
            frame_idx += 1
            continue

        near_speed = 0.0
        far_speed = 0.0

        for (x1, y1, x2, y2, tid, det_idx) in tracks:
            box_h = y2 - y1
            if box_h < min_box_h:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # near/far ROI 判定
            in_near = False
            in_far = False
            if near_roi is not None:
                xn, yn, wn, hn = near_roi
                if xn <= cx <= xn + wn and yn <= cy <= yn + hn:
                    in_near = True
            if far_roi is not None:
                xf, yf, wf, hf = far_roi
                if xf <= cx <= xf + wf and yf <= cy <= yf + hf:
                    in_far = True

            if not in_near and not in_far:
                # コート外（隣コート・観客など）
                continue

            # 対応する keypoints
            if det_idx < 0 or det_idx >= len(kpts):
                continue
            kp = kpts[int(det_idx)]  # (K,3) [x,y,conf]

            prev = prev_kps.get(tid, None)
            if prev is None:
                prev_kps[tid] = kp
                continue

            conf_mask = (kp[:, 2] >= kps_min_conf) & (prev[:, 2] >= kps_min_conf)
            if not np.any(conf_mask):
                prev_kps[tid] = kp
                continue

            dx = kp[conf_mask, 0] - prev[conf_mask, 0]
            dy = kp[conf_mask, 1] - prev[conf_mask, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            speed = float(np.mean(dist))  # フレームあたりの平均移動量

            # near / far ごとに最大速度を保持
            if in_near:
                near_speed = max(near_speed, speed)
            if in_far:
                far_speed = max(far_speed, speed)

            prev_kps[tid] = kp

        # grid index に投影
        idx = int(round((t - grid[0]) / grid_dt))
        if 0 <= idx < n_grid:
            near_series[idx] = max(near_series[idx], near_speed)
            far_series[idx] = max(far_series[idx], far_speed)

        frame_idx += 1

    cap.release()

    # ---- 欠損補間 & 平滑化 ----
    def _smooth_fill(series, max_gap_frames=5, smooth_k=5):
        s = series.copy()
        n = len(s)

        last_val = 0.0
        gap = 0
        for i in range(n):
            if s[i] > 0:
                last_val = s[i]
                gap = 0
            else:
                gap += 1
                if 0 < gap <= max_gap_frames:
                    s[i] = last_val
                else:
                    last_val = 0.0

        if smooth_k > 1:
            k = int(max(1, smooth_k))
            window = np.ones(k, dtype=np.float32) / float(k)
            s = np.convolve(s, window, mode="same")

        return s

    max_gap_frames = int(0.5 / grid_dt)  # 0.5秒くらいまでなら前値で埋める
    near_series = _smooth_fill(near_series, max_gap_frames=max_gap_frames, smooth_k=smooth_k)
    far_series = _smooth_fill(far_series, max_gap_frames=max_gap_frames, smooth_k=smooth_k)

    # ---- 近側 / 遠側の重み付き合成 ----
    w_near = 0.8
    w_far = 0.2

    # 奥側がほとんど検出されていないなら w_far を 0 にする
    if np.count_nonzero(far_series > 1e-4) < 0.1 * n_grid:
        w_far = 0.0

    pose_raw = w_near * near_series + w_far * far_series

    # ---- 正規化 ----
    if speed_high <= speed_low:
        speed_low, speed_high = 0.0, max(1.0, float(np.max(pose_raw) + 1e-6))

    pose_norm = (pose_raw - speed_low) / (speed_high - speed_low)
    pose_norm = np.clip(pose_norm, 0.0, 1.0).astype(np.float32)

    if debug:
        print(
            f"[pose] near>0: {np.count_nonzero(near_series > 0)}, "
            f"far>0: {np.count_nonzero(far_series > 0)}, "
            f"w_far={w_far}, "
            f"near_roi={near_roi}, far_roi={far_roi}"
        )

    return pose_norm

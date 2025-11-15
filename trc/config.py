from dataclasses import dataclass
from typing import Optional, Tuple
import yaml


@dataclass
class IOConf:
    scale: float = 0.5     # 映像の縮小率
    step: int = 1          # 何フレームごとにサンプリングするか


@dataclass
class MotionConf:
    method: str = "flow"   # "flow" or "diff"
    smooth: int = 7
    norm: str = "robust"   # "robust" or "minmax"


@dataclass
class AudioConf:
    method: str = "flux"      # "flux" or "rms"
    rms_win: int = 2048
    hop: int = 512
    smooth: int = 9
    bandpass: Optional[Tuple[Optional[int], Optional[int]]] = None
    weight: float = 0.35      # fuse 時の audio 重み


@dataclass
class PoseConf:
    enabled: bool = False     # Pose を使うかどうか
    weight: float = 0.0       # fuse 時の pose 重み
    step: int = 3
    min_box_h: int = 32
    kps_min_conf: float = 0.2
    topk: int = 1
    smooth_k: int = 5
    speed_low: float = 0.04
    speed_high: float = 0.18


@dataclass
class FuseConf:
    motion_weight: float = 0.65
    audio_weight: float = 0.35
    pose_weight: float = 0.0
    clip_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class SegmentConf:
    hyst_low: float = 0.12
    hyst_high: float = 0.22
    min_play: float = 1.4   # 秒
    min_gap: float = 0.8    # 秒
    snap_peak_sec: float = 0.3


@dataclass
class EvalConf:
    tolerance_sec: float = 0.5
    iou_thr: float = 0.1


@dataclass
class Config:
    io: IOConf
    motion: MotionConf
    audio: AudioConf
    pose: PoseConf
    fuse: FuseConf
    segment: SegmentConf
    eval: EvalConf


def _subdict(d: dict, key: str) -> dict:
    v = d.get(key, {})
    return v if isinstance(v, dict) else {}


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    return Config(
        io=IOConf(**_subdict(y, "io")),
        motion=MotionConf(**_subdict(y, "motion")),
        audio=AudioConf(**_subdict(y, "audio")),
        pose=PoseConf(**_subdict(y, "pose")),
        fuse=FuseConf(**_subdict(y, "fuse")),
        segment=SegmentConf(**_subdict(y, "segment")),
        eval=EvalConf(**_subdict(y, "eval")),
    )

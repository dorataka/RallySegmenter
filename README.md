# RallySegmenter

Tennis rally segmentation from a single fixed camera video.

RallySegmenter detects and cuts **actual rallies** from a long tennis video (smartphone, broadcast-like fixed camera, etc.).  
It fuses multiple signals (motion, audio, pose, etc.) and optionally uses machine learning (Logistic Regression / LightGBM) to obtain stable rally segments and evaluate them against ground-truth.

> ⚠️ This project is currently under active development.  
> APIs, CLI options, and directory structure may change without notice.

---

## Features

- 🎾 **Automatic rally detection**
  - Detects intervals where a real rally is being played
  - Designed for fixed camera videos (baseline or side-view)

- 🧠 **Multi-modal features**
  - Motion (frame differences, optical-like motion features)
  - Audio intensity / statistics
  - Pose-based features (YOLOv8 pose, walk mask, etc.) – WIP

- 📊 **Evaluation tools**
  - Compare predicted segments with ground-truth segments
  - Compute metrics such as F1 score
  - JSON/CSV-based stats for further analysis

- 🛠️ **Command line interface**
  - Simple CLI to run full pipeline on a video
  - Options for padding, gap merging, thresholds, etc.

---

## Requirements

- Python 3.10+ (currently developed on Windows + Python 3.11)
- ffmpeg (must be available in `PATH`)
- Typical Python packages:
  - `numpy`, `pandas`
  - `scikit-learn`, `lightgbm`
  - `opencv-python`
  - `librosa`
  - `moviepy`
  - `PyYAML`
  - `ultralytics` (for YOLOv8 pose, if pose features are used)
- A GPU is recommended for YOLOv8 pose, but not strictly required for basic motion+audio pipeline.

> Check `pyproject.toml` for the exact list of dependencies used in this repository.

---

## Installation

```bash
# clone this repository
git clone https://github.com/dorataka/RallySegmenter.git
cd RallySegmenter

# (optional but recommended) create venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# or on Unix-like systems# RallySegmenter

Tennis rally segmentation from a single fixed camera video.

RallySegmenter detects and cuts **actual rallies** from a long tennis video (smartphone, broadcast-like fixed camera, etc.).  
It fuses multiple signals (motion, audio, pose, etc.) and optionally uses machine learning (Logistic Regression / LightGBM) to obtain stable rally segments and evaluate them against ground-truth.

> ⚠️ This project is currently under active development.  
> APIs, CLI options, and directory structure may change without notice.

---

## Features

- 🎾 **Automatic rally detection**
  - Detects intervals where a real rally is being played
  - Designed for fixed camera videos (baseline or side-view)

- 🧠 **Multi-modal features**
  - Motion (frame differences, optical-like motion features)
  - Audio intensity / statistics
  - Pose-based features (YOLOv8 pose, walk mask, etc.) – WIP

- 📊 **Evaluation tools**
  - Compare predicted segments with ground-truth segments
  - Compute metrics such as F1 score
  - JSON/CSV-based stats for further analysis

- 🛠️ **Command line interface**
  - Simple CLI to run full pipeline on a video
  - Options for padding, gap merging, thresholds, etc.

---

## Requirements

- Python 3.10+ (currently developed on Windows + Python 3.11)
- ffmpeg (must be available in `PATH`)
- Typical Python packages:
  - `numpy`, `pandas`
  - `scikit-learn`, `lightgbm`
  - `opencv-python`
  - `librosa`
  - `moviepy`
  - `PyYAML`
  - `ultralytics` (for YOLOv8 pose, if pose features are used)
- A GPU is recommended for YOLOv8 pose, but not strictly required for basic motion+audio pipeline.

> Check `pyproject.toml` for the exact list of dependencies used in this repository.

---

## Installation

```bash
# clone this repository
git clone https://github.com/dorataka/RallySegmenter.git
cd RallySegmenter

# (optional but recommended) create venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# or on Unix-like systems
# source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install dependencies
# (option 1) editable install if pyproject.toml is configured as a standard project
pip install -e .

# (option 2) or install dependencies manually, e.g.
# pip install numpy pandas scikit-learn lightgbm opencv-python librosa moviepy pyyaml ultralytics
Make sure ffmpeg is installed and accessible from the command line:

bash
コードをコピーする
ffmpeg -version
Basic Usage
1. Run segmentation (classical / non-ML pipeline)
Example:

bash
コードをコピーする
python -m trc.cli.segment \
  input.mp4 \
  --pred_csv runs/input.pred_segments.csv \
  --overlay runs/input_overlay.mp4 \
  --pad_head 0.60 \
  --pad_tail 0.15 \
  --max_gap 1.45 \
  --penalty 10
This will:

Load input.mp4

Extract motion/audio (and optionally pose) features

Fuse them to obtain a 1D “rally score” over time

Detect change points / segments

Output:

runs/input.pred_segments.csv – predicted rally segments

runs/input_overlay.mp4 – debug overlay video (score, mask, etc.), if enabled

2. Run segmentation with LightGBM frame classifier (experimental)
If you have already trained a LightGBM frame classifier:

bash
コードをコピーする
python -m trc.cli.segment_lgbm \
  input.mp4 \
  --model stats/frame_lgbm_model.txt \
  --pred_csv stats/pred_lgbm.csv \
  --overlay stats/overlay_lgbm.mp4 \
  --thr 0.80 \
  --min-len 1.0 \
  --max-gap 0.4
--model : path to trained LightGBM model

--thr : probability threshold for “rally frame”

--min-len : minimum segment length (seconds)

--max-gap : maximum gap between segments to be merged (seconds)

Note: CLI options and names are still subject to change while the project evolves.

Ground-truth format
Ground-truth rally segments are typically stored as a CSV file, e.g.:

text
コードをコピーする
start,end
6.00,11.00
15.00,26.00
54.00,58.00
61.00,65.00
...
start / end are given in seconds from the beginning of the video.

No header or a simple start,end header is assumed, depending on the evaluation script.

Evaluation
Use analyze_rally_params.py to compute metrics such as F1 score and to analyze parameter settings.

Example:

bash
コードをコピーする
python analyze_rally_params.py \
  --gt gt_segments.csv \
  --pred runs/input.pred_segments.csv \
  --stats stats/input.json
--gt : ground-truth segments (CSV)

--pred : predicted segments from the segmentation script

--stats : output JSON containing evaluation metrics and analysis

Project Structure
A rough overview of the directory layout:

text
コードをコピーする
RallySegmenter/
├── configs/               # YAML config files for experiments / defaults
├── scripts/               # Utility scripts, helpers
├── stats/                 # Models, metrics, intermediate CSV/JSON (often ignored in git)
├── trc/                   # Core package: features, fusion, CLI, etc.
│   ├── cli/               # Command line interfaces (segment, segment_lgbm, ...)
│   ├── features/          # Feature extraction (motion, audio, pose, walk mask, ...)
│   ├── models/            # ML models and related utilities (WIP)
│   └── ...                # Other utilities
├── Rallysegmenter.py      # Legacy entry point / compatibility script (may be refactored)
├── analyze_rally_params.py
├── cut_segments.py        # Cut rally segments out of original video
├── court_roi.json         # ROI definition for the court (if used)
├── pyproject.toml
└── README.md
Note: Some of the above paths are still evolving and may not exactly match the current implementation.

Roadmap / TODO
 Stabilize CLI interface and config system

 Improve pose-based features and walk mask

 Add more robust evaluation scripts and visualizations

 Prepare example dataset and ground-truth for public demonstration

 Add unit tests and CI

 Decide and add an open-source license (MIT/BSD/etc.)

License
No explicit license has been specified yet.
Please add a proper license file (LICENSE) before redistributing or using this code in production.

Notes (Japanese)
このリポジトリは、固定カメラで撮影したテニス動画からラリー区間だけを自動抽出することを目的とした実験的プロジェクトです。

モーション・音声・ポーズなどの特徴量を組み合わせてラリー検出を行います

analyze_rally_params.py で各種パラメータと F1 スコアの関係を解析できます

実装・インターフェースはまだ頻繁に変更される可能性があります
# source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# install dependencies
# (option 1) editable install if pyproject.toml is configured as a standard project
pip install -e .

# (option 2) or install dependencies manually, e.g.
# pip install numpy pandas scikit-learn lightgbm opencv-python librosa moviepy pyyaml ultralytics

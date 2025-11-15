import csv
import os
import sys
import subprocess
import tempfile
from pathlib import Path


def load_segments(csv_path):
    """CSV から (start, end) のリストを読み込む"""
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)

        segs = []
        for row in reader:
            try:
                s = float(row["start"])
                e = float(row["end"])
            except (KeyError, ValueError):
                continue
            if e > s:
                segs.append((s, e))
    return segs


def run_ffmpeg(cmd):
    print("[ffmpeg]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) != 4:
        print("Usage: python cut_segments.py input.mp4 pred_segments.csv output.mp4")
        sys.exit(1)

    in_mp4 = sys.argv[1]
    csv_path = sys.argv[2]
    out_mp4 = sys.argv[3]

    if not os.path.exists(in_mp4):
        print("input video not found:", in_mp4)
        sys.exit(1)

    segs = load_segments(csv_path)
    if not segs:
        print("no segments found in CSV")
        sys.exit(0)

    in_mp4 = str(Path(in_mp4))
    out_mp4 = str(Path(out_mp4))

    with tempfile.TemporaryDirectory(prefix="rally_cut_") as tmp_dir:
        part_files = []

        # --- 各セグメントを個別に切り出し（再エンコードあり） ---
        for i, (s, e) in enumerate(segs):
            duration = e - s
            part = os.path.join(tmp_dir, f"part_{i:03d}.mp4")

            # ポイント：
            #  -ss を -i の「後ろ」に置く → 正確シーク（やや遅いが画質/音が安定）
            #  -c:v libx264 / -c:a aac で再エンコード → GOP途中切り出しによるブロックノイズを回避
            cmd = [
                "ffmpeg", "-y",
                "-i", in_mp4,
                "-ss", f"{s:.3f}",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",      # 必要に応じて slow / medium に変更可
                "-crf", "20",               # 画質とファイルサイズのバランス（18〜23あたりで調整）
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                part,
            ]
            run_ffmpeg(cmd)
            part_files.append(part)

        # --- concat 用リストファイル作成 ---
        list_path = os.path.join(tmp_dir, "list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in part_files:
                # パスにスペースがあっても大丈夫なようにシングルクォートで囲む
                f.write(f"file '{p}'\n")

        # --- 連結（ここは copy でOK：同じフォーマットで揃っているため） ---
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_mp4,
        ]
        run_ffmpeg(cmd_concat)

    print("done ->", os.path.abspath(out_mp4))


if __name__ == "__main__":
    main()

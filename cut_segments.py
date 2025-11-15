import csv
import os
import sys
import subprocess
import tempfile

def load_segments(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        # 区切り文字を自動判定（カンマ or タブ）
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

def main():
    if len(sys.argv) != 4:
        print("Usage: python cut_segments.py input.mp4 pred_segments.csv output.mp4")
        sys.exit(1)

    in_mp4 = sys.argv[1]
    csv_path = sys.argv[2]
    out_mp4 = sys.argv[3]

    segs = load_segments(csv_path)
    if not segs:
        print("no segments found in CSV")
        sys.exit(0)

    tmp_dir = tempfile.mkdtemp(prefix="rally_cut_")
    part_files = []

    # 各セグメントを個別に切り出し
    for i, (s, e) in enumerate(segs):
        duration = e - s
        part = os.path.join(tmp_dir, f"part_{i:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{s:.3f}",
            "-i", in_mp4,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            part,
        ]
        print("[ffmpeg] clip", i, ":", " ".join(cmd))
        subprocess.run(cmd, check=True)
        part_files.append(part)

    # concat 用リストファイル作成
    list_path = os.path.join(tmp_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in part_files:
            # パスにスペースがあっても大丈夫なようにシングルクォートで囲む
            f.write(f"file '{p}'\n")

    # 連結
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_mp4,
    ]
    print("[ffmpeg] concat:", " ".join(cmd_concat))
    subprocess.run(cmd_concat, check=True)

    print("done ->", os.path.abspath(out_mp4))

if __name__ == "__main__":
    main()

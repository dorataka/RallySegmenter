import os
import subprocess
import tempfile

import librosa
import numpy as np


def _extract_audio_wav(video_path: str, sr: int = 22050) -> str:
    """
    動画ファイルから mono の wav を一時ファイルとして切り出して、そのパスを返す。
    呼び出し側で finally で削除すること。
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",          # 映像なし
        "-ac", "1",     # mono
        "-ar", str(sr), # サンプリングレート
        "-acodec", "pcm_s16le",
        tmp_path,
    ]
    # ffmpeg のログは邪魔なので捨てる
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp_path


def audio_flux_series(
    wav_path: str,
    hop_length: int = 512,
    sr: int = 22050,
    smooth_win: int = 9,
):
    """
    旧 audio_feat.py 相当：
    wav からスペクトルフラックスを計算して 0-1 正規化した 1D series と dt を返す。
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    flux = np.maximum(0.0, np.diff(S, axis=1)).sum(axis=0)

    if smooth_win > 1:
        flux = np.convolve(flux, np.ones(smooth_win) / smooth_win, mode="same")

    # 0-1 正規化
    flux = (flux - flux.min()) / (flux.max() - flux.min() + 1e-9)

    # 時間軸と dt
    t = librosa.frames_to_time(np.arange(len(flux)), sr=sr, hop_length=hop_length)
    if len(t) > 1:
        dt = float(t[1] - t[0])
    else:
        dt = hop_length / float(sr)

    return flux.astype(np.float32), dt


def audio_rms_series(
    wav_path: str,
    frame_length: int = 2048,
    hop_length: int = 512,
    sr: int = 22050,
    smooth_win: int = 9,
):
    """
    RMS ベースの音量系列。method="rms" を選びたい場合用。
    とりあえず flux と同じように 0-1 正規化して dt を返す。
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    if smooth_win > 1:
        rms = np.convolve(rms, np.ones(smooth_win) / smooth_win, mode="same")

    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)

    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    if len(t) > 1:
        dt = float(t[1] - t[0])
    else:
        dt = hop_length / float(sr)

    return rms.astype(np.float32), dt


def audio_series(
    path: str,
    method: str = "flux",   # "flux" or "rms"
    rms_win: int = 2048,
    hop: int = 512,
    sr: int = 22050,
    smooth: int = 9,
    bandpass=None,          # いまは未使用（必要なら後で実装）
):
    """
    動画 or 音声ファイル path から、1D の audio series と dt を返す共通入口。
    mp4 のような動画が来た場合は ffmpeg で一時 wav を切り出してから librosa.load する。
    """
    # 拡張子で「そのまま読めるか」を簡易判定
    ext = os.path.splitext(path)[1].lower()
    direct_ok = ext in (".wav", ".flac", ".ogg", ".mp3", ".m4a")

    tmp_wav = None
    try:
        if direct_ok:
            wav_path = path
        else:
            # 動画ファイルとみなして一時 wav に抽出
            wav_path = _extract_audio_wav(path, sr=sr)
            tmp_wav = wav_path

        if method == "flux":
            series, dt = audio_flux_series(
                wav_path,
                hop_length=hop,
                sr=sr,
                smooth_win=smooth,
            )
        elif method == "rms":
            series, dt = audio_rms_series(
                wav_path,
                frame_length=rms_win,
                hop_length=hop,
                sr=sr,
                smooth_win=smooth,
            )
        else:
            raise ValueError(f"Unknown audio method: {method}")

        # bandpass したくなったらここにフィルタ処理を挿す
        return series, dt

    finally:
        if tmp_wav is not None and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

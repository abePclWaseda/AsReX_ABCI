"""
音声処理ユーティリティ
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import torch
import torchaudio


def resample_2ch(
    wav: torch.Tensor, sr: int, target_sr: int = 16000
) -> torch.Tensor:
    """
    音声を2チャンネル、ターゲットサンプルレートに変換

    Args:
        wav: 音声テンソル (1D, 2D, or 3D)
        sr: 現在のサンプルレート
        target_sr: ターゲットサンプルレート

    Returns:
        2チャンネル、ターゲットサンプルレートのCPUテンソル (2, T)
    """
    # リサンプル
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    # 2チャンネルに変換
    if wav.ndim == 1:
        wav = wav.unsqueeze(0).repeat(2, 1)
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.ndim == 3:
        # (batch, channels, time) -> (channels, time)
        if wav.shape[0] == 1:
            wav = wav.squeeze(0)
        else:
            raise ValueError(f"Unexpected tensor shape: {wav.shape}")

    return wav.cpu()


def chunk_audio(
    wav: torch.Tensor, chunk_seconds: int, sample_rate: int = 16000
) -> Iterator[Tuple[float, float, torch.Tensor]]:
    """
    音声を時間チャンクに分割

    Args:
        wav: 音声テンソル (1D or 2D)
        chunk_seconds: チャンクの長さ（秒）
        sample_rate: サンプルレート

    Yields:
        (start_seconds, end_seconds, chunk_tensor) のタプル
    """
    total_samples = wav.shape[-1]
    if total_samples == 0:
        return

    hop_samples = chunk_seconds * sample_rate
    for start in range(0, total_samples, hop_samples):
        end = min(start + hop_samples, total_samples)
        start_sec = start / sample_rate
        end_sec = end / sample_rate
        yield start_sec, end_sec, wav[..., start:end]


def load_audio(path: Path | str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    音声ファイルを読み込み

    Args:
        path: 音声ファイルのパス
        target_sr: ターゲットサンプルレート（Noneの場合はリサンプルしない）

    Returns:
        (audio_tensor, sample_rate) のタプル
    """
    path = Path(path)
    wav, sr = torchaudio.load(path)

    if target_sr is not None and sr != target_sr:
        wav = resample_2ch(wav, sr, target_sr)
        sr = target_sr
    else:
        wav = resample_2ch(wav, sr, sr)

    return wav, sr


def is_processed(wav_path: Path, alignment_root: Path) -> bool:
    """
    音声ファイルが既に処理済みかどうかをチェック

    Args:
        wav_path: 音声ファイルのパス
        alignment_root: アライメント結果のルートディレクトリ

    Returns:
        処理済みの場合True
    """
    output_path = alignment_root / f"{wav_path.stem}.json"
    return output_path.is_file() and output_path.stat().st_size > 0



"""
音源分離処理モジュール
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from asrex_pkg.config import ProcessingConfig


class SeparationProcessor:
    """音源分離処理クラス"""

    def __init__(
        self,
        config: ProcessingConfig,
        device: str,
        model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k",
    ):
        """
        Args:
            config: 処理設定
            device: デバイス（"cuda:0"など）
            model_name: 音源分離モデル名（デフォルト: ConvTasNet_Libri2Mix_sepclean_16k）
        """
        self.config = config
        self.device = device
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """音源分離モデルをロード"""
        from asteroid.models import ConvTasNet

        self.model = (
            ConvTasNet.from_pretrained(self.model_name).to(self.device).eval()
        )

    def separate_mono(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        cache_path: Optional[Path] = None,
    ) -> torch.Tensor:
        """
        モノラル音声を音源分離して2チャンネルに変換

        Args:
            audio: 音声テンソル（任意の形状、モノラルまたはマルチチャンネル）
            sample_rate: サンプルレート
            cache_path: キャッシュファイルのパス（Noneの場合はキャッシュしない）

        Returns:
            分離後の音声テンソル (2, T) - CPUテンソル
        """
        # キャッシュファイルが存在する場合は読み込み
        if cache_path is not None and cache_path.exists():
            try:
                stereo, cached_sr = torchaudio.load(cache_path)
                # キャッシュファイルは常に16kHzで保存されているはず
                # ただし、後続処理で使用するサンプルレートに合わせてリサンプル
                if cached_sr != 16000:
                    # キャッシュファイルが16kHzでない場合は再生成
                    os.remove(cache_path)
                    print(f"[warn] cached file has wrong sample rate ({cached_sr}Hz) → regenerate")
                else:
                    # キャッシュファイルは16kHzで保存されているので、そのまま返す
                    # （後続処理は16kHzを想定）
                    return stereo
            except Exception as e:
                # キャッシュファイルが壊れている場合は削除して再生成
                os.remove(cache_path)
                print(f"[warn] cached file broken → regenerate ({cache_path}): {e}")

        # モノラル音声に変換
        if audio.ndim == 1:
            mono = audio.unsqueeze(0)
        elif audio.ndim == 2:
            # (channels, time) -> モノラルに変換
            if audio.shape[0] > 1:
                mono = audio.mean(0, keepdim=True)
            else:
                mono = audio
        elif audio.ndim == 3:
            # (batch, channels, time) -> モノラルに変換
            if audio.shape[1] > 1:
                mono = audio.mean(1, keepdim=True).squeeze(0)
            else:
                mono = audio.squeeze(0)
        else:
            raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")

        # サンプルレートを16kHzに統一（ConvTasNetは16kHzを想定）
        if sample_rate != 16000:
            mono = torchaudio.functional.resample(mono, sample_rate, 16000)
            sample_rate = 16000

        # デバイスに移動して分離
        mono_device = mono.to(self.device)
        with torch.no_grad():
            separated = self.model.separate(mono_device)[0]

        # CPUに移動
        stereo = separated.cpu().float()

        # キャッシュに保存
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(cache_path, stereo, sample_rate)

        return stereo

    def separate_file(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
    ) -> torch.Tensor:
        """
        音声ファイルを読み込んで音源分離

        Args:
            audio_path: 入力音声ファイルのパス
            output_path: 出力先パス（Noneの場合は保存しない）
            cache_path: キャッシュファイルのパス（Noneの場合はoutput_pathを使用）

        Returns:
            分離後の音声テンソル (2, T) - CPUテンソル
        """
        # キャッシュパスを決定
        if cache_path is None:
            cache_path = output_path

        # 音声ファイルを読み込み
        wav_tensor, sr = torchaudio.load(audio_path)

        # 音源分離
        stereo = self.separate_mono(wav_tensor, sr, cache_path=cache_path)

        # 出力パスが指定されている場合は保存（キャッシュパスと異なる場合のみ）
        if output_path is not None and output_path != cache_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, stereo, self.config.sample_rate)

        return stereo


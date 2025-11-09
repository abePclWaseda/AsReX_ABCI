"""
ASR処理モジュール
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch

from asrex_pkg.config import ProcessingConfig


class ASRProcessor:
    """ASR処理クラス"""

    def __init__(self, config: ProcessingConfig, device: str):
        """
        Args:
            config: 処理設定
            device: デバイス（"cuda:0"など）
        """
        self.config = config
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """ASRモデルをロード"""
        if self.config.asr_model_type == "espnet":
            from reazonspeech.espnet.asr import load_model

            self.model = load_model(device=self.device)
        elif self.config.asr_model_type == "nemo":
            from reazonspeech.nemo.asr import load_model

            self.model = load_model(device=self.device)
        else:
            raise ValueError(
                f"Unknown ASR model type: {self.config.asr_model_type}. "
                "Use 'espnet' or 'nemo'"
            )

    def transcribe_whole_file(self, audio: torch.Tensor, sample_rate: int) -> dict:
        """
        ファイル全体を一括で音声認識

        Args:
            audio: 音声テンソル (1D)
            sample_rate: サンプルレート

        Returns:
            {"text": str, "segments": List[dict]} の辞書
        """
        if self.config.asr_model_type == "espnet":
            from reazonspeech.espnet.asr import transcribe, audio_from_numpy

            ret = transcribe(self.model, audio_from_numpy(audio.numpy(), sample_rate))
        else:  # nemo
            from reazonspeech.nemo.asr import transcribe, audio_from_numpy

            ret = transcribe(self.model, audio_from_numpy(audio.numpy(), sample_rate))

        segments = [
            {
                "start": seg.start_seconds,
                "end": seg.end_seconds,
                "text": seg.text,
            }
            for seg in ret.segments
        ]

        return {"text": ret.text, "segments": segments}

    def transcribe_chunked(
        self, audio: torch.Tensor, sample_rate: int, chunk_seconds: int
    ) -> dict:
        """
        チャンクに分割して音声認識（ストリーミングモード）

        Args:
            audio: 音声テンソル (1D)
            sample_rate: サンプルレート
            chunk_seconds: チャンクの長さ（秒）

        Returns:
            {"text": str, "segments": List[dict]} の辞書
        """
        from asrex_pkg.audio_utils import chunk_audio

        if self.config.asr_model_type == "espnet":
            from reazonspeech.espnet.asr import transcribe, audio_from_numpy
        else:  # nemo
            from reazonspeech.nemo.asr import transcribe, audio_from_numpy

        all_segments = []
        text_accum = []

        for start_sec, end_sec, chunk in chunk_audio(audio, chunk_seconds, sample_rate):
            # nemoの場合はoffset_secondsをサポート
            if self.config.asr_model_type == "nemo":
                # offset_secondsパラメータがあるかチェック
                import inspect
                sig = inspect.signature(transcribe)
                if "offset_seconds" in sig.parameters:
                    ret = transcribe(
                        self.model,
                        audio_from_numpy(chunk.numpy(), sample_rate),
                        offset_seconds=start_sec,
                    )
                else:
                    # offset_secondsがサポートされていない場合は手動で追加
                    ret = transcribe(
                        self.model,
                        audio_from_numpy(chunk.numpy(), sample_rate),
                    )
                    for seg in ret.segments:
                        seg.start_seconds += start_sec
                        seg.end_seconds += start_sec
            else:  # espnet doesn't support offset_seconds
                ret = transcribe(
                    self.model,
                    audio_from_numpy(chunk.numpy(), sample_rate),
                )
                # オフセットを手動で追加
                for seg in ret.segments:
                    seg.start_seconds += start_sec
                    seg.end_seconds += start_sec

            text_accum.append(ret.text)
            all_segments.extend(
                {
                    "start": round(seg.start_seconds, 3),
                    "end": round(seg.end_seconds, 3),
                    "text": seg.text,
                }
                for seg in ret.segments
            )

        return {"text": "".join(text_accum), "segments": all_segments}

    def process_channel(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        チャンネルの音声認識を実行

        Args:
            audio: 音声テンソル (1D)
            sample_rate: サンプルレート
            output_path: 出力先パス（Noneの場合は保存しない）

        Returns:
            {"text": str, "segments": List[dict]} の辞書
        """
        # 既存ファイルがあれば読み込み
        if output_path is not None and output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        # 音声認識実行
        if self.config.chunk_seconds is None:
            result = self.transcribe_whole_file(audio, sample_rate)
        else:
            result = self.transcribe_chunked(
                audio, sample_rate, self.config.chunk_seconds
            )

        # 結果を保存
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result


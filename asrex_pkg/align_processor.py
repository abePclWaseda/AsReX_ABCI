"""
アライメント処理モジュール
"""

from __future__ import annotations

import json
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import whisperx

from asrex_pkg.config import ProcessingConfig


class AlignProcessor:
    """アライメント処理クラス"""

    def __init__(self, config: ProcessingConfig, device: str):
        """
        Args:
            config: 処理設定
            device: デバイス（"cuda:0"など）
        """
        self.config = config
        self.device = device
        self.align_model = None
        self.meta = None
        self.executor = ThreadPoolExecutor(max_workers=config.align_threads)
        self._load_model()

    def _load_model(self):
        """アライメントモデルをロード"""
        self.align_model, self.meta = whisperx.load_align_model(
            self.config.language, self.device
        )

    def align_segments(
        self, segments: List[dict], audio: "torch.Tensor"
    ) -> dict:
        """
        セグメントをアライメント

        Args:
            segments: セグメントのリスト
            audio: 音声テンソル（numpy array）

        Returns:
            アライメント結果の辞書
        """
        # numpy arrayに変換（torch.Tensorの場合は）
        if hasattr(audio, "numpy"):
            audio_np = audio.numpy()
        else:
            audio_np = audio

        aligned = whisperx.align(
            segments,
            self.align_model,
            self.meta,
            audio_np,
            self.device,
            return_char_alignments=False,
        )

        return aligned

    def align_channel_async(
        self, segments: List[dict], audio: "torch.Tensor"
    ) -> futures.Future:
        """
        チャンネルのアライメントを非同期で実行

        Args:
            segments: セグメントのリスト
            audio: 音声テンソル

        Returns:
            Futureオブジェクト
        """
        return self.executor.submit(self.align_segments, segments, audio)

    def merge_alignments(
        self, alignments: List[dict], speakers: List[str]
    ) -> List[dict]:
        """
        複数チャンネルのアライメント結果をマージ

        Args:
            alignments: アライメント結果のリスト（チャンネルごと）
            speakers: 話者のリスト

        Returns:
            マージされた単語レベルのリスト
        """
        merged = []

        for idx, aligned in enumerate(alignments):
            spk = speakers[idx]
            for seg in aligned["segments"]:
                for w in seg.get("words", []):
                    merged.append(
                        {
                            "speaker": spk,
                            "word": w["word"],
                            "start": round(w["start"], 3),
                            "end": round(w["end"], 3),
                        }
                    )

        # 時系列でソート
        merged.sort(key=lambda x: x["start"])

        return merged

    def save_alignment(
        self, merged: List[dict], output_path: Path
    ):
        """
        アライメント結果を保存

        Args:
            merged: マージされたアライメント結果
            output_path: 出力先パス
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    def __del__(self):
        """リソースのクリーンアップ"""
        if self.executor is not None:
            self.executor.shutdown(wait=False)


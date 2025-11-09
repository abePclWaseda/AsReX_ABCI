"""
メインパイプライン
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from asrex_pkg.align_processor import AlignProcessor
from asrex_pkg.asr_processor import ASRProcessor
from asrex_pkg.audio_utils import is_processed, load_audio
from asrex_pkg.config import Config


class StereoASRPipeline:
    """ステレオ音声ASRパイプライン"""

    def __init__(self, config: Config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.config.data.ensure_dirs()

    def process_file(
        self,
        wav_path: Path,
        device: str,
        asr_processor: Optional[ASRProcessor] = None,
        align_processor: Optional[AlignProcessor] = None,
    ) -> bool:
        """
        単一ファイルを処理

        Args:
            wav_path: 音声ファイルのパス
            device: デバイス
            asr_processor: ASRプロセッサ（Noneの場合は新規作成）
            align_processor: アライメントプロセッサ（Noneの場合は新規作成）

        Returns:
            成功した場合True
        """
        try:
            # モデルをロード（まだの場合は）
            if asr_processor is None:
                asr_processor = ASRProcessor(self.config.processing, device)
            if align_processor is None:
                align_processor = AlignProcessor(self.config.processing, device)

            # 音声を読み込み
            wav_tensor, sr = load_audio(wav_path, self.config.processing.sample_rate)

            # 相対パスを取得
            rel = wav_path.relative_to(self.config.data.audio_root)
            txt_dir = self.config.data.transcripts_root / rel.parent
            txt_dir.mkdir(parents=True, exist_ok=True)

            # 各チャンネルを処理
            segments_per_channel = []

            for ch, spk in enumerate(self.config.processing.speakers):
                txt_json = txt_dir / f"{wav_path.stem}_{spk}.json"

                # ASR処理
                result = asr_processor.process_channel(
                    wav_tensor[ch],
                    self.config.processing.sample_rate,
                    output_path=txt_json,
                )
                segments_per_channel.append(result["segments"])

            # アライメント処理（並列）
            futures = []
            for idx in range(len(self.config.processing.speakers)):
                future = align_processor.align_channel_async(
                    segments_per_channel[idx],
                    wav_tensor[idx],
                )
                futures.append(future)

            # 結果を取得
            alignments = [fut.result() for fut in futures]

            # マージ
            merged = align_processor.merge_alignments(
                alignments, self.config.processing.speakers
            )

            # 保存
            output_path = (
                self.config.data.alignment_root / rel.parent / f"{wav_path.stem}.json"
            )
            align_processor.save_alignment(merged, output_path)

            return True

        except Exception as e:
            error_msg = f"ERROR processing {wav_path}: {str(e)}"
            print(error_msg, flush=True)
            if self.config.logging.log_errors:
                log_path = self.config.logging.get_log_path("errors")
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(error_msg + "\n")
                    traceback.print_exc(file=log_file)
                    log_file.write("\n")
            return False

    def get_target_files(
        self, sub_dirs: Optional[List[str]] = None
    ) -> List[Path]:
        """
        処理対象のファイルリストを取得

        Args:
            sub_dirs: 処理するサブディレクトリのリスト（Noneの場合は全て）

        Returns:
            処理対象のファイルパスのリスト
        """
        targets = []

        if sub_dirs is None:
            sub_dirs = ["."]

        for d in sub_dirs:
            sub = self.config.data.audio_root / d
            if not sub.is_dir():
                print(f"[warn] {sub} not found", file=sys.stderr)
                continue

            # 既に処理済みのファイルをスキップ
            for wav_path in sub.rglob("*.wav"):
                if not is_processed(wav_path, self.config.data.alignment_root):
                    targets.append(wav_path)

        return targets

    def process_worker(
        self, device: str, wav_paths: List[Path]
    ):
        """
        ワーカー関数（マルチプロセス用）

        Args:
            device: デバイス
            wav_paths: 処理するファイルのリスト
        """
        # デバイスを設定
        torch.cuda.set_device(device)
        os.environ["OMP_NUM_THREADS"] = "1"

        # プロセッサを作成（ワーカーごとに1回だけ）
        asr_processor = ASRProcessor(self.config.processing, device)
        align_processor = AlignProcessor(self.config.processing, device)

        # 各ファイルを処理
        for wav_path in tqdm(wav_paths, desc=f"[GPU {device}]"):
            self.process_file(wav_path, device, asr_processor, align_processor)

    def run(
        self,
        sub_dirs: Optional[List[str]] = None,
        devices: Optional[List[str]] = None,
    ):
        """
        パイプラインを実行

        Args:
            sub_dirs: 処理するサブディレクトリのリスト
            devices: 使用するデバイスのリスト（Noneの場合は自動検出）
        """
        import multiprocessing as mp

        # 処理対象ファイルを取得
        targets = self.get_target_files(sub_dirs)
        print(f"{len(targets)} wav files to process.")

        if not targets:
            print("No files to process.")
            return

        # デバイスを決定
        if devices is None:
            if torch.cuda.is_available():
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                devices = ["cpu"]

        if not devices:
            print("No devices available.", file=sys.stderr)
            sys.exit(1)

        # ファイルをデバイスごとに分散
        chunks = [targets[i :: len(devices)] for i in range(len(devices))]

        # マルチプロセスで実行
        mp.set_start_method("spawn", force=True)

        processes = []
        for device, chunk in zip(devices, chunks):
            if not chunk:
                continue
            p = mp.Process(target=self.process_worker, args=(device, chunk))
            processes.append(p)
            p.start()

        # 全てのプロセスが終了するまで待機
        for p in processes:
            p.join()

        print("=== Processing complete ===")



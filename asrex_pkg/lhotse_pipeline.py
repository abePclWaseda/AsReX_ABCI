"""
Lhotse形式データ処理パイプライン
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from lhotse import CutSet
except ImportError:
    raise ImportError(
        "lhotse is not installed. Please install it with: pip install lhotse>=1.21.0"
    )

from asrex_pkg.align_processor import AlignProcessor
from asrex_pkg.asr_processor import ASRProcessor
from asrex_pkg.audio_utils import resample_2ch
from asrex_pkg.config import Config
from asrex_pkg.separation_processor import SeparationProcessor


class LhotsePipeline:
    """Lhotse形式データ処理パイプライン"""

    def __init__(self, config: Config):
        """
        Args:
            config: 設定オブジェクト
        """
        self.config = config
        self.config.data.ensure_dirs()

    def process_audio_tensor(
        self,
        wav_tensor: torch.Tensor,
        sr: int,
        cut_id: str,
        output_stem: str,
        device: str,
        asr_processor: Optional[ASRProcessor] = None,
        align_processor: Optional[AlignProcessor] = None,
        separation_processor: Optional[SeparationProcessor] = None,
        output_subdir: Optional[Path] = None,
    ) -> bool:
        """
        音声テンソルを処理（既存のprocess_fileロジックを再利用）

        Args:
            wav_tensor: 音声テンソル (channels, samples) または (samples,)
            sr: サンプルレート
            cut_id: カットID（エラーメッセージ用）
            output_stem: 出力ファイル名のstem（拡張子なし）
            device: デバイス
            asr_processor: ASRプロセッサ（Noneの場合は新規作成）
            align_processor: アライメントプロセッサ（Noneの場合は新規作成）
            separation_processor: 音源分離プロセッサ（Noneの場合は新規作成）
            output_subdir: 出力サブディレクトリ（Noneの場合は transcripts_root/alignment_root 直下）

        Returns:
            成功した場合True
        """
        try:
            # モデルをロード（まだの場合は）
            if asr_processor is None:
                asr_processor = ASRProcessor(self.config.processing, device)
            if align_processor is None:
                align_processor = AlignProcessor(self.config.processing, device)

            # 出力ディレクトリを決定
            if output_subdir is None:
                txt_dir = self.config.data.transcripts_root
                align_dir = self.config.data.alignment_root
            else:
                txt_dir = self.config.data.transcripts_root / output_subdir
                align_dir = self.config.data.alignment_root / output_subdir
            txt_dir.mkdir(parents=True, exist_ok=True)

            # 音源分離が有効な場合
            if self.config.processing.enable_separation:
                if separation_processor is None:
                    separation_processor = SeparationProcessor(
                        self.config.processing,
                        device,
                        self.config.processing.separation_model_name,
                    )

                # モノラルに変換（複数チャンネルの場合は平均化）
                if wav_tensor.ndim == 1:
                    mono_audio = wav_tensor
                elif wav_tensor.ndim == 2:
                    if wav_tensor.shape[0] > 1:
                        # マルチチャンネルの場合は平均化
                        mono_audio = wav_tensor.mean(dim=0)
                    else:
                        mono_audio = wav_tensor.squeeze(0)
                else:
                    raise ValueError(
                        f"Unexpected audio tensor shape: {wav_tensor.shape}"
                    )

                # 音源分離を実行（常に16kHzで処理される）
                wav_tensor = separation_processor.separate_mono(
                    mono_audio,
                    sr,
                    cache_path=None,  # Lhotse形式ではキャッシュパスを設定しない
                )
                # 音源分離後の音声は常に16kHz
                sr = 16000
                # 設定のサンプルレートが16kHzでない場合は警告
                if self.config.processing.sample_rate != 16000:
                    print(
                        f"[warn] Source separation outputs 16kHz audio, but config.sample_rate is {self.config.processing.sample_rate}. "
                        "Using 16kHz for processing."
                    )
            else:
                # 2チャンネルに変換
                if wav_tensor.ndim == 1:
                    wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1)
                elif wav_tensor.ndim == 2:
                    if wav_tensor.shape[0] == 1:
                        wav_tensor = wav_tensor.repeat(2, 1)
                    elif wav_tensor.shape[0] > 2:
                        # 2チャンネルより多い場合は最初の2チャンネルを使用
                        wav_tensor = wav_tensor[:2]
                    # 既に2チャンネルの場合はそのまま使用
                else:
                    raise ValueError(
                        f"Unexpected audio tensor shape: {wav_tensor.shape}"
                    )

                # リサンプル
                if sr != self.config.processing.sample_rate:
                    wav_tensor = resample_2ch(
                        wav_tensor, sr, self.config.processing.sample_rate
                    )
                    sr = self.config.processing.sample_rate

            # 各チャンネルを処理
            segments_per_channel = []

            for ch, spk in enumerate(self.config.processing.speakers):
                txt_json = txt_dir / f"{output_stem}_{spk}.json"

                # ASR処理
                result = asr_processor.process_channel(
                    wav_tensor[ch],
                    sr,
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
            output_path = align_dir / f"{output_stem}.json"
            align_processor.save_alignment(merged, output_path)

            return True

        except Exception as e:
            error_msg = f"ERROR processing {cut_id}: {str(e)}"
            print(error_msg, flush=True)
            if self.config.logging.log_errors:
                log_path = self.config.logging.get_log_path("errors")
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(error_msg + "\n")
                    traceback.print_exc(file=log_file)
                    log_file.write("\n")
            return False

    def process_cut(
        self,
        cut,
        device: str,
        asr_processor: Optional[ASRProcessor] = None,
        align_processor: Optional[AlignProcessor] = None,
        separation_processor: Optional[SeparationProcessor] = None,
        output_subdir: Optional[Path] = None,
    ) -> bool:
        """
        単一のカットを処理

        Args:
            cut: LhotseのCutオブジェクト
            device: デバイス
            asr_processor: ASRプロセッサ（Noneの場合は新規作成）
            align_processor: アライメントプロセッサ（Noneの場合は新規作成）
            separation_processor: 音源分離プロセッサ（Noneの場合は新規作成）
            output_subdir: 出力サブディレクトリ

        Returns:
            成功した場合True
        """
        # 音声データを読み込み
        audio_tensor = cut.load_audio()  # (channels, samples)
        sr = cut.sampling_rate

        # カットIDを出力ファイル名として使用
        cut_id = cut.id
        output_stem = cut_id

        return self.process_audio_tensor(
            audio_tensor,
            sr,
            cut_id,
            output_stem,
            device,
            asr_processor,
            align_processor,
            separation_processor,
            output_subdir,
        )

    def process_cutset_file(
        self,
        cutset_path: Path,
        device: str,
        asr_processor: Optional[ASRProcessor] = None,
        align_processor: Optional[AlignProcessor] = None,
        separation_processor: Optional[SeparationProcessor] = None,
        output_subdir: Optional[Path] = None,
    ) -> int:
        """
        カットセットファイルを処理

        Args:
            cutset_path: カットセットファイルのパス（*.jsonl.gz）
            device: デバイス
            asr_processor: ASRプロセッサ（Noneの場合は新規作成）
            align_processor: アライメントプロセッサ（Noneの場合は新規作成）
            separation_processor: 音源分離プロセッサ（Noneの場合は新規作成）
            output_subdir: 出力サブディレクトリ

        Returns:
            処理成功したカット数
        """
        # カットセットを読み込み
        cuts = CutSet.from_file(cutset_path)

        success_count = 0
        for cut in cuts:
            if self.process_cut(
                cut,
                device,
                asr_processor,
                align_processor,
                separation_processor,
                output_subdir,
            ):
                success_count += 1

        return success_count

    def process_worker(
        self,
        device: str,
        cutset_paths: List[Path],
        output_subdirs: Optional[List[Optional[Path]]] = None,
    ):
        """
        ワーカー関数（マルチプロセス用）

        Args:
            device: デバイス
            cutset_paths: 処理するカットセットファイルのリスト
            output_subdirs: 各カットセットファイルに対応する出力サブディレクトリのリスト
        """
        # デバイスを設定
        torch.cuda.set_device(device)
        os.environ["OMP_NUM_THREADS"] = "1"

        # プロセッサを作成（ワーカーごとに1回だけ）
        asr_processor = ASRProcessor(self.config.processing, device)
        align_processor = AlignProcessor(self.config.processing, device)
        separation_processor = None
        if self.config.processing.enable_separation:
            separation_processor = SeparationProcessor(
                self.config.processing,
                device,
                self.config.processing.separation_model_name,
            )

        # 各カットセットファイルを処理
        for idx, cutset_path in enumerate(tqdm(cutset_paths, desc=f"[GPU {device}]")):
            output_subdir = (
                output_subdirs[idx] if output_subdirs is not None else None
            )
            self.process_cutset_file(
                cutset_path,
                device,
                asr_processor,
                align_processor,
                separation_processor,
                output_subdir,
            )

    def get_target_cutsets(
        self, root_dir: Path, sub_dirs: Optional[List[str]] = None
    ) -> List[Tuple[Path, Optional[Path]]]:
        """
        処理対象のカットセットファイルリストを取得

        Args:
            root_dir: ルートディレクトリ
            sub_dirs: 処理するサブディレクトリのリスト（Noneの場合は全て）

        Returns:
            (cutset_path, output_subdir) のタプルのリスト
        """
        targets = []

        if sub_dirs is None:
            # ルートディレクトリ内の全てのディレクトリを検索
            sub_dirs = [
                str(d.name)
                for d in root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

        for d in sub_dirs:
            sub = root_dir / d
            if not sub.is_dir():
                print(f"[warn] {sub} not found", file=sys.stderr)
                continue

            # cuts.*.jsonl.gz ファイルを検索
            for cutset_path in sub.glob("cuts.*.jsonl.gz"):
                # 出力サブディレクトリを設定（サブディレクトリ名を使用）
                output_subdir = Path(d)
                targets.append((cutset_path, output_subdir))

        return targets

    def run(
        self,
        root_dir: Path,
        sub_dirs: Optional[List[str]] = None,
        devices: Optional[List[str]] = None,
    ):
        """
        パイプラインを実行

        Args:
            root_dir: Lhotseデータのルートディレクトリ
            sub_dirs: 処理するサブディレクトリのリスト（Noneの場合は全て）
            devices: 使用するデバイスのリスト（Noneの場合は自動検出）
        """
        import multiprocessing as mp

        # 処理対象カットセットファイルを取得
        targets = self.get_target_cutsets(root_dir, sub_dirs)
        print(f"{len(targets)} cutset files to process.")

        if not targets:
            print("No cutset files to process.")
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
        for device_idx, device in enumerate(devices):
            chunk = chunks[device_idx]
            if not chunk:
                continue

            # チャンクをcutset_pathsとoutput_subdirsに分割
            cutset_paths = [t[0] for t in chunk]
            output_subdirs = [t[1] for t in chunk]

            p = mp.Process(
                target=self.process_worker,
                args=(device, cutset_paths, output_subdirs),
            )
            processes.append(p)
            p.start()

        # 全てのプロセスが終了するまで待機
        for p in processes:
            p.join()

        print("=== Processing complete ===")


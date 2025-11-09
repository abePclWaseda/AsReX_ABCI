"""
CLIエントリーポイント
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asrex_pkg.config import Config
from asrex_pkg.pipeline import StereoASRPipeline


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="AsReX: ステレオ音声のASRとアライメント処理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 設定ファイルを使用
  python -m asrex_pkg.cli --config config.yaml --dirs .

  # コマンドライン引数で設定
  python -m asrex_pkg.cli --root /path/to/data --dirs subdir1 subdir2

  # デバイスを指定
  python -m asrex_pkg.cli --config config.yaml --devices cuda:0 cuda:1
        """,
    )

    # 設定ファイル関連
    parser.add_argument(
        "--config",
        type=Path,
        help="設定ファイルのパス（YAML形式）",
    )

    # データディレクトリ
    parser.add_argument(
        "--root",
        type=Path,
        help="データのルートディレクトリ（--configが指定されない場合に使用）",
    )
    parser.add_argument(
        "--audio-dir",
        default="audio",
        help="音声ファイルのディレクトリ名（デフォルト: audio）",
    )
    parser.add_argument(
        "--transcripts-dir",
        default="transcripts",
        help="トランスクリプトのディレクトリ名（デフォルト: transcripts）",
    )
    parser.add_argument(
        "--alignment-dir",
        default="text",
        help="アライメント結果のディレクトリ名（デフォルト: text）",
    )

    # 処理設定
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["."],
        help="処理するサブディレクトリ（デフォルト: .）",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        help="使用するデバイス（例: cuda:0 cuda:1）。未指定の場合は自動検出",
    )
    parser.add_argument(
        "--asr-model",
        choices=["espnet", "nemo"],
        help="ASRモデルのタイプ（デフォルト: espnet）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="サンプルレート（デフォルト: 16000）",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="言語コード（デフォルト: ja）",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        help="チャンクの長さ（秒）。指定しない場合はファイル全体を処理",
    )
    parser.add_argument(
        "--align-threads",
        type=int,
        default=2,
        help="アライメントのスレッド数（デフォルト: 2）",
    )

    # ログ設定
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="ログディレクトリ",
    )
    parser.add_argument(
        "--log-prefix",
        default="asrex",
        help="ログファイルのプレフィックス（デフォルト: asrex）",
    )

    args = parser.parse_args()

    # 設定を読み込み
    if args.config:
        try:
            config = Config.from_yaml(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.root:
        config = Config.from_env(args.root)
        # コマンドライン引数で上書き
        config.data.audio_dir = args.audio_dir
        config.data.transcripts_dir = args.transcripts_dir
        config.data.alignment_dir = args.alignment_dir
    else:
        print(
            "Error: --config or --root must be specified",
            file=sys.stderr,
        )
        sys.exit(1)

    # 処理設定を上書き
    if args.asr_model:
        config.processing.asr_model_type = args.asr_model
    if args.sample_rate:
        config.processing.sample_rate = args.sample_rate
    if args.language:
        config.processing.language = args.language
    if args.chunk_seconds:
        config.processing.chunk_seconds = args.chunk_seconds
    if args.align_threads:
        config.processing.align_threads = args.align_threads

    # ログ設定を上書き
    if args.log_dir:
        config.logging.log_dir = args.log_dir
    if args.log_prefix:
        config.logging.log_prefix = args.log_prefix

    # パイプラインを実行
    pipeline = StereoASRPipeline(config)
    pipeline.run(sub_dirs=args.dirs, devices=args.devices)


if __name__ == "__main__":
    main()



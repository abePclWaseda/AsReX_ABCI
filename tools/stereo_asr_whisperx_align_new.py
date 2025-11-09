#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stereo → ReazonSpeech‑ESPnet ASR → WhisperX alignment (新パッケージ版)
==============================================================================
新しいasrex_pkgパッケージを使用したバージョン
"""

from pathlib import Path

from asrex_pkg import Config, StereoASRPipeline

if __name__ == "__main__":
    # 設定ファイルから読み込み、または直接設定
    root_dir = Path("/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi")
    
    # 設定を作成
    config = Config.from_env(root_dir)
    config.data.audio_dir = "audio"
    config.data.transcripts_dir = "transcripts"
    config.data.alignment_dir = "text"
    config.processing.speakers = ["A", "B"]
    config.processing.sample_rate = 16000
    config.processing.language = "ja"
    config.processing.asr_model_type = "espnet"
    config.processing.chunk_seconds = None  # whole-file mode
    config.processing.align_threads = 2
    config.logging.log_prefix = "Tabidachi"
    
    # パイプラインを作成して実行
    pipeline = StereoASRPipeline(config)
    pipeline.run(sub_dirs=["."])



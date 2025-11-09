"""
設定管理モジュール
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    """データディレクトリ設定"""

    root: Path
    audio_dir: str = "audio"
    transcripts_dir: str = "transcripts"
    alignment_dir: str = "text"

    def __post_init__(self):
        """パスをPathオブジェクトに変換"""
        if isinstance(self.root, str):
            self.root = Path(self.root)

    @property
    def audio_root(self) -> Path:
        """音声ファイルのルートディレクトリ"""
        return self.root / self.audio_dir

    @property
    def transcripts_root(self) -> Path:
        """トランスクリプトのルートディレクトリ"""
        return self.root / self.transcripts_dir

    @property
    def alignment_root(self) -> Path:
        """アライメント結果のルートディレクトリ"""
        return self.root / self.alignment_dir

    def ensure_dirs(self):
        """必要なディレクトリを作成"""
        for p in (self.transcripts_root, self.alignment_root):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ProcessingConfig:
    """処理設定"""

    speakers: List[str] = field(default_factory=lambda: ["A", "B"])
    sample_rate: int = 16000
    language: str = "ja"
    asr_model_type: str = "espnet"  # "espnet" or "nemo"
    align_threads: int = 2
    chunk_seconds: Optional[int] = None  # None = whole file mode, int = chunk mode
    device: Optional[str] = None  # None = auto-detect


@dataclass
class LoggingConfig:
    """ログ設定"""

    log_dir: Optional[Path] = None
    log_prefix: str = "asrex"
    log_errors: bool = True

    def __post_init__(self):
        """パスをPathオブジェクトに変換"""
        if self.log_dir is not None and isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)

    def get_log_path(self, suffix: str = "errors") -> Path:
        """ログファイルのパスを取得"""
        if self.log_dir is None:
            return Path(f"{self.log_prefix}_{suffix}.log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{self.log_prefix}_{suffix}.log"


@dataclass
class Config:
    """全体設定"""

    data: DataConfig
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> Config:
        """辞書から設定を作成"""
        data_config = DataConfig(**config_dict.get("data", {}))
        processing_config = ProcessingConfig(**config_dict.get("processing", {}))
        logging_config = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            data=data_config,
            processing=processing_config,
            logging=logging_config,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> Config:
        """YAMLファイルから設定を読み込み"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls, root_dir: Path | str) -> Config:
        """環境変数から設定を作成（デフォルト設定）"""
        root_dir = Path(root_dir)
        data_config = DataConfig(root=root_dir)
        return cls(data=data_config)

    def to_dict(self) -> dict:
        """設定を辞書に変換"""
        return {
            "data": {
                "root": str(self.data.root),
                "audio_dir": self.data.audio_dir,
                "transcripts_dir": self.data.transcripts_dir,
                "alignment_dir": self.data.alignment_dir,
            },
            "processing": {
                "speakers": self.processing.speakers,
                "sample_rate": self.processing.sample_rate,
                "language": self.processing.language,
                "asr_model_type": self.processing.asr_model_type,
                "align_threads": self.processing.align_threads,
                "chunk_seconds": self.processing.chunk_seconds,
                "device": self.processing.device,
            },
            "logging": {
                "log_dir": str(self.logging.log_dir) if self.logging.log_dir else None,
                "log_prefix": self.logging.log_prefix,
                "log_errors": self.logging.log_errors,
            },
        }

    def save_yaml(self, yaml_path: Path | str):
        """設定をYAMLファイルに保存"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)



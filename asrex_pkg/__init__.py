"""
AsReX: Audio Speech Recognition and Alignment Pipeline
======================================================
ステレオ音声（2チャンネル対話）の音声認識と単語レベルの時間アライメントを行うパッケージ
"""

__version__ = "0.1.0"

from asrex_pkg.config import Config
from asrex_pkg.pipeline import StereoASRPipeline

__all__ = ["Config", "StereoASRPipeline"]



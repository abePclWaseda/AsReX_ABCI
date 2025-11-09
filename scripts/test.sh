#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=50:00:00
#PBS -j oe
#PBS -N 0162_Tabidachi

set -eu

cd "$PBS_O_WORKDIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asrex310

# 音声認識＋アライメント
python -m asrex_pkg.cli --config config_example.yaml --dirs .

# 音源分離＋音声認識＋アライメント
python -m asrex_pkg.cli --config config_jchat_separation.yaml --dirs .
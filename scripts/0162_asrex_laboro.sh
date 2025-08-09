#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=1
#PBS -l walltime=100:00:00
#PBS -N 0162_laboro_align
#PBS -j oe

set -eux

echo "JOB_ID : $PBS_JOBID"
echo "ARRAY_ID: ${PBS_ARRAY_INDEX:-0}"
echo "WORKDIR: $PBS_O_WORKDIR"
cd "$PBS_O_WORKDIR"

# 環境
module list || true
source ~/miniforge3/etc/profile.d/conda.sh
conda activate asrex310
which python
python --version

# 実行
exec python -m tools.asrex_espnet_laboro \
  --archive /groups/gcg51557/experiments/0185_speech_dialogue_corpus/archives/LaboroTVSpeech_v2.1b.tar.gz \
  --out_dir /home/acg17145sv/experiments/0162_dialogue_model/LaboroTVSpeech/archive \
  --device cuda:0 \
  --samples_per_shard 100000 \
  --resume


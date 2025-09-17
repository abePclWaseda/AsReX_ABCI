#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=100:00:00
#PBS -j oe
#PBS -N 0162_RWCP-SP97

set -eu

echo "JOB_ID : $PBS_JOBID"
echo "WORKDIR: $PBS_O_WORKDIR"
cd   "$PBS_O_WORKDIR"

module list

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asrex310

echo "==== which python ===="
which python               
python --version

exec python -m tools.stereo_asr_whisperx_align

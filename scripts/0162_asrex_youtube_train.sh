#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=168:00:00
#PBS -J 1-36%10
#PBS -j oe

set -eu

echo "JOB_ID : $PBS_JOBID"
echo "ARRAY_ID: $PBS_ARRAY_INDEX"
echo "WORKDIR: $PBS_O_WORKDIR"
cd   "$PBS_O_WORKDIR"

module list

source ~/miniforge3/etc/profile.d/conda.sh
conda activate asrex310

echo "==== which python ===="
which python               
python --version

IDX0=$(( PBS_ARRAY_INDEX - 1 ))
START=$(( IDX0 * 11 ))
END=$(( START + 10 ))

DIRS=""
for i in $(seq $START $END); do
    printf -v D "%05d-of-00396" "$i"
    DIRS="$DIRS $D"
done
echo "processing sub-dirs:$DIRS"

exec python -m tools.asrex_espnet_y_t --dirs $DIRS

#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -v RTYPE=rt_HF,USE_SSH=1
#PBS -l select=1:ngpus=8
#PBS -l walltime=168:00:00
#PBS -J 1-179%20
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

# ジョブアレイIDから処理するサブディレクトリを計算
IDX0=$(( PBS_ARRAY_INDEX - 1 ))
START=$(( IDX0 * 8 ))
END=$(( START + 7 ))

DIRS=""
for i in $(seq $START $END); do
    printf -v D "%05d-of-01432" "$i"
    DIRS="$DIRS $D"
done
echo "processing sub-dirs:$DIRS"

# asrex_pkgを使用して処理
# 設定ファイルを使用する場合（推奨）
# 注意: 設定ファイルのパスを実際の環境に合わせて変更してください
exec python -m asrex_pkg.cli --config config_podcast_separation.yaml --dirs $DIRS

# 注意: 
# - デバイスを指定しない場合、自動的に全てのGPU（8個）が検出されて使用されます
# - 各GPUで並列処理が実行されます
# - ジョブアレイの各ジョブが異なるサブディレクトリセットを処理するため、
#   全体として大量のデータを効率的に並列処理できます

# オプション: 特定のGPUのみを使用したい場合
# exec python -m asrex_pkg.cli --config config_jchat_separation.yaml --dirs $DIRS --devices cuda:0

# オプション: コマンドライン引数で直接指定する場合
# exec python -m asrex_pkg.cli \
#   --root /path/to/data \
#   --enable-separation \
#   --separated-dir separated \
#   --dirs $DIRS


#!/bin/bash
set -euxo pipefail

BASE_SRC_DIR="/home/acg17145sv/experiments/0215_audio_llm/datasets/llmjp-zoom1"
BASE_DST_DIR="/home/acg17145sv/experiments/0162_dialogue_model/AsReX/data/llmjp-zoom1/audio"

for i in $(seq -w 1 2000); do
    SRC_DIR="${BASE_SRC_DIR}/${i}"
    DST_DIR="${BASE_DST_DIR}/${i}"

    # SRC が存在しない場合はスキップ
    if [ ! -d "$SRC_DIR" ]; then
        echo "Skipping $SRC_DIR (not found)"
        continue
    fi

    mkdir -p "$DST_DIR"

    echo "Processing directory: $SRC_DIR"

    for f in "$SRC_DIR"/*.m4a; do
        # m4a が無い場合スキップ
        [ -e "$f" ] || { echo "No m4a files in $SRC_DIR"; break; }

        base=$(basename "$f" .m4a)
        out="$DST_DIR/${base}.wav"

        echo "Converting $f -> $out"
        ffmpeg -y -i "$f" -acodec pcm_s16le -ar 16000 "$out"
    done
done

echo "All Done!"

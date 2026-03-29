#!/bin/bash
set -euxo pipefail

SRC_DIR="data/llmjp-zoom1/test/audio"
DST_DIR="data/llmjp-zoom1/test/audio_split"

mkdir -p "$DST_DIR"

# 各ディレクトリを走査
for dir in "$SRC_DIR"/*/; do
    dir="${dir%/}"           # "data/llmjp-zoom1/test/audio/0059"
    base=$(basename "$dir")  # "0059"

    echo "Processing directory: $base"

    out_dir="$DST_DIR/$base"
    mkdir -p "$out_dir"

    shopt -s nullglob
    for wav_file in "$dir"/*.wav; do
        fname=$(basename "$wav_file")
        fname_no_ext="${fname%.wav}"

        echo "  Processing: $fname"

        # ファイル名から情報を抽出
        # 例: 0059_W01_W05_T02.wav -> base=0059, chL=W01, chR=W05, t_part=T02
        rest=${fname_no_ext#${base}_}  # "W01_W05_T02" または "W01_W05_T01_T06"
        
        # WコードとTコード部分を分離
        # W01_W05_T02 の場合: chL=W01, chR=W05, t_part=T02
        # W01_W05_T01_T06 の場合: chL=W01, chR=W05, t_part=T01_T06
        
        # 最初の2つのWコードを取得
        first_w=${rest%%_*}  # "W01"
        rest_after_first=${rest#*_}  # "W05_T02" または "W05_T01_T06"
        second_w=${rest_after_first%%_*}  # "W05"
        
        # Tコード部分を取得（最初のTから最後まで）
        t_part=$(echo "$rest" | sed -n 's/.*\(T[0-9]\+\(_T[0-9]\+\)*\).*/\1/p')
        
        # 左チャンネルと右チャンネルのファイル名を生成
        out_left="$out_dir/${base}_${first_w}_${t_part}.wav"
        out_right="$out_dir/${base}_${second_w}_${t_part}.wav"

        # 既に存在する場合はスキップ
        if [[ -f "$out_left" ]] && [[ -f "$out_right" ]]; then
            echo "    Skip existing: $out_left, $out_right"
            continue
        fi

        echo "    Splitting:"
        echo "      L -> $out_left"
        echo "      R -> $out_right"

        # 左チャンネルを抽出（モノラル）
        ffmpeg -y -i "$wav_file" -af "pan=mono|c0=FL" "$out_left"

        # 右チャンネルを抽出（モノラル）
        ffmpeg -y -i "$wav_file" -af "pan=mono|c0=FR" "$out_right"
    done
    shopt -u nullglob
done

echo "All done."



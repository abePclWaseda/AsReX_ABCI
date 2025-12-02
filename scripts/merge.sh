#!/bin/bash
set -euxo pipefail

SRC_DIR="data/llmjp-zoom1/audio"
DST_DIR="data/llmjp-zoom1/audio_merged"

mkdir -p "$DST_DIR"

# audio/0001, audio/0002, ... を走査
for dir in "$SRC_DIR"/*/; do
    dir="${dir%/}"           # "data/llmjp-zoom1/audio/0001"
    base=$(basename "$dir")  # "0001"

    echo "Processing directory: $base"

    out_dir="$DST_DIR/$base"
    mkdir -p "$out_dir"

    # Tごとにファイルをグルーピングする連想配列
    declare -A t_groups=()

    shopt -s nullglob
    for f in "$dir"/"${base}_W"*"_T"*.wav; do
        fname=$(basename "$f")

        # すでにマージ済みの 0001_W02_W03_T01.wav などを弾くため、
        # 「<ID>_W数字_T数字.wav」だけを対象にする
        if [[ ! "$fname" =~ ^${base}_W[0-9]+_T[0-9]+\.wav$ ]]; then
            continue
        fi

        # 例: fname = 0088_W03_T13.wav
        rest=${fname#${base}_}      # "W03_T13.wav"
        ch=${rest%%_*}              # "W03"
        t_with_ext=${rest#*_}       # "T13.wav"
        t=${t_with_ext%.wav}        # "T13"

        key="$t"  # Tコードごとにグルーピング

        if [[ -z "${t_groups[$key]+x}" ]]; then
            t_groups[$key]="$f"
        else
            t_groups[$key]+=" $f"
        fi
    done
    shopt -u nullglob

    # 各 Txx について 2 ファイルをステレオ結合
    for t in "${!t_groups[@]}"; do
        # 例: files = ("0088_W03_T13.wav" "0088_W06_T13.wav")
        files=(${t_groups[$t]})

        if (( ${#files[@]} != 2 )); then
            echo "  Warning: $base $t は ${#files[@]} 個のファイルしかないためスキップ (2個必要)"
            continue
        fi

        f1=${files[0]}
        f2=${files[1]}

        fname1=$(basename "$f1")
        fname2=$(basename "$f2")
        rest1=${fname1#${base}_}    # "Wxx_Tyy.wav"
        rest2=${fname2#${base}_}
        ch1=${rest1%%_*}            # "Wxx"
        ch2=${rest2%%_*}

        n1=${ch1#W}                  # 数値部分
        n2=${ch2#W}

        # W番号の小さい方を左チャンネル、大きい方を右チャンネルにする
        if (( n1 < n2 )); then
            L="$f1"; chL="$ch1"
            R="$f2"; chR="$ch2"
        else
            L="$f2"; chL="$ch2"
            R="$f1"; chR="$ch1"
        fi

        out="$out_dir/${base}_${chL}_${chR}_${t}.wav"

        if [[ -f "$out" ]]; then
            echo "  Skip existing: $out"
            continue
        fi

        echo "  Merging:"
        echo "    L($chL): $L"
        echo "    R($chR): $R"
        echo "      -> $out"

        ffmpeg -y -i "$L" -i "$R" \
        -filter_complex "\
        [0:a]pan=mono|c0=FL[aL]; \
        [1:a]pan=mono|c0=FL[aR]; \
        [aL][aR]join=inputs=2:channel_layout=stereo[aout]" \
        -map "[aout]" "$out"

    done

    unset t_groups
done

echo "All done."

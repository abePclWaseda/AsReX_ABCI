#!/bin/bash
set -euxo pipefail

SRC_DIR="data/llmjp-zoom1/audio"
DST_DIR="data/llmjp-zoom1/test/audio"
DIFF_DIRS_FILE="data/llmjp-zoom1/diff_dirs.txt"

mkdir -p "$DST_DIR"

# diff_dirs.txtに載っているディレクトリのみを処理
while IFS= read -r base || [[ -n "$base" ]]; do
    # 空行をスキップ
    [[ -z "$base" ]] && continue

    dir="$SRC_DIR/$base"
    
    # ディレクトリが存在するか確認
    if [[ ! -d "$dir" ]]; then
        echo "Warning: Directory $dir does not exist, skipping."
        continue
    fi

    echo "Processing directory: $base"

    out_dir="$DST_DIR/$base"
    mkdir -p "$out_dir"

    # ディレクトリ内のすべての.wavファイルを取得
    shopt -s nullglob
    files=("$dir"/*.wav)
    shopt -u nullglob

    if (( ${#files[@]} != 3 )); then
        echo "  Warning: $base には ${#files[@]} 個のファイルがあります (3個必要)。スキップします。"
        continue
    fi

    # ファイル名の長さでソート（長い順）
    # パス名（ファイル名）の長さで比較するため、basenameを使用
    declare -a files_with_length=()
    for f in "${files[@]}"; do
        fname=$(basename "$f")
        len=${#fname}
        files_with_length+=("$len|$f")
    done

    # 長さでソート（長い順）
    IFS=$'\n' sorted=($(sort -t'|' -k1,1rn <<<"${files_with_length[*]}"))
    unset IFS

    # 一番長いファイルを除外
    longest="${sorted[0]#*|}"
    f1="${sorted[1]#*|}"
    f2="${sorted[2]#*|}"

    fname1=$(basename "$f1")
    fname2=$(basename "$f2")
    longest_name=$(basename "$longest")

    echo "  Excluding longest filename: $longest_name"
    echo "  Using files:"
    echo "    $fname1"
    echo "    $fname2"

    # ファイル名からチャンネル情報を抽出
    # 例: 0055_W02_T10_T15.wav -> W02
    rest1=${fname1#${base}_}    # "Wxx_Tyy.wav" または "Wxx_Wyy_Tzz.wav"
    rest2=${fname2#${base}_}
    
    # 最初のWコードを取得
    ch1=${rest1%%_*}            # "Wxx"
    ch2=${rest2%%_*}

    # W番号を抽出
    n1=${ch1#W}
    n2=${ch2#W}

    # W番号の小さい方を左チャンネル、大きい方を右チャンネルにする
    if (( n1 < n2 )); then
        L="$f1"; chL="$ch1"
        R="$f2"; chR="$ch2"
    else
        L="$f2"; chL="$ch2"
        R="$f1"; chR="$ch1"
    fi

    # 出力ファイル名を生成（Tコードも含める場合）
    # ファイル名からTコード部分を抽出（T01_T06のような形式も対応）
    # 例: 1936_W01_T01_T06.wav -> T01_T06
    # 例: 1936_W01_W20_T01_T06.wav -> T01_T06
    
    # ファイル名からbase_と.wavを除いた部分を取得
    suffix1=${fname1#${base}_}
    suffix1=${suffix1%.wav}  # "W01_T01_T06" または "W01_W20_T01_T06"
    suffix2=${fname2#${base}_}
    suffix2=${suffix2%.wav}
    
    # Tコード部分を抽出（最初のTから最後まで）
    # 例: "W01_T01_T06" -> "T01_T06"
    t_part1=$(echo "$suffix1" | sed -n 's/.*\(T[0-9]\+\(_T[0-9]\+\)*\).*/\1/p')
    t_part2=$(echo "$suffix2" | sed -n 's/.*\(T[0-9]\+\(_T[0-9]\+\)*\).*/\1/p')
    
    # Tコード部分が一致することを確認
    if [[ "$t_part1" != "$t_part2" ]]; then
        echo "  Warning: T code parts differ ($t_part1 vs $t_part2), using $t_part1"
        t_part="$t_part1"
    else
        t_part="$t_part1"
    fi
    
    # Tコード部分が空の場合は、単一のTコードを探す
    if [[ -z "$t_part" ]]; then
        t1=$(echo "$fname1" | grep -oE 'T[0-9]+' | head -1)
        t2=$(echo "$fname2" | grep -oE 'T[0-9]+' | head -1)
        if [[ "$t1" == "$t2" ]]; then
            t_part="$t1"
        else
            t_part="$t1"
            echo "  Warning: Using first T code: $t_part"
        fi
    fi
    
    out="$out_dir/${base}_${chL}_${chR}_${t_part}.wav"

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

done < "$DIFF_DIRS_FILE"

echo "All done."

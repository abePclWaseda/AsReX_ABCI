#!/usr/bin/env bash
set -euo pipefail

in_root="/home/acg17145sv/experiments/0185_speech_dialogue_corpus/tabidachi/Tabidachi2109-3"
out_root="/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi/audio"
log_file="${out_root}/tabidachi-3.log"

mkdir -p "$out_root"

# ログ初期化
echo "=== Tabidachi merge started at $(date) ===" > "$log_file"

# サブディレクトリごとに処理
for dir in "$in_root"/*/; do
  base="$(basename "$dir")"
  op="${dir}${base}_operator.m4a"
  user="${dir}${base}_user.m4a"
  out="${out_root}/${base}.wav"

  if [[ ! -f "$op" ]]; then
    echo "[WARN] operator 側が見つかりません: $op" | tee -a "$log_file"
    continue
  fi
  if [[ ! -f "$user" ]]; then
    echo "[WARN] user 側が見つかりません: $user" | tee -a "$log_file"
    continue
  fi

  echo "[INFO] 🎧 結合中: $base" | tee -a "$log_file"
  if ffmpeg -y -i "$op" -i "$user" \
    -filter_complex "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]" \
    -map "[a]" "$out" >>"$log_file" 2>&1; then
    echo "[OK]   出力完了: $out" | tee -a "$log_file"
  else
    echo "[FAIL] 変換失敗: $base" | tee -a "$log_file"
  fi
done

echo "=== Tabidachi merge finished at $(date) ===" | tee -a "$log_file"

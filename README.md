# AsReX: Audio Speech Recognition and Alignment Pipeline

ステレオ音声（2チャンネル対話）の音声認識と単語レベルの時間アライメントを行うパッケージ

## 概要

AsReXは、ステレオ音声ファイル（2チャンネル = 2話者）を処理し、以下の処理を実行します：

1. **音源分離（オプション）**: ConvTasNet を用いてモノラル対話音声を 2 話者に分離
2. **音声認識（ASR）**: ReazonSpeech-ESPnet/Nemo ASR を使用して各チャンネル（または分離後の各話者）の音声を認識
3. **時間アライメント**: WhisperX を使用して単語レベルの時間情報を付与
4. **マージ**: 両話者の発話を時系列でマージ

## パッケージ構造

```
asrex_pkg/
├── __init__.py          # パッケージ初期化
├── config.py            # 設定管理
├── audio_utils.py       # 音声処理ユーティリティ
├── asr_processor.py     # ASR処理
├── align_processor.py   # アライメント処理
├── pipeline.py          # メインパイプライン
└── cli.py               # CLIエントリーポイント
```

## インストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 設定ファイルを使用（推奨）

設定ファイルを作成します：

```yaml
# config.yaml
data:
  root: /path/to/data
  audio_dir: audio
  transcripts_dir: transcripts
  alignment_dir: text

processing:
  speakers: ["A", "B"]
  sample_rate: 16000
  language: ja
  asr_model_type: espnet
  align_threads: 2
  chunk_seconds: null  # null = whole-file mode

logging:
  log_dir: null
  log_prefix: asrex
  log_errors: true
```

実行：

```bash
python -m asrex_pkg.cli --config config.yaml --dirs .
```

### 2. コマンドライン引数で設定

```bash
python -m asrex_pkg.cli \
  --root /path/to/data \
  --dirs subdir1 subdir2 \
  --asr-model espnet \
  --sample-rate 16000
```

### 3. Pythonスクリプトから使用

```python
from asrex_pkg import Config, StereoASRPipeline

# 設定を読み込み
config = Config.from_yaml("config.yaml")

# パイプラインを作成
pipeline = StereoASRPipeline(config)

# 実行
pipeline.run(sub_dirs=["."], devices=["cuda:0", "cuda:1"])
```

## コマンドラインオプション

### 基本オプション

- `--config`: 設定ファイルのパス（YAML形式）
- `--root`: データのルートディレクトリ（--configが指定されない場合に使用）
- `--dirs`: 処理するサブディレクトリ（複数指定可能、デフォルト: .）

### データディレクトリ

- `--audio-dir`: 音声ファイルのディレクトリ名（デフォルト: audio）
- `--transcripts-dir`: トランスクリプトのディレクトリ名（デフォルト: transcripts）
- `--alignment-dir`: アライメント結果のディレクトリ名（デフォルト: text）

### 処理設定

- `--asr-model`: ASRモデルのタイプ（espnet または nemo、デフォルト: espnet）
- `--sample-rate`: サンプルレート（デフォルト: 16000）
- `--language`: 言語コード（デフォルト: ja）
- `--chunk-seconds`: チャンクの長さ（秒）。指定しない場合はファイル全体を処理
- `--align-threads`: アライメント処理のスレッド数（デフォルト: 2）
- `--devices`: 使用するデバイス（例: cuda:0 cuda:1）。未指定の場合は自動検出

### ログ設定

- `--log-dir`: ログディレクトリ
- `--log-prefix`: ログファイルのプレフィックス（デフォルト: asrex）

## ディレクトリ構造

```
data/
├── audio/           # 入力音声ファイル（*.wav, モノラルも可）
├── separated/       # 音源分離結果（enable_separation=true の場合に生成）
├── transcripts/     # 話者別トランスクリプト（*.json）
│   ├── file1_A.json
│   └── file1_B.json
└── text/            # アライメント結果（*.json）
    └── file1.json
```

## 音源分離

### 目的と概要

モノラル対話音声に対して ConvTasNet ベースの音源分離（`JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`）を適用し、擬似的な 2 チャンネル音声を生成してから ASR・アライメント処理を行えます。音源分離は以下の前提で設計されています：

- 入力はモノラル音声でも良い（自動的に平均化してモノラル化）
- ConvTasNet は 16kHz を前提とするため、必要に応じてリサンプリングを実施
- 出力は 2 話者（チャンネル）のテンソルで、後続の ASR/アライメント処理にそのまま渡される

### 有効化方法

設定ファイルの `processing` セクションで以下のフラグを有効にします：

```yaml
processing:
  enable_separation: true
  separation_model_name: "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"
  save_separated: true
```

- `enable_separation`: 音源分離のオン/オフを制御します。
- `separation_model_name`: Asteroid Hub で配布されている分離モデルの識別子です。ConvTasNet 以外に差し替えることも可能です。
- `save_separated`: 分離結果を `data/separated`（または `config.data.separated_dir` で指定した場所）にキャッシュとして保存します。再実行時はキャッシュを利用し、処理を高速化します。

### 処理フロー

1. 入力音声を読み込み、モノラルに変換します（複数チャネルの場合は平均化）。
2. 必要があれば 16kHz にリサンプリングします。
3. ConvTasNet で 2 話者の波形に分離します。
4. 分離結果をキャッシュへ保存し、以降の ASR/アライメント処理はキャッシュ済みの 2 チャンネル波形を利用します。

### 注意事項

- 音源分離後のサンプルレートは常に 16kHz となります。`processing.sample_rate` が異なる値に設定されている場合でも、下流処理は 16kHz を用います。
- キャッシュファイルが存在してサンプルレートが 16kHz でない場合は自動的に再生成されます。
- GPU を使用する場合、分離モデルは ASR モデルと同じ `device` 設定を共有します。

## 出力形式

### トランスクリプト（transcripts/*_A.json, *_B.json）

```json
{
  "text": "全文",
  "segments": [
    {
      "start": 0.5,
      "end": 2.3,
      "text": "セグメントのテキスト"
    }
  ]
}
```

### アライメント結果（text/*.json）

```json
[
  {
    "speaker": "A",
    "word": "こ",
    "start": 3.648,
    "end": 4.242
  },
  {
    "speaker": "A",
    "word": "ん",
    "start": 4.242,
    "end": 4.262
  },
  {
    "speaker": "A",
    "word": "に",
    "start": 4.262,
    "end": 4.283
  },
  {
    "speaker": "A",
    "word": "ち",
    "start": 4.283,
    "end": 4.303
  },
  {
    "speaker": "A",
    "word": "は",
    "start": 4.303,
    "end": 4.897
  },
  {
    "speaker": "A",
    "word": "。",
    "start": 4.897,
    "end": 4.918
  }
]
```

## 処理モード

### Whole-file Mode（デフォルト）

ファイル全体を一度に処理します。文脈が保持されるため、精度が高い場合があります。

```yaml
processing:
  chunk_seconds: null
```

### Chunk Mode（ストリーミング）

ファイルをチャンクに分割して処理します。長いファイルやメモリが不足する場合に使用します。

```yaml
processing:
  chunk_seconds: 30  # 30秒チャンク
```

## ジョブスクリプト

ABCIで実行する場合のジョブスクリプト例：

```bash
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

python -m asrex_pkg.cli --config config.yaml --dirs .
```

## 既存スクリプトとの互換性

既存の `tools/stereo_asr_whisperx_align.py` スクリプトは、新しいパッケージを使用するように更新できます：

```python
from asrex_pkg import Config, StereoASRPipeline

config = Config.from_yaml("config.yaml")
pipeline = StereoASRPipeline(config)
pipeline.run()
```

## トラブルシューティング

### メモリ不足

- `chunk_seconds` を設定してチャンクモードを使用
- `align_threads` を減らす
- 使用するGPUの数を減らす

### エラーログ

エラーログは `{log_prefix}_errors.log` に保存されます。エラーが発生したファイルはスキップされ、処理は継続されます。

## ライセンス

（ライセンス情報をここに追加）

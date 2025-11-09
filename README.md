# AsReX: Audio Speech Recognition and Alignment Pipeline

ステレオ音声（2チャンネル対話）の音声認識と単語レベルの時間アライメントを行うパッケージ

## 概要

AsReXは、ステレオ音声ファイル（2チャンネル = 2話者）を処理し、以下の処理を実行します：

1. **音声認識（ASR）**: ReazonSpeech-ESPnet/Nemo ASR を使用して各チャンネルの音声を認識
2. **時間アライメント**: WhisperX を使用して単語レベルの時間情報を付与
3. **マージ**: 両話者の発話を時系列でマージ

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
├── audio/           # 入力音声ファイル（*.wav）
├── transcripts/     # 話者別トランスクリプト（*.json）
│   ├── file1_A.json
│   └── file1_B.json
└── text/            # アライメント結果（*.json）
    └── file1.json
```

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

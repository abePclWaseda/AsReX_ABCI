# AsReX パッケージ 使用ガイド

## クイックスタート

### 1. 設定ファイルの作成

既存の設定ファイル例をコピーして編集：

```bash
cp config_example.yaml config.yaml
# config.yamlを編集
```

### 2. 実行

```bash
python -m asrex_pkg.cli --config config.yaml --dirs .
```

## 設定ファイルのカスタマイズ

### Tabidachiデータセット用

```yaml
data:
  root: /path/to/Tabidachi
  audio_dir: audio
  transcripts_dir: transcripts
  alignment_dir: text

processing:
  speakers: ["A", "B"]
  sample_rate: 16000
  language: ja
  asr_model_type: espnet
  chunk_seconds: null  # whole-file mode
  align_threads: 2

logging:
  log_prefix: Tabidachi
```

### CallHomeデータセット用

```yaml
data:
  root: /path/to/CallHome
  audio_dir: audio
  transcripts_dir: transcripts
  alignment_dir: alignment

processing:
  speakers: ["A", "B"]
  sample_rate: 16000
  language: ja
  asr_model_type: nemo
  chunk_seconds: 30  # ストリーミングモード
  align_threads: 2

logging:
  log_prefix: callhome
```

## コマンドラインオプション

### 基本的な使い方

```bash
# 設定ファイルを使用
python -m asrex_pkg.cli --config config.yaml

# 特定のサブディレクトリを処理
python -m asrex_pkg.cli --config config.yaml --dirs subdir1 subdir2

# コマンドラインで設定を上書き
python -m asrex_pkg.cli --config config.yaml --asr-model nemo --chunk-seconds 30

# デバイスを指定
python -m asrex_pkg.cli --config config.yaml --devices cuda:0 cuda:1
```

### 設定ファイルなしで実行

```bash
python -m asrex_pkg.cli \
  --root /path/to/data \
  --audio-dir audio \
  --transcripts-dir transcripts \
  --alignment-dir text \
  --dirs .
```

## Pythonスクリプトから使用

### 基本的な使い方

```python
from asrex_pkg import Config, StereoASRPipeline

# 設定を読み込み
config = Config.from_yaml("config.yaml")

# パイプラインを作成
pipeline = StereoASRPipeline(config)

# 実行
pipeline.run()
```

### プログラムで設定をカスタマイズ

```python
from pathlib import Path
from asrex_pkg import Config, StereoASRPipeline

# 設定を作成
config = Config.from_env(Path("/path/to/data"))
config.data.audio_dir = "audio"
config.data.transcripts_dir = "transcripts"
config.data.alignment_dir = "text"

# 処理設定
config.processing.asr_model_type = "espnet"
config.processing.chunk_seconds = None  # whole-file mode
config.processing.align_threads = 2

# ログ設定
config.logging.log_prefix = "my_dataset"

# パイプラインを作成して実行
pipeline = StereoASRPipeline(config)
pipeline.run(sub_dirs=["."], devices=["cuda:0"])
```

### 単一ファイルを処理

```python
from pathlib import Path
from asrex_pkg import Config, StereoASRPipeline

config = Config.from_yaml("config.yaml")
pipeline = StereoASRPipeline(config)

# 単一ファイルを処理
wav_path = Path("/path/to/audio/file.wav")
success = pipeline.process_file(wav_path, device="cuda:0")
```

## 既存スクリプトからの移行

### 旧スクリプト（tools/stereo_asr_whisperx_align.py）

```python
# 旧: ハードコードされたパスと設定
ROOT = Path("/home/.../Tabidachi")
IN_ROOT = ROOT / "audio"
# ...
```

### 新パッケージ版

```python
# 新: 設定ファイルまたはプログラムで設定
from asrex_pkg import Config, StereoASRPipeline

config = Config.from_yaml("config.yaml")
pipeline = StereoASRPipeline(config)
pipeline.run()
```

## トラブルシューティング

### メモリ不足エラー

- `chunk_seconds` を設定（例: 30）
- `align_threads` を減らす（例: 1）
- 使用するGPUの数を減らす

### デバイスエラー

- `--devices` オプションで明示的に指定
- `CUDA_VISIBLE_DEVICES` 環境変数を使用

### パスエラー

- 設定ファイルの `data.root` を確認
- 相対パスではなく絶対パスを使用

## パフォーマンスの最適化

### Whole-file Mode vs Chunk Mode

- **Whole-file Mode** (`chunk_seconds: null`): 精度が高いが、メモリ使用量が多い
- **Chunk Mode** (`chunk_seconds: 30`): メモリ使用量が少ないが、文脈が切れる可能性

### 並列処理

- 複数GPUを使用: `--devices cuda:0 cuda:1 cuda:2 cuda:3`
- アライメントスレッド数: `--align-threads 2`（GPUメモリに応じて調整）

## 出力ファイル

### トランスクリプト

- 場所: `{root}/{transcripts_dir}/**/*_{speaker}.json`
- 形式: `{"text": "...", "segments": [...]}`

### アライメント結果

- 場所: `{root}/{alignment_dir}/**/*.json`
- 形式: `[{"speaker": "A", "word": "...", "start": 0.5, "end": 1.2}, ...]`

## エラーログ

エラーが発生したファイルは `{log_prefix}_errors.log` に記録されます。
処理は継続され、エラーが発生したファイルはスキップされます。



# 既存コードからの移行ガイド

## 概要

このドキュメントは、既存の `tools/stereo_asr_whisperx_align.py` から新しい `asrex_pkg` パッケージへの移行方法を説明します。

## 主な変更点

### 1. ハードコードされたパスの削除

**旧:**
```python
ROOT = Path("/home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi")
IN_ROOT = ROOT / "audio"
TXT_ROOT = ROOT / "transcripts"
ALN_ROOT = ROOT / "text"
```

**新:**
```python
config = Config.from_yaml("config.yaml")
# または
config = Config.from_env(Path("/path/to/data"))
```

### 2. 設定の外部化

**旧:**
```python
SPEAKERS = ("A", "B")
SAMPLE_RATE = 16_000
```

**新:**
```yaml
# config.yaml
processing:
  speakers: ["A", "B"]
  sample_rate: 16000
```

### 3. モジュール化

**旧:**
- すべての処理が1つのファイルに記述

**新:**
- `config.py`: 設定管理
- `audio_utils.py`: 音声処理ユーティリティ
- `asr_processor.py`: ASR処理
- `align_processor.py`: アライメント処理
- `pipeline.py`: メインパイプライン
- `cli.py`: CLIエントリーポイント

## 移行手順

### ステップ1: 設定ファイルの作成

既存の設定を `config.yaml` に移行：

```yaml
data:
  root: /home/acg17145sv/experiments/0162_dialogue_model/data_stage_3/Tabidachi
  audio_dir: audio
  transcripts_dir: transcripts
  alignment_dir: text

processing:
  speakers: ["A", "B"]
  sample_rate: 16000
  language: ja
  asr_model_type: espnet
  chunk_seconds: null
  align_threads: 2

logging:
  log_prefix: Tabidachi
  log_errors: true
```

### ステップ2: スクリプトの更新

**旧スクリプト（tools/stereo_asr_whisperx_align.py）:**
```python
# 250行以上のコード
ROOT = Path("...")
# ...
```

**新スクリプト:**
```python
from asrex_pkg import Config, StereoASRPipeline

config = Config.from_yaml("config.yaml")
pipeline = StereoASRPipeline(config)
pipeline.run()
```

### ステップ3: ジョブスクリプトの更新

**旧（scripts/0162_stereo_asr_whisperx_align.sh）:**
```bash
exec python -m tools.stereo_asr_whisperx_align
```

**新（scripts/0162_stereo_asr_whisperx_align_new.sh）:**
```bash
exec python -m asrex_pkg.cli --config config.yaml --dirs .
```

## 機能の比較

| 機能 | 旧コード | 新パッケージ |
|------|---------|------------|
| 設定ファイル | ❌ | ✅ YAML |
| コマンドライン引数 | 限定的 | ✅ 豊富 |
| モジュール化 | ❌ | ✅ |
| エラーハンドリング | 基本的 | ✅ 改善 |
| ログ設定 | ハードコード | ✅ 設定可能 |
| 再利用性 | 低い | ✅ 高い |
| テスト容易性 | 低い | ✅ 高い |

## 互換性

### 出力形式

出力ファイルの形式は**同じ**です：

- トランスクリプト: `{file}_{speaker}.json`
- アライメント: `{file}.json`

### ディレクトリ構造

ディレクトリ構造も**同じ**です：

```
data/
├── audio/
├── transcripts/
└── text/
```

## 既存データとの互換性

既存の処理済みファイルはそのまま使用できます。新しいパッケージは既存の出力を認識し、処理済みファイルをスキップします。

## 段階的な移行

### オプション1: 並行運用

既存のスクリプトと新しいパッケージを並行して使用：

```bash
# 既存スクリプト
python -m tools.stereo_asr_whisperx_align

# 新パッケージ
python -m asrex_pkg.cli --config config.yaml
```

### オプション2: 段階的置き換え

1. 新しいパッケージでテスト実行
2. 出力を比較
3. 問題がなければ完全移行

## トラブルシューティング

### 設定ファイルが見つからない

```bash
# エラー: Config file not found
# 解決: 設定ファイルのパスを確認
python -m asrex_pkg.cli --config /absolute/path/to/config.yaml
```

### パスエラー

```bash
# エラー: audio directory not found
# 解決: config.yamlのdata.rootを確認
```

### インポートエラー

```bash
# エラー: No module named 'asrex_pkg'
# 解決: PYTHONPATHを設定
export PYTHONPATH="${PYTHONPATH}:/path/to/AsReX"
```

## サポート

問題が発生した場合は、以下を確認してください：

1. 設定ファイルの形式が正しいか
2. 必要な依存パッケージがインストールされているか
3. パスが正しいか（絶対パス推奨）

## 次のステップ

1. ✅ 設定ファイルを作成
2. ✅ 新しいパッケージでテスト実行
3. ✅ 出力を確認
4. ✅ 本番環境に適用



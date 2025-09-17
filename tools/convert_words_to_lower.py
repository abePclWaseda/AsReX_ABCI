import json
import os
from glob import glob

# 入力ディレクトリと出力ディレクトリ
in_dir = "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet/podcast_test/00000-of-00001/cuts.000000/"
out_dir = "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet_lower/podcast_test/00000-of-00001/cuts.000000/"

# 出力先ディレクトリを作成
os.makedirs(out_dir, exist_ok=True)

# JSONファイルをすべて処理
for in_path in glob(os.path.join(in_dir, "*.json")):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "word" in item and isinstance(item["word"], str):
            item["word"] = item["word"].lower()

    # 出力先パスを決定
    out_path = os.path.join(out_dir, os.path.basename(in_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"変換して保存しました → {out_path}")

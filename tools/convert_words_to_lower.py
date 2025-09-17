import json
import os

# 入力と出力のルートディレクトリ
in_root = "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet/podcast_train"
out_root = (
    "/home/acg17145sv/experiments/0162_dialogue_model/J-CHAT/text_by_espnet_lower/podcast_train"
)

# 再帰的に探索
for dirpath, _, filenames in os.walk(in_root):
    for filename in filenames:
        if not filename.endswith(".json"):
            continue

        in_path = os.path.join(dirpath, filename)

        # 出力先のディレクトリ構造を入力と同じにする
        rel_path = os.path.relpath(in_path, in_root)
        out_path = os.path.join(out_root, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # JSON 読み込み
        with open(in_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"読み込み失敗: {in_path} ({e})")
                continue

        # word を小文字化
        for item in data:
            if "word" in item and isinstance(item["word"], str):
                item["word"] = item["word"].lower()

        # JSON 書き込み
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"変換して保存しました → {out_path}")

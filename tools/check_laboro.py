from pathlib import Path
import json
import webdataset as wds

# ここを shard の出力ディレクトリに変更
out_dir = Path("/path/to/output_dir")  # 例: Path("/groups/.../shards")
pattern = str(out_dir / "shard-*.tar")

dataset = wds.WebDataset(pattern, shardshuffle=False)

checked = 0
ok_pairs = 0
results = []

for sample in dataset:
    # 念のため wav があるサンプルだけ数える
    if "wav" not in sample:
        continue

    key = sample.get("__key__", "unknown")
    url = sample.get("__url__", "unknown")  # どのshardから来たか
    has_json = "json" in sample

    # json があればパースして確認
    json_ok = False
    json_len = None
    if has_json:
        try:
            data = json.loads(sample["json"].decode("utf-8"))
            # list であることだけ簡易チェック（あなたの出力は list のはず）
            if isinstance(data, list):
                json_ok = True
                json_len = len(data)
        except Exception as e:
            json_ok = False

    results.append(
        {
            "key": key,
            "shard": url,
            "wav": True,
            "json_exists": has_json,
            "json_parsed": json_ok,
            "json_items": json_len,
        }
    )

    checked += 1
    if has_json and json_ok:
        ok_pairs += 1

    # 10サンプルで打ち切り
    if checked >= 10:
        break

# 表示
print("=== First 10 wav samples (or fewer if not enough) ===")
for i, r in enumerate(results, 1):
    print(f"[{i}] key={r['key']}")
    print(f"     shard={r['shard']}")
    print(
        f"     wav=OK, json_exists={r['json_exists']}, json_parsed={r['json_parsed']}, json_items={r['json_items']}"
    )

print("\n--- Summary ---")
print(f"Checked wav samples: {checked}")
print(f"Pairs with valid JSON: {ok_pairs}/{checked}")

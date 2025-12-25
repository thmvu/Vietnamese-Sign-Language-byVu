import csv
import json
from pathlib import Path

CSV_PATH = "Dataset/Text/label_clean.csv"
LABEL_MAP_PATH = "Logs/label_map.json"

Path("Logs").mkdir(exist_ok=True)

# nếu đã có label_map thì load (để KHÔNG đổi ID cũ)
if Path(LABEL_MAP_PATH).exists():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
else:
    label_map = {}

# tìm ID tiếp theo
used_ids = set(label_map.values())
next_id = max(used_ids) + 1 if used_ids else 0

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["LABEL"].strip()

        if label not in label_map:
            label_map[label] = next_id
            next_id += 1

# ghi lại json
with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("✅ DONE")
print(f"📌 Tổng label: {len(label_map)}")
print(f"📄 Saved: {LABEL_MAP_PATH}")

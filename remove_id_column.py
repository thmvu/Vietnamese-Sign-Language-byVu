import csv

INPUT = "Dataset/Text/label.csv"
OUTPUT = "Dataset/Text/label_clean.csv"

with open(INPUT, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = [{"VIDEO": r["VIDEO"], "LABEL": r["LABEL"]} for r in reader]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["VIDEO", "LABEL"])
    writer.writeheader()
    writer.writerows(rows)

print("✅ DONE")
print("📄 File mới:", OUTPUT)

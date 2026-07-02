from pathlib import Path
import shutil
from collections import Counter

LABEL_DIRS = [
    Path("data/train/labels"),
    Path("data/valid/labels"),
    Path("data/test/labels"),
]

# Eski düzen:
# 0 = UAI
# 1 = UAP
# 2 = vehicle
# 3 = person

# Yeni yarışma düzeni:
# 0 = vehicle
# 1 = person
# 2 = UAP
# 3 = UAI

CLASS_MAP = {
    "0": "3",  # UAI -> UAI
    "1": "2",  # UAP -> UAP
    "2": "0",  # vehicle -> vehicle
    "3": "1",  # person -> person
}

old_counts = Counter()
new_counts = Counter()
changed_lines = 0
processed_files = 0

for label_dir in LABEL_DIRS:
    if not label_dir.exists():
        print(f"Klasör bulunamadı: {label_dir}")
        continue

    backup_dir = label_dir.parent / "labels_backup_before_competition_order"

    if not backup_dir.exists():
        shutil.copytree(label_dir, backup_dir)
        print(f"Yedek alındı: {backup_dir}")
    else:
        print(f"Yedek zaten var: {backup_dir}")

    for txt_file in label_dir.glob("*.txt"):
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) >= 5:
                old_id = parts[0]
                old_counts[old_id] += 1

                if old_id not in CLASS_MAP:
                    raise ValueError(
                        f"Bilinmeyen class id bulundu: {old_id} | Dosya: {txt_file}"
                    )

                new_id = CLASS_MAP[old_id]
                parts[0] = new_id
                new_counts[new_id] += 1

                if old_id != new_id:
                    changed_lines += 1

            new_lines.append(" ".join(parts))

        txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        processed_files += 1

print("\n========== DÖNÜŞÜM BİTTİ ==========")
print(f"İşlenen dosya: {processed_files}")
print(f"Değişen etiket satırı: {changed_lines}")

print("\nEski class sayıları:")
print(f"0 UAI     : {old_counts['0']}")
print(f"1 UAP     : {old_counts['1']}")
print(f"2 vehicle : {old_counts['2']}")
print(f"3 person  : {old_counts['3']}")

print("\nYeni class sayıları:")
print(f"0 vehicle : {new_counts['0']}")
print(f"1 person  : {new_counts['1']}")
print(f"2 UAP     : {new_counts['2']}")
print(f"3 UAI     : {new_counts['3']}")

print("\nYeni data.yaml şöyle olmalı:")
print("""
nc: 4
names: ['vehicle', 'person', 'UAP', 'UAI']
""")
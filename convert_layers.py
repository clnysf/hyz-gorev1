from pathlib import Path

LABEL_DIRS = [
    Path("data/train/labels"),
    Path("data/valid/labels"),
    Path("data/test/labels"),   # İstersen testi de dönüştür
]

CLASS_MAP = {
    "0": "0",  # UAI -> UAI
    "1": "1",  # UAP -> UAP
    "2": "2",  # car -> vehicle
    "3": "2",  # harvester -> vehicle
    "4": "3",  # person -> person
}

changed_files = 0
changed_lines = 0

for label_dir in LABEL_DIRS:

    if not label_dir.exists():
        continue

    for txt_file in label_dir.glob("*.txt"):

        lines = txt_file.read_text(encoding="utf-8").splitlines()

        new_lines = []
        file_changed = False

        for line in lines:

            parts = line.split()

            if len(parts) >= 5:

                old = parts[0]
                new = CLASS_MAP[old]

                if old != new:
                    parts[0] = new
                    changed_lines += 1
                    file_changed = True

            new_lines.append(" ".join(parts))

        if file_changed:
            txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1

print(f"Değişen dosya: {changed_files}")
print(f"Değişen etiket: {changed_lines}")
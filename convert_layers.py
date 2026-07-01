from pathlib import Path
import shutil

LABEL_DIRS = [
    Path("data/train/labels"),
    Path("data/valid/labels"),
    Path("data/test/labels"),
]

CLASS_MAP = {
    "0": "3",
    "1": "2",
    "2": "0",
    "3": "1",
}

for label_dir in LABEL_DIRS:
    if not label_dir.exists():
        print("Klasör yok:", label_dir)
        continue

    backup_dir = label_dir.parent / "labels_backup"

    if not backup_dir.exists():
        shutil.copytree(label_dir, backup_dir)
        print("Yedek alındı:", backup_dir)

    changed_lines = 0
    processed_files = 0

    for txt_file in label_dir.glob("*.txt"):
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) >= 5:
                old_id = parts[0]

                if old_id in CLASS_MAP:
                    parts[0] = CLASS_MAP[old_id]

                    if old_id != parts[0]:
                        changed_lines += 1

            new_lines.append(" ".join(parts))

        txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        processed_files += 1

    print(f"{label_dir} işlendi.")
    print(f"Dosya: {processed_files}")
    print(f"Değişen etiket: {changed_lines}")
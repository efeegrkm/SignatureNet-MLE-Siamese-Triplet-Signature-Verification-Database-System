import os, csv

ROOT_DIR = "data/raw"
OUT_CSV = "data/labels.csv"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm"}

rows = []

# hem train hem test altını tara
for split in ["train", "test"]:
    base = os.path.join(ROOT_DIR, split)
    if not os.path.exists(base):
        continue

    for folder in sorted(os.listdir(base)):
        full_path = os.path.join(base, folder)
        if not os.path.isdir(full_path):
            continue

        person_id = folder.replace("_forg", "")
        label = "forgery" if "forg" in folder.lower() else "genuine"

        for file in os.listdir(full_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in VALID_EXTS:
                filepath = os.path.join(full_path, file).replace("\\", "/")
                rows.append([filepath, label, person_id])

os.makedirs("data", exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "label", "person_id"])
    writer.writerows(rows)

print(f"[OK] {len(rows)} kayıt bulundu, kaydedildi → {OUT_CSV}")

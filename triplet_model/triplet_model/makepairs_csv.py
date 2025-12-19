import os
import csv
import random
from pathlib import Path
from itertools import combinations

# ====== AYARLAR ======
# Bu dosyanın yeri: .../DB-based-signature-verification/triplet_model/triplet_model/makepairs_csv.py

# __file__ -> makepairs_csv.py
# parents[0] -> triplet_model (içteki)
# parents[1] -> triplet_model (dıştakı)
# parents[2] -> DB-based-signature-verification (ANA KÖK)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_DIR = PROJECT_ROOT / "triplet_model"

SPLIT_ROOT = DATA_ROOT_DIR / "sign_data" / "split"
VAL_DIR = SPLIT_ROOT / "val"
TEST_DIR = SPLIT_ROOT / "test"

# CSV'lerin kaydedileceği yer
CSV_OUTPUT_DIR = DATA_ROOT_DIR / "data"
VAL_CSV = CSV_OUTPUT_DIR / "val_pairs.csv"
TEST_CSV = CSV_OUTPUT_DIR / "test_pairs.csv"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
random.seed(42)

def list_images(folder: Path):
    if not folder.exists():
        # Klasör yoksa boş liste dön
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

def build_pairs_for_split(split_dir: Path):
    all_pairs = []
    
    if not split_dir.exists():
        print(f"HATA: Klasör bulunamadı -> {split_dir}")
        return []

    # _forg olmayan klasörleri (kişileri) bul
    person_ids = sorted(
        d.name for d in split_dir.iterdir()
        if d.is_dir() and not d.name.endswith("_forg")
    )

    print(f"{split_dir.name} klasöründe {len(person_ids)} kişi bulundu.")

    for pid in person_ids:
        real_dir = split_dir / pid
        forg_dir = split_dir / f"{pid}_forg"

        real_imgs = list_images(real_dir)
        forg_imgs = list_images(forg_dir)

        # Eğer yeterli gerçek imza yoksa atla
        if len(real_imgs) < 2:
            continue

        # --- POZİTİF ÇİFTLER (Gerçek - Gerçek) ---
        # Aynı klasördeki resimlerin ikili kombinasyonları
        for a, b in combinations(real_imgs, 2):
            # Dosya yollarını PROJECT_ROOT'a göre göreceli yapıyoruz
            p1 = a.relative_to(PROJECT_ROOT).as_posix()
            p2 = b.relative_to(PROJECT_ROOT).as_posix()
            all_pairs.append((p1, p2, 1)) # 1 = Aynı Kişi

        # --- NEGATİF ÇİFTLER (Gerçek - Sahte) ---
        for ra in real_imgs:
            for fb in forg_imgs:
                p1 = ra.relative_to(PROJECT_ROOT).as_posix()
                p2 = fb.relative_to(PROJECT_ROOT).as_posix()
                all_pairs.append((p1, p2, 0)) # 0 = Farklı/Sahte

    random.shuffle(all_pairs)
    print(f"{split_dir.name} için toplam {len(all_pairs)} çift oluşturuldu.")
    return all_pairs

def write_csv(pairs, csv_path: Path):
    if not pairs:
        print(f"Uyarı: Kaydedilecek çift yok -> {csv_path}")
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(pairs)
    print(f"CSV Kaydedildi: {csv_path}")

# ==========================================
# İŞTE EKSİK OLAN KISIM: KODU ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    print(f"Ana Dizin (Project Root): {PROJECT_ROOT}")
    
    print("\n--- Validation (Val) Hazırlanıyor ---")
    val_pairs = build_pairs_for_split(VAL_DIR)
    write_csv(val_pairs, VAL_CSV)

    print("\n--- Test Hazırlanıyor ---")
    test_pairs = build_pairs_for_split(TEST_DIR)
    write_csv(test_pairs, TEST_CSV)
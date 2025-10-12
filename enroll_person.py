# scripts/enroll_person.py
# Kullanım:
#   python scripts/enroll_person.py --person_id 999 --name "Test" --images data/raw/test_person/999/*.jpg
#
# Ne yapar?
# 1) persons.csv'yi güvenli biçimde oluşturur/düzeltir ve kişiyi ekler (yoksa)
# 2) labels.csv'ye genuine kayıtları ekler (güvenli başlık düzeltmesiyle)
# 3) Görselleri işler -> data/processed/raw/... altına yazar
# 4) Siamese kullanmıyorsan klasik embedding/indeks, burada ise SADECE klasik indeks güncellenir.
#    (Siamese ile arama için build_index_siamese.py'yi ayrıca çalıştır.)
import argparse, os, csv, glob, joblib
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors

# --- Proje yolları
PERSONS_CSV = "data/persons.csv"
LABELS_CSV  = "data/labels.csv"
PROCESSED_ROOT = "data/processed"
EMBED_NPY   = "data/embeddings/embeddings.npy"
META_CSV    = "data/embeddings/meta.csv"
INDEX_FILE  = "models/nn_index.joblib"

# --- KLASIK embedding (4104-dim) - embedding.py'den kopya bağımsız minimal sürüm
def preprocess_basic(img):
    g = cv2.GaussianBlur(img, (3,3), 0)
    th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,35,15)
    h, w = th.shape[:2]; size = max(h, w)
    top = (size - h)//2; bottom = size - h - top
    left = (size - w)//2; right = size - w - left
    sq = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(sq, (224,224), interpolation=cv2.INTER_AREA)
    return out

def embed_from_processed(img224):
    x = cv2.resize(img224, (64,64), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    flat = x.reshape(-1)  # 4096
    sx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    hist, _ = np.histogram(mag, bins=8, range=(0.0, mag.max() if mag.max()>0 else 1.0))
    hist = hist.astype(np.float32); hist = hist / (np.linalg.norm(hist)+1e-8)
    v = np.concatenate([flat, hist], axis=0).astype(np.float32)  # 4104
    v = v / (np.linalg.norm(v)+1e-8)
    return v

# --- Yardımcılar
def processed_path(raw_path):
    # raw_path: data/raw/... -> data/processed/raw/...
    rel = os.path.relpath(raw_path, start="data").replace("\\","/")
    return os.path.join(PROCESSED_ROOT, rel).replace("\\","/")

def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.dirname(EMBED_NPY), exist_ok=True)
    os.makedirs("models", exist_ok=True)

def read_csv_rows(path):
    if not os.path.exists(path): return []
    with open(path, newline="", encoding="utf-8") as f:
        try:
            rdr = csv.DictReader(f)
            if rdr.fieldnames and all(k in rdr.fieldnames for k in ["person_id","person_name"]) and path.endswith("persons.csv"):
                return list(rdr)
            if rdr.fieldnames and all(k in rdr.fieldnames for k in ["filepath","label","person_id"]) and path.endswith("labels.csv"):
                return list(rdr)
            # Başlık uyumsuzsa alt fall-back
        except Exception:
            pass
    # Başlık uyumsuz: ham satırları reader ile al
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            rows.append(row)
    return rows

def fix_or_init_persons_csv():
    """persons.csv'yi person_id,person_name başlığına getirir."""
    if not os.path.exists(PERSONS_CSV):
        with open(PERSONS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["person_id","person_name"])
        return

    rows = read_csv_rows(PERSONS_CSV)
    # Eğer DictReader ile uygun geldi ise dokunma
    if rows and isinstance(rows[0], dict):
        # Doğru başlıkta ise çık
        return
    # Aksi halde satırları dönüştür
    cleaned = []
    for r in rows:
        if isinstance(r, list):
            if len(r) >= 2 and r[0] != "person_id":
                cleaned.append([r[0], r[1]])
    with open(PERSONS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["person_id","person_name"]); w.writerows(cleaned)

def fix_or_init_labels_csv():
    """labels.csv'yi filepath,label,person_id başlığına getirir (varsa dönüştürür)."""
    if not os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["filepath","label","person_id"])
        return

    rows = read_csv_rows(LABELS_CSV)
    if rows and isinstance(rows[0], dict):
        return
    cleaned = []
    for r in rows:
        if isinstance(r, list):
            if len(r) >= 3 and r[0] != "filepath":
                cleaned.append([r[0], r[1], r[2]])
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath","label","person_id"]); w.writerows(cleaned)

def get_existing_person_ids():
    fix_or_init_persons_csv()
    ids = set()
    with open(PERSONS_CSV, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if "person_id" in r:
                ids.add(r["person_id"])
    return ids

def ensure_person(person_id, name):
    fix_or_init_persons_csv()
    existing = get_existing_person_ids()
    if person_id not in existing:
        with open(PERSONS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow([person_id, name])

def append_labels(genuine_paths, person_id):
    fix_or_init_labels_csv()
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p in genuine_paths:
            w.writerow([p.replace("\\","/"), "genuine", person_id])

def ensure_processed(raw_paths):
    out_files = []
    for rp in raw_paths:
        img = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[SKIP] okunamadı:", rp); continue
        proc = preprocess_basic(img)
        outp = processed_path(rp)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        cv2.imwrite(outp, proc)
        out_files.append(outp)
    return out_files

def load_existing_embeddings():
    if os.path.exists(EMBED_NPY):
        X = np.load(EMBED_NPY)
    else:
        X = np.zeros((0, 4104), dtype=np.float32)
    meta = []
    if os.path.exists(META_CSV):
        with open(META_CSV, newline="", encoding="utf-8") as f:
            meta = list(csv.DictReader(f))
    return X, meta

def save_embeddings_and_index(X, meta_rows):
    np.save(EMBED_NPY, X)
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath","label","person_id"]); w.writerows(meta_rows)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(X)
    joblib.dump({"nn": nn, "meta_csv": META_CSV, "embeddings": EMBED_NPY}, INDEX_FILE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person_id", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--images", nargs="+", required=True, help="data/raw/... içindeki genuine imzalar (wildcard destekler).")
    args = ap.parse_args()

    ensure_dirs()
    ensure_person(args.person_id, args.name)

    # glob ile imzaları çek
    raw_paths = []
    for pattern in args.images:
        raw_paths.extend(glob.glob(pattern))
    raw_paths = [p for p in raw_paths if os.path.isfile(p)]
    if not raw_paths:
        print("Görüntü bulunamadı. --images desenini kontrol et.")
        return

    processed_files = ensure_processed(raw_paths)
    append_labels(raw_paths, args.person_id)

    X_old, meta_old = load_existing_embeddings()
    new_vecs = []
    new_meta = []
    for rp, pp in zip(raw_paths, processed_files):
        v = embed_from_processed(cv2.imread(pp, cv2.IMREAD_GRAYSCALE))
        new_vecs.append(v)
        new_meta.append([pp, "genuine", args.person_id])

    if new_vecs:
        X_new = np.stack(new_vecs, axis=0)
        X = np.concatenate([X_old, X_new], axis=0) if X_old.size else X_new
        # meta_old DictReader listesi ise dönüştür
        if meta_old and isinstance(meta_old[0], dict):
            meta_old = [[m["filepath"], m["label"], m["person_id"]] for m in meta_old]
        meta_rows = meta_old + new_meta
    else:
        X = X_old
        if meta_old and isinstance(meta_old[0], dict):
            meta_rows = [[m["filepath"], m["label"], m["person_id"]] for m in meta_old]
        else:
            meta_rows = meta_old

    save_embeddings_and_index(X, meta_rows)
    print(f"[OK] {args.person_id} için {len(new_vecs)} örnek eklendi ve indeks güncellendi.")
    print(f"[INFO] Klasik indeks güncellendi: {INDEX_FILE}")
    print(f"[NOTE] Siamese ile arama istersen: python scripts/build_index_siamese.py")

if __name__ == "__main__":
    main()

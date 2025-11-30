# scripts/build_index.py
import os, csv, joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

from embedding import embed_from_file

LABELS_CSV = "data/labels.csv"
PROCESSED_ROOT = "data/processed"
EMBED_NPY = "data/embeddings/embeddings.npy"
META_CSV = "data/embeddings/meta.csv"
INDEX_FILE = "models/nn_index.joblib"

def processed_path(raw_path):
    # raw_path: data/raw/...  ->  data/processed/...
    rel = os.path.relpath(raw_path, start="data").replace("\\", "/")
    return os.path.join("data/processed", rel).replace("\\", "/")

def load_rows():
    rows = []
    with open(LABELS_CSV, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def main():
    rows = load_rows()
    X = []
    meta = []
    miss = 0

    for r in rows:
        p = processed_path(r["filepath"])
        if not os.path.exists(p):
            miss += 1
            continue
        try:
            v = embed_from_file(p, already_processed=True)  # processed olduğu için
            X.append(v)
            meta.append([p, r["label"], r["person_id"]])
        except Exception as e:
            print("[WARN]", e)

    X = np.stack(X, axis=0) if len(X)>0 else np.zeros((0, 4104), dtype=np.float32)
    os.makedirs(os.path.dirname(EMBED_NPY), exist_ok=True)
    np.save(EMBED_NPY, X)

    with open(META_CSV, "w", newline='', encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath","label","person_id"]); w.writerows(meta)

    os.makedirs("models", exist_ok=True)
    # cosine distance -> benzerlik = 1 - distance
    nn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="auto")
    nn.fit(X)
    joblib.dump({"nn": nn, "meta_csv": META_CSV, "embeddings": EMBED_NPY}, INDEX_FILE)

    print(f"[OK] embeddings: {X.shape}, kaçırılan processed: {miss}")
    print(f"[SAVED] {EMBED_NPY}, {META_CSV}, {INDEX_FILE}")

if __name__ == "__main__":
    main()

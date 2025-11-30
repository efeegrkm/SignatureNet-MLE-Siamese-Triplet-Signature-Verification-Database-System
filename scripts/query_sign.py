# scripts/query_sign.py
import os, joblib, numpy as np, cv2, argparse, csv
from embedding import embed_from_file

INDEX_FILE = "models/nn_index.joblib"
THRESHOLD = 0.25  # cosine distance eşiği (0.25 altı = aynı kişi)

def load_index():
    data = joblib.load(INDEX_FILE)
    return data["nn"], data["meta_csv"], data["embeddings"]

def load_meta(meta_csv):
    with open(meta_csv, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def cosine_sim(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

def query(image_path, claimed_id=None):
    nn, meta_csv, emb_npy = load_index()
    meta = load_meta(meta_csv)
    X = np.load(emb_npy)
    v = embed_from_file(image_path, already_processed=False).reshape(1, -1)

    dist, idx = nn.kneighbors(v, n_neighbors=3, return_distance=True)
    dist, idx = dist[0], idx[0]

    print("\n🔍 En yakın 3 eşleşme:")
    for i, d in zip(idx, dist):
        print(f"→ {meta[i]['person_id']} | {meta[i]['label']} | {meta[i]['filepath']} | uzaklık={d:.3f}")

    best = meta[idx[0]]
    best_dist = dist[0]
    predicted_id = best["person_id"]

    if claimed_id:
        print(f"\nİddia edilen kişi: {claimed_id}")
        if predicted_id == claimed_id and best_dist < THRESHOLD:
            print(f"✅ İmza {claimed_id} kişisine ait (distance={best_dist:.3f})")
        else:
            print(f"⚠️ Taklit veya farklı kişi (en yakın {predicted_id}, distance={best_dist:.3f})")
    else:
        print(f"\nTahmin: kişi {predicted_id}, benzerlik distance={best_dist:.3f}")
        if best_dist < THRESHOLD:
            print("→ Bu imza zaten veri tabanında olabilir (aynı kişi).")
        else:
            print("→ Yeni bir kişi olabilir (distance yüksek).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", help="Sorgulanacak imza dosyası")
    ap.add_argument("--claimed", help="İmzanın iddia edilen sahibi (ör: 001)")
    args = ap.parse_args()

    if not os.path.exists(args.image_path):
        print("Dosya bulunamadı:", args.image_path)
    else:
        query(args.image_path, claimed_id=args.claimed)

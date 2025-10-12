# scripts/calibrate_threshold.py
import os, csv, numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from embedding import embed_from_file

LABELS_CSV = "data/labels.csv"
PROCESSED_ROOT = "data/processed"

def processed_path(raw_path):
    import os
    rel = os.path.relpath(raw_path, start="data").replace("\\","/")
    return os.path.join("data/processed", rel).replace("\\","/")

def cos_dist(a, b):
    return 1 - (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

def load_meta():
    items = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = processed_path(r["filepath"])
            if os.path.exists(p):
                items.append({"path": p, "pid": r["person_id"], "label": r["label"]})
    return items

def main():
    items = load_meta()
    # Kişiye göre ayır
    per_pid = defaultdict(list)
    for it in items:
        per_pid[it["pid"]].append(it)

    g_g, g_f = [], []  # genuine-genuine ve genuine-forgery mesafeleri

    for pid, arr in per_pid.items():
        # Bu kişiye ait genuine ve forgery ayrımı
        genu = [a for a in arr if a["label"]=="genuine"]
        forg = [a for a in arr if a["label"]=="forgery"]
        if len(genu) < 2:  # en az iki genuine lazım
            continue

        # Tüm genuine’ların embeddinglerini al
        vecs_g = []
        for a in genu:
            v = embed_from_file(a["path"], already_processed=True)
            vecs_g.append(v)
        vecs_g = np.stack(vecs_g, axis=0)

        # genuine-genuine çiftleri
        for i in range(len(vecs_g)):
            for j in range(i+1, len(vecs_g)):
                g_g.append(cos_dist(vecs_g[i], vecs_g[j]))

        # genuine-forgery çiftleri (aynı kişi id’si altında)
        if forg:
            vecs_f = []
            for a in forg:
                v = embed_from_file(a["path"], already_processed=True)
                vecs_f.append(v)
            vecs_f = np.stack(vecs_f, axis=0)
            for i in range(len(vecs_g)):
                for j in range(len(vecs_f)):
                    g_f.append(cos_dist(vecs_g[i], vecs_f[j]))

    if not g_g or not g_f:
        print("Yeterli veri bulunamadı (genuine/forgery dağılımları çıkarılamadı).")
        return

    g_g = np.array(g_g); g_f = np.array(g_f)
    print(f"[INFO] genuine-genuine örnek sayısı: {len(g_g)}, ort={g_g.mean():.3f} medyan={np.median(g_g):.3f}")
    print(f"[INFO] genuine-forgery örnek sayısı: {len(g_f)}, ort={g_f.mean():.3f} medyan={np.median(g_f):.3f}")

    # Basit EER benzeri eşik: yanlış pozitif ve yanlış negatif oranlarını dengelemeye çalış
    # Aday eşikler: birleşik dağılımdan yüzde adımları
    candidates = np.quantile(np.concatenate([g_g, g_f]), np.linspace(0.01, 0.99, 99))
    best_thr, best_gap = None, 1e9
    for t in candidates:
        # genuine-genuine için yanlış red: dist > t
        fnr = (g_g > t).mean()
        # genuine-forgery için yanlış kabul: dist <= t
        fpr = (g_f <= t).mean()
        gap = abs(fnr - fpr)
        if gap < best_gap:
            best_gap, best_thr = gap, t

    print(f"[RECOMMENDED] THRESHOLD ≈ {best_thr:.3f}  (FNR≈FPR dengesi)")
    # Ayrıca daha güvenli bir eşik de gösterebiliriz (genuine medyan + 1 std, vs.)
    print(f"[ALT] Güvenli tarafta bir eşik için öneri: max({np.median(g_g):.3f} + 1σ={g_g.std():.3f}) ≈ {(np.median(g_g)+g_g.std()):.3f}")

if __name__ == "__main__":
    main()

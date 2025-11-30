# scripts/query_siamese.py
import os, argparse, csv, joblib, numpy as np, cv2, torch
from sklearn.metrics.pairwise import cosine_distances
from train_siamese import EmbeddingNet, EMB_DIM, DEVICE

INDEX_FILE  = "models/nn_index_siamese.joblib"
META_CSV    = "data/embeddings/meta_siamese.csv"
MODEL_PATH  = "models/siamese_resnet18.pt"
INPUT_SIZE  = 128
THRESHOLD   = 0.40  # dist < 0.40 → aynı kişi

def preprocess_query(img, size=128):
    # 1) kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(img)
    # 2) binarize + invert (imza beyaz, zemin siyah)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 3) en büyük bileşene kırp
    cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        th = th[y:y+h, x:x+w]
    # 4) kare pad + resize
    h, w = th.shape[:2]
    s = max(h, w)
    top = (s-h)//2; bottom = s-h-top
    left = (s-w)//2; right = s-w-left
    sq = cv2.copyMakeBorder(th, top,bottom,left,right, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(sq, (size,size), interpolation=cv2.INTER_AREA)
    return out

def read_query_tensor(path, size=128):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    proc = preprocess_query(img, size=size)
    ten = torch.from_numpy(proc).float().unsqueeze(0).unsqueeze(0)/255.0
    return ten.to(DEVICE)

def load_meta(meta_csv):
    with open(meta_csv, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--claimed", help="İddia edilen kişi id (ör: 999)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # 1) indeks + meta
    data = joblib.load(INDEX_FILE)
    meta = load_meta(META_CSV)
    X = np.load(data["embeddings"])

    # 2) model
    net = EmbeddingNet(EMB_DIM, use_pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    # 3) sorgu embedding
    with torch.no_grad():
        q = net(read_query_tensor(args.image_path, size=INPUT_SIZE)).cpu().numpy()

    # 4) benzerlik
    dists = cosine_distances(q, X)[0]
    order = np.argsort(dists)[:args.topk]

    print("\n🔍 En yakın eşleşmeler (indeks yalnızca genuine):")
    for i in order:
        print(f"→ {meta[i]['person_id']} | {meta[i]['label']} | {meta[i]['filepath']} | uzaklık={dists[i]:.3f}")

    best_i = order[0]
    best_dist = dists[best_i]
    best_pid = meta[best_i]['person_id']

    if args.claimed:
        print(f"\nİddia edilen: {args.claimed}")
        if (best_pid == args.claimed) and (best_dist < args.threshold):
            print(f"✅ Doğrulandı (distance={best_dist:.3f})")
        else:
            print(f"⚠️ Uyuşmuyor / taklit şüphesi (en yakın {best_pid}, dist={best_dist:.3f})")
    else:
        print(f"\nTahmin: kişi {best_pid}, distance={best_dist:.3f}")
        print("→", "Aynı kişi olabilir ✅" if best_dist < args.threshold else "Yeni kişi olabilir ⚠️")

if __name__ == "__main__":
    main()

# scripts/build_index_siamese.py
import os, csv, argparse, numpy as np, torch, joblib, cv2
from sklearn.neighbors import NearestNeighbors
from datasets import load_items
from train_siamese import EmbeddingNet, EMB_DIM, DEVICE

MODEL_PATH = "models/siamese_resnet18.pt"
EMBED_NPY = "data/embeddings/embeddings_siamese.npy"
META_CSV  = "data/embeddings/meta_siamese.csv"
INDEX_FILE = "models/nn_index_siamese.joblib"
INPUT_SIZE = 128  # eğitimde kullandığın boyut

def read_img_gray(path, size=128):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size,size))
    ten = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)/255.0
    return ten.to(DEVICE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-forgery", action="store_true",
                    help="Varsayılan: indekse sadece genuine konur. Bunu verirsen forgery de dahil edilir.")
    args = ap.parse_args()

    # 1) veri: processed yollar
    all_items = load_items()  # [{'path','pid','label'}]
    items = [it for it in all_items if (args.include_forgery or it["label"]=="genuine")]
    if not items:
        print("[ERR] Indekse konacak öğe bulunamadı.")
        return
    print(f"[INFO] genuine count={sum(it['label']=='genuine' for it in items)}, forgery count={sum(it['label']=='forgery' for it in items)}")

    # 2) model
    net = EmbeddingNet(EMB_DIM, use_pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    # 3) embedding çıkar
    vecs, meta = [], []
    with torch.no_grad():
        for it in items:
            z = net(read_img_gray(it["path"], size=INPUT_SIZE)).cpu().numpy()[0]
            vecs.append(z)
            meta.append([it["path"], it["label"], it["pid"]])

    X = np.stack(vecs, axis=0)
    os.makedirs(os.path.dirname(EMBED_NPY), exist_ok=True)
    np.save(EMBED_NPY, X)
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["filepath","label","person_id"]); w.writerows(meta)

    # 4) NN indeksi
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(X)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"nn": nn, "meta_csv": META_CSV, "embeddings": EMBED_NPY}, INDEX_FILE)
    print(f"[OK] embeddings: {X.shape}  → saved {INDEX_FILE}")

if __name__ == "__main__":
    main()

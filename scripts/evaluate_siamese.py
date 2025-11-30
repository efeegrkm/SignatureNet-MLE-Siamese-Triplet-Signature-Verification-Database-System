# scripts/evaluate_siamese.py
import os, random, argparse, torch
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from datasets import load_items
from train_siamese import EmbeddingNet, DEVICE, EMB_DIM

MODEL_PATH = "models/siamese_resnet18.pt"

def cos_dist(z1, z2):
    # cosine distance = 1 - cosine similarity
    return 1 - F.cosine_similarity(z1, z2).item()

def read_img_gray(path, size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size))
    ten = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    return ten.to(DEVICE)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--input-size", type=int, default=128)  # son eğitiminde 128 kullandın
    args = ap.parse_args()

    print("[INFO] Model yükleniyor...")
    net = EmbeddingNet(EMB_DIM, use_pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    items = load_items()  # [{'path': processed_path, 'pid': '001', 'label': 'genuine'|'forgery'}, ...]
    if not items:
        print("[ERR] data/processed altında öğe bulunamadı. preprocess ve yolları kontrol et.")
        return

    # Kişiye göre grupla
    by_pid = {}
    for it in items:
        by_pid.setdefault(it["pid"], {"g": [], "f": []})
        by_pid[it["pid"]][it["label"][0]].append(it["path"])  # 'g' veya 'f'

    # Pozitif (aynı kişi, 2 genuine) çiftleri
    positive = []
    for pid, grp in by_pid.items():
        g = grp["g"]
        if len(g) >= 2:
            # kişi başı daha fazla örnek üret, sonra keseriz
            k = min(len(g)//2, 20)
            for _ in range(k):
                a, b = random.sample(g, 2)
                positive.append((a, b, 1))

    if not positive:
        print("[ERR] Pozitif çift üretilemedi (>=2 genuine imzası olan kişi yok).")
        return

    # Negatif (farklı kişi) çiftleri
    pids = [p for p, grp in by_pid.items() if grp["g"] or grp["f"]]
    negative = []
    for _ in range(len(positive)):
        pid1, pid2 = random.sample(pids, 2)
        pool1 = by_pid[pid1]["g"] or by_pid[pid1]["f"]
        pool2 = by_pid[pid2]["g"] or by_pid[pid2]["f"]
        a = random.choice(pool1)
        b = random.choice(pool2)
        negative.append((a, b, 0))

    pairs = positive + negative
    random.shuffle(pairs)
    pairs = pairs[:args.max_pairs]

    # Tüm görüntüler için embedding önbelleği (tek geçiş)
    uniq_paths = sorted({p for (a,b,_) in pairs for p in (a,b)})
    cache = {}
    with torch.no_grad():
        for p in tqdm(uniq_paths, desc="Embedding cache"):
            i = read_img_gray(p, size=args.input_size)
            cache[p] = net(i)

    # Değerlendirme
    correct = 0
    with torch.no_grad():
        for a, b, label in tqdm(pairs, desc="Evaluating"):
            z1 = cache[a]; z2 = cache[b]
            d = cos_dist(z1, z2)
            pred_same = d < args.threshold
            if pred_same == bool(label):
                correct += 1

    acc = correct / len(pairs)
    print(f"[RESULT] N={len(pairs)}  threshold={args.threshold:.3f}  →  accuracy={acc*100:.2f}%")

if __name__ == "__main__":
    main()

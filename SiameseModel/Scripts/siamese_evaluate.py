"""
siamese_evaluate.py
===================

Evaluate a Siamese signature verification model on a validation or test set.

This script loads a trained Siamese model (weights only) and computes
Euclidean distances between a reference signature and other signatures
belonging to the same user as well as forgeries. The distances and labels
are used to either evaluate the model at a given threshold or search for
an optimal threshold that maximizes accuracy.

Usage::

    python siamese_evaluate.py --model ../models/signature_siamese_best.pth \
        --data ../sign_data/split/val

You can optionally specify ``--threshold`` to evaluate at a fixed threshold.
If omitted, the script will sweep over the observed distance range and
select the threshold yielding the highest accuracy.
"""

import argparse
import os
import json
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from model import SignatureNet
from preprocess import preprocess_image   # 🔥 Preprocess pipeline


# ---------------------------------
# Model yükleme
# ---------------------------------
def load_model(model_path: str) -> Tuple[SignatureNet, torch.device]:
    """Load the trained Siamese model from disk and set it to evaluation mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignatureNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# ---------------------------------
# Distance hesaplama
# ---------------------------------
def compute_distances(
    model: SignatureNet,
    device: torch.device,
    data_dir: str
) -> Tuple[List[float], List[int]]:
    """Compute distances and labels between reference and test signatures."""
    distances: List[float] = []
    labels: List[int] = []

    users = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.endswith('_forg')
    ]

    for user in users:
        real_dir = os.path.join(data_dir, user)
        forg_dir = os.path.join(data_dir, user + '_forg')

        # Gerçek ve sahte imzaları topla
        real_imgs = [
            os.path.join(real_dir, f) for f in os.listdir(real_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        forg_imgs: List[str] = []
        if os.path.exists(forg_dir):
            forg_imgs = [
                os.path.join(forg_dir, f) for f in os.listdir(forg_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]

        # En az 2 gerçek imza olmalı ki karşılaştıralım
        if len(real_imgs) < 2:
            continue

        # İlk gerçek imzayı referans al
        ref_path = real_imgs[0]
        with torch.no_grad():
            ref_tensor = preprocess_image(ref_path).unsqueeze(0).to(device)  # [1,1,400,400]
            ref_emb = model(ref_tensor)

        # Genuine karşılaştırmalar (real vs other real)
        for other_path in real_imgs[1:]:
            with torch.no_grad():
                img_tensor = preprocess_image(other_path).unsqueeze(0).to(device)
                emb = model(img_tensor)
                dist = F.pairwise_distance(ref_emb, emb).item()

            distances.append(dist)
            labels.append(1)  # genuine

        # Imposter karşılaştırmalar (real vs forgery)
        for forg_path in forg_imgs:
            with torch.no_grad():
                img_tensor = preprocess_image(forg_path).unsqueeze(0).to(device)
                emb = model(img_tensor)
                dist = F.pairwise_distance(ref_emb, emb).item()

            distances.append(dist)
            labels.append(0)  # forgery

    return distances, labels


# ---------------------------------
# En iyi threshold’u bul
# ---------------------------------
def find_best_threshold(distances: List[float], labels: List[int]) -> Tuple[float, float]:
    """Search for the threshold that maximizes accuracy."""
    if not distances:
        return 0.0, 0.0

    unique_dists = sorted(set(distances))
    candidates = []
    if unique_dists:
        candidates.append(unique_dists[0] - 1e-6)
        candidates.extend(unique_dists)
        candidates.append(unique_dists[-1] + 1e-6)
    else:
        candidates = [0.0]

    best_thr = 0.0
    best_acc = 0.0

    for thr in candidates:
        tp = tn = fp = fn = 0
        for d, lbl in zip(distances, labels):
            pred = 1 if d < thr else 0
            if pred == 1 and lbl == 1:
                tp += 1
            elif pred == 0 and lbl == 0:
                tn += 1
            elif pred == 1 and lbl == 0:
                fp += 1
            elif pred == 0 and lbl == 1:
                fn += 1

        total = len(labels)
        acc = (tp + tn) / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_thr, best_acc


# ---------------------------------
# Log + grafik kaydet
# ---------------------------------
def save_eval_logs(
    distances: List[float],
    labels: List[int],
    threshold: float,
    tp: int, tn: int, fp: int, fn: int,
    model_path: str,
    data_dir: str
) -> None:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "logs"))
    os.makedirs(LOG_DIR, exist_ok=True)

    split_name = os.path.basename(os.path.normpath(data_dir))  # val / test vs.
    json_path = os.path.join(LOG_DIR, f"siamese_eval_{split_name}.json")
    fig_path = os.path.join(LOG_DIR, f"siamese_dist_{split_name}.png")

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    log = {
        "model_path": os.path.abspath(model_path),
        "data_dir": os.path.abspath(data_dir),
        "split": split_name,
        "threshold": threshold,
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "num_distances": len(distances),
    }

    # Ham veriyi de ekleyelim (çok büyük değilse)
    log["distances"] = distances
    log["labels"] = labels

    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[INFO] Evaluation log saved to {json_path}")

    # Histogram plot (genuine vs forgery)
    try:
        d = np.array(distances)
        lbl = np.array(labels)

        genuine = d[lbl == 1]
        forgery = d[lbl == 0]

        plt.figure()
        plt.hist(genuine, bins=30, alpha=0.6, label="Genuine", density=True)
        plt.hist(forgery, bins=30, alpha=0.6, label="Forgery", density=True)
        plt.axvline(threshold, color="red", linestyle="--", label=f"thr={threshold:.3f}")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.title(f"Siamese distance distribution ({split_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"[INFO] Distance histogram saved to {fig_path}")
    except Exception as e:
        print(f"[WARN] Could not generate histogram plot: {e}")


# ---------------------------------
# Ana evaluate fonksiyonu
# ---------------------------------
def evaluate(
    model_path: str,
    data_dir: str,
    threshold: float = None
) -> None:
    """Evaluate the Siamese model on a dataset."""
    model, device = load_model(model_path)
    print(f"Loaded model from {model_path}")

    distances, labels = compute_distances(model, device, data_dir)
    if not distances:
        print("No comparisons were generated. Make sure your dataset contains at least two genuine signatures per user.")
        return

    # Threshold yoksa, en iyisini bul
    if threshold is None:
        thr, best_acc = find_best_threshold(distances, labels)
        print(f"Best threshold determined: {thr:.4f} with accuracy {best_acc:.4f}")
        threshold = thr

    # Seçilen threshold ile metrikleri hesapla
    tp = tn = fp = fn = 0
    for d, lbl in zip(distances, labels):
        pred = 1 if d < threshold else 0
        if pred == 1 and lbl == 1:
            tp += 1
        elif pred == 0 and lbl == 0:
            tn += 1
        elif pred == 1 and lbl == 0:
            fp += 1
        elif pred == 0 and lbl == 1:
            fn += 1

    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    print("Evaluation results:")
    print(f"  Total comparisons: {total}")
    print(f"  True Positives:    {tp}")
    print(f"  True Negatives:    {tn}")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    print(f"  Accuracy:          {accuracy:.4f}")

    # Log + grafik
    save_eval_logs(distances, labels, threshold, tp, tn, fp, fn, model_path, data_dir)


# ---------------------------------
# CLI
# ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a Siamese signature model.")
    parser.add_argument('--model', required=True, help='Path to the trained Siamese model weights.')
    parser.add_argument('--data', required=True, help='Path to the evaluation data directory (e.g. sign_data/split/val).')
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Fixed threshold to use for classification. If omitted, the script will determine the best threshold.'
    )
    args = parser.parse_args()
    evaluate(args.model, args.data, args.threshold)


if __name__ == '__main__':
    main()

import os
import sys
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


# Path ayarları
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(CURRENT_DIR.parent))

try:
    from model import SignatureNet
except ImportError:
    print("HATA: model.py bulunamadı.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_DIR = PROJECT_ROOT / "triplet_model"

VAL_CSV = DATA_ROOT_DIR / "data" / "val_pairs.csv"
TEST_CSV = DATA_ROOT_DIR / "data" / "test_pairs.csv"

MODEL_PATH = DATA_ROOT_DIR / "triplet_model" / "models" / "signature_cnn_augmented.pth"


# ======================================================
# TRANSFORM (ESKİ BAŞARILI MODEL İÇİN)
# ======================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 224)),       # sabit resize — eski başarılı pipeline
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


# ======================================================
# DATASET
# ======================================================
class SignaturePairCSVDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform is not None else get_transform()
        self.pairs = []

        if not os.path.exists(csv_path):
            print(f"⚠️ CSV Yok: {csv_path}")
            return

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                self.pairs.append(row)

        print(f"Dataset yüklendi: {len(self.pairs)} satır -> {Path(csv_path).name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = Image.open(self.root_dir / p1).convert("L")
        img2 = Image.open(self.root_dir / p2).convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, torch.tensor(int(label), dtype=torch.long)


# ======================================================
# MODEL YÜKLEME
# ======================================================
def load_model_instance(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)

    if not os.path.exists(model_path):
        print(f"❌ Model yok: {model_path}")
        return None, device

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"✅ Model yüklendi: {model_path}")
    return model, device


# ======================================================
# CONFUSION MATRIX
# ======================================================
def eval_with_confusion(model, dataloader, device, threshold, name="SET"):
    all_dists = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)

            dist = F.pairwise_distance(model(img1), model(img2))

            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    preds = (all_dists < threshold).astype(int)

    TP = np.sum((preds == 1) & (all_labels == 1))
    TN = np.sum((preds == 0) & (all_labels == 0))
    FP = np.sum((preds == 1) & (all_labels == 0))
    FN = np.sum((preds == 0) & (all_labels == 1))

    acc = (TP + TN) / (TP + TN + FP + FN) * 100

    print(f"\n--- {name} CONFUSION (Eşik = {threshold:.2f}) ---")
    print(f"TP : {TP}")
    print(f"TN : {TN}")
    print(f"FP : {FP}")
    print(f"FN : {FN}")
    print(f"Accuracy: %{acc:.2f}")
    print("----------------------------------------")

    return all_dists, all_labels, acc


# ======================================================
# DISTANCE–DISTANCE (SEPARATION) GRAFİĞİ
# ======================================================
def plot_distance_table(pos, neg, save_path, set_name="TEST"):
    plt.figure(figsize=(10, 5))

    # Görsel olarak daha düzgün olsun diye sırala
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    plt.plot(pos_sorted, label="Genuine (Positive)", linewidth=2)
    plt.plot(neg_sorted, label="Forgery (Negative)", linewidth=2)

    # Ortalama çizgileri
    plt.axhline(pos.mean(), linestyle="--", alpha=0.7,
                label=f"Genuine Mean = {pos.mean():.3f}")
    plt.axhline(neg.mean(), linestyle="--", alpha=0.7,
                label=f"Forgery Mean = {neg.mean():.3f}")

    # Margin anotasyonu
    margin = neg.mean() - pos.mean()
    plt.text(
        len(neg) * 0.65,
        (neg.mean() + pos.mean()) / 2,
        f"Margin = {margin:.3f}",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6)
    )

    plt.title(f"{set_name} Distance Separation Table", fontsize=14)
    plt.xlabel("Sample Index (sorted)")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📊 Distance–Distance Separation Chart kaydedildi:\n{save_path}")

def plot_pr_curve(all_dists, all_labels, save_path, title="Precision–Recall Curve (Triplet Model)"):
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    # Mesafe küçükse "pozitif" (aynı kişi) -> skor olarak -distance kullanıyoruz
    scores = -all_dists

    precision, recall, _ = precision_recall_curve(all_labels, scores, pos_label=1)
    ap = average_precision_score(all_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📈 PR grafiği kaydedildi:\n{save_path}")

# ======================================================
# DISTANCE STATISTICS
# ======================================================
def print_distance_stats(all_dists, all_labels, set_name="SET"):
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    pos = all_dists[all_labels == 1]
    neg = all_dists[all_labels == 0]

    print(f"\n===== {set_name} DISTANCE STATS =====")
    print(f"Pozitif (Aynı kişi): {len(pos)}")
    print(f"Negatif (Farklı kişi): {len(neg)}")

    print("\n--- Pozitif Mesafeler ---")
    print(f"Mean : {pos.mean():.4f}")
    print(f"Std  : {pos.std():.4f}")
    print(f"Min  : {pos.min():.4f}")
    print(f"Max  : {pos.max():.4f}")

    print("\n--- Negatif Mesafeler ---")
    print(f"Mean : {neg.mean():.4f}")
    print(f"Std  : {neg.std():.4f}")
    print(f"Min  : {neg.min():.4f}")
    print(f"Max  : {neg.max():.4f}")

    print("\n--- Ayrılabilirlik ---")
    print(f"Margin (neg_mean - pos_mean): {(neg.mean() - pos.mean()):.4f}")
    print("====================================\n")

    return pos, neg

def plot_distance_hist(all_dists, all_labels, threshold, save_path, title="Distance distribution"):
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    genuine = all_dists[all_labels == 1]
    forgery = all_dists[all_labels == 0]

    plt.figure(figsize=(6,4))
    plt.hist(genuine, bins=30, density=True, alpha=0.6, label="Genuine")
    plt.hist(forgery, bins=30, density=True, alpha=0.6, label="Forgery")

    # Eşik çizgisi
    plt.axvline(threshold, linestyle="--", linewidth=2,
                label=f"thr={threshold:.3f}")

    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc_curve(all_dists, all_labels, save_path, title="ROC Curve (Triplet Model)"):
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    # Bizde MESAFE küçükse "aynı kişi" demek.
    # ROC için "büyüdükçe daha pozitif" bir skor lazım.
    # Bu yüzden skoru = -distance alıyoruz.
    scores = -all_dists

    fpr, tpr, _ = roc_curve(all_labels, scores, pos_label=1)
    auc = roc_auc_score(all_labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"📈 ROC grafiği kaydedildi:\n{save_path}")

# ======================================================
# EN İYİ THRESHOLD HESAPLAMA
# ======================================================
def find_best_threshold(model, dataloader, device):
    all_dists = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            dist = F.pairwise_distance(model(img1), model(img2))

            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    thresholds = np.arange(0.05, 2.0, 0.01)
    best_acc = 0
    best_th = 0

    for th in thresholds:
        preds = (all_dists < th).astype(int)
        acc = np.mean(preds == all_labels)

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_acc * 100, best_th, all_dists, all_labels


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    model, device = load_model_instance(MODEL_PATH)
    if model is None:
        exit()

    # -------------------------
    # VALIDATION
    # -------------------------
    val_dataset = SignaturePairCSVDataset(VAL_CSV, root_dir=PROJECT_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\n--- VALIDATION ANALİZİ ---")
    best_acc, best_th, all_dist_val, all_label_val = find_best_threshold(model, val_loader, device)

    print(f"📌 En iyi eşik: {best_th:.2f}")
    print(f"📌 Max Val Accuracy: %{best_acc:.2f}")

    # Confusion + stats + grafik (VAL)
    all_dists, all_labels, _ = eval_with_confusion(model, val_loader, device, best_th, name="VAL")
    pos_val, neg_val = print_distance_stats(all_dists, all_labels, "VAL")
    val_fig_path = PROJECT_ROOT / "training_logs" / "dist_table_val.png"
    plot_distance_table(pos_val, neg_val, val_fig_path, set_name="VAL")

    # -------------------------
    # TEST
    # -------------------------
    test_dataset = SignaturePairCSVDataset(TEST_CSV, root_dir=PROJECT_ROOT)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("\n--- TEST ANALİZİ ---")
    all_dists, all_labels, _ = eval_with_confusion(model, test_loader, device, best_th, name="TEST")
    pos_test, neg_test = print_distance_stats(all_dists, all_labels, "TEST")
    hist_path = PROJECT_ROOT / "training_logs" / "distance_distribution_test.png"
    plot_distance_hist(all_dists, all_labels, best_th, hist_path,
                   title="TEST Distance distribution")
        # ROC curve (TEST)
    roc_path = PROJECT_ROOT / "training_logs" / "roc_curve_triplet.png"
    plot_roc_curve(all_dists, all_labels, roc_path,
                   title="ROC Curve (Triplet Model)")
    
    pr_path = PROJECT_ROOT / "training_logs" / "pr_curve_triplet.png"
    plot_pr_curve(all_dists, all_labels, pr_path,
              title="Precision–Recall Curve (Triplet Model)")


    test_fig_path = PROJECT_ROOT / "training_logs" / "dist_table_test.png"
    plot_distance_table(pos_test, neg_test, test_fig_path, set_name="TEST")

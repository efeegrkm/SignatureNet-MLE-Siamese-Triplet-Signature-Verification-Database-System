import os
import sys
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

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

def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def preprocess_pil(img):
    # Veri seti zaten taranmış ve temiz olduğu için 
    # ağır işlemleri kaldırdık, sadece basit invert yapıyoruz.
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)
    return img

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
                if len(row) < 3: continue
                self.pairs.append(row)
        print(f"Dataset yüklendi: {len(self.pairs)} satır -> {Path(csv_path).name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        full1 = self.root_dir / p1
        full2 = self.root_dir / p2

        try:
            img1 = Image.open(full1).convert("L")
            img2 = Image.open(full2).convert("L")
        except Exception:
            img1 = Image.new("L", (224, 128))
            img2 = Image.new("L", (224, 128))

        # Dataset resimleri genelde temizdir, çok bozmadan işleyelim
        # img1 = preprocess_pil(img1) 
        # img2 = preprocess_pil(img2)
        # NOT: Eğer skorlar yine kötü çıkarsa üstteki 2 satırı yorumdan çıkarıp dene.
        # Genelde dataset resimleri zaten işlenmiştir.

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(int(label), dtype=torch.long)

def load_model_instance(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureNet().to(device)
    if not os.path.exists(model_path):
        print(f"❌ Model yok: {model_path}")
        return None, device
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def eval_with_confusion(model, dataloader, device, threshold, name="SET"):
    model.eval()
    all_dists = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            dist = F.pairwise_distance(emb1, emb2)

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

    print(f"\n--- {name} CONFUSION (eşik = {threshold:.2f}) ---")
    print(f"TP (True Positive):  {TP}")
    print(f"TN (True Negative):  {TN}")
    print(f"FP (False Positive): {FP}")
    print(f"FN (False Negative): {FN}")
    print(f"Accuracy: %{acc:.2f}")
    print("----------------------------------------")

    return acc, TP, TN, FP, FN

def find_best_threshold(model, dataloader, device):
    model.eval()
    all_dists = []
    all_labels = []

    print("Mesafe hesaplanıyor...")
    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            dist = F.pairwise_distance(emb1, emb2)
            
            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)

    # 0.1'den 3.0'a kadar tüm eşikleri dene
    thresholds = np.arange(0.1, 3.0, 0.05)
    best_acc = 0
    best_thresh = 0

    for th in thresholds:
        # Mesafe < Threshold ise 1 (Aynı), değilse 0 (Farklı)
        preds = (all_dists < th).astype(int)
        acc = np.mean(preds == all_labels)
        
        if acc > best_acc:
            best_acc = acc
            best_thresh = th

    return best_acc * 100, best_thresh, np.mean(all_dists[all_labels==1]), np.mean(all_dists[all_labels==0])

if __name__ == "__main__":
    model, device = load_model_instance(MODEL_PATH)
    if model is None:
        exit()

    # VAL ÜZERİNDEN EN İYİ THRESHOLD
    val_dataset = SignaturePairCSVDataset(str(VAL_CSV), root_dir=PROJECT_ROOT)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\n--- VALIDATION ANALİZİ ---")
    best_acc, best_th, avg_pos, avg_neg = find_best_threshold(model, val_loader, device)
    
    print(f"✅ EN İYİ EŞİK DEĞERİ BULUNDU: {best_th:.2f}")
    print(f"📊 Maksimum Val Başarısı: %{best_acc:.2f}")
    print(f"   Ort. Pozitif (Aynı): {avg_pos:.4f}")
    print(f"   Ort. Negatif (Farklı): {avg_neg:.4f}")
    print("-" * 30)

    # Bu eşiğe göre VAL confusion matrix
    eval_with_confusion(model, val_loader, device, best_th, name="VALIDATION")

    # TEST SETİ
    test_dataset = SignaturePairCSVDataset(str(TEST_CSV), root_dir=PROJECT_ROOT)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\n--- TEST SONUÇLARI (Eşik: {best_th:.2f}) ---")
    eval_with_confusion(model, test_loader, device, best_th, name="TEST")

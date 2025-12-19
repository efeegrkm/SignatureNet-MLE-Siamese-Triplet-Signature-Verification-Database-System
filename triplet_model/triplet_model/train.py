import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json
import argparse
import yaml

from dataloader import get_train_loader
from model import SignatureNet

# train.py'nin bulunduğu klasör
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/triplet_base.yaml",
        help="YAML config dosyası"
    )
    return parser.parse_args()

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def train():
    # ---- CONFIG YÜKLE ----
    args = parse_args()

    # Göreli yol geldiyse train.py'nin yanına göre çöz
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(BASE_DIR, config_path)

    cfg = load_config(config_path)

    train_cfg = cfg["train"]
    data_cfg  = cfg["data"]
    log_cfg   = cfg["logging"]

    # --- AYARLAR (artık YAML'den) ---
    root_dir      = os.path.join(BASE_DIR, data_cfg["train_dir"])
    epochs        = train_cfg["epochs"]
    batch_size    = train_cfg["batch_size"]
    learning_rate = train_cfg["learning_rate"]
    margin        = train_cfg["margin"]

    model_save_path = os.path.join(BASE_DIR, log_cfg["model_path"])
    log_dir         = os.path.join(BASE_DIR, log_cfg["log_dir"])

    # GRAFİK VERİLERİ — RAPOR İÇİN
    train_losses = []
    pos_dists = []
    neg_dists = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim şu cihazda yapılacak: {device}")
    print(f"Train dir: {root_dir}")

    train_loader = get_train_loader(root_dir, batch_size)
    model = SignatureNet().to(device)

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Eğitim başlıyor...")

    for epoch in range(epochs):
        running_loss = 0.0
        running_pos_dist = 0.0
        running_neg_dist = 0.0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():
                pos_dist = F.pairwise_distance(emb_anchor, emb_positive).mean()
                neg_dist = F.pairwise_distance(emb_anchor, emb_negative).mean()
                running_pos_dist += pos_dist.item()
                running_neg_dist += neg_dist.item()

        avg_loss = running_loss / len(train_loader)
        avg_pos  = running_pos_dist / len(train_loader)
        avg_neg  = running_neg_dist / len(train_loader)

        train_losses.append(avg_loss)
        pos_dists.append(avg_pos)
        neg_dists.append(avg_neg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] -> "
                  f"Loss: {avg_loss:.4f} | PosDist: {avg_pos:.4f} | NegDist: {avg_neg:.4f}")

    # --- MODEL KAYDETME ---
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Eğitim bitti. Model kaydedildi: {model_save_path}")

    # --- METRİKLERİ JSON'E KAYDET ---
    os.makedirs(log_dir, exist_ok=True)

    metrics = {
        "epochs": epochs,
        "train_loss": train_losses,
        "pos_dist": pos_dists,
        "neg_dist": neg_dists
    }

    metrics_path = os.path.join(log_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrikler kaydedildi: {metrics_path}")

if __name__ == "__main__":
    train()

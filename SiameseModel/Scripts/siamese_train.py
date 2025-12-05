# siamese_train.py

import os
import json
import yaml  # YAML config için
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # grafik için

from model import SignatureNet
from siamese_dataset import SignatureSiameseDataset


# ----------------------------------
# Contrastive Loss
# ----------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, o1, o2, label):
        dist = F.pairwise_distance(o1, o2)
        loss = (1 - label) * dist.pow(2) + label * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


# ----------------------------------
# Config yükleme
# ----------------------------------
def load_train_config():
    """
    SiameseModel/config/siamese_train.yaml dosyasını okumaya çalışır.
    Bulamazsa varsayılan hyperparametrelerle devam eder.

    Önerilen örnek config (siamese_train.yaml):

        root_dir: "../sign_data/split/train"
        epochs: 150
        batch_size: 16
        lr: 0.0005
        pos_fraction: 0.5
        margin: 1.0
        rotation: 5.0
        translate: 0.03
        scale_min: 1.0
        scale_max: 1.0
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))          # Scripts
    config_dir = os.path.abspath(os.path.join(base_dir, "..", "config"))
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "siamese_train.yaml")

    # Varsayılanlar (mevcut kodun parametreleriyle uyumlu)
    cfg = {
        "root_dir": "../sign_data/split/train",
        "epochs": 150,
        "batch_size": 16,
        "lr": 5e-4,
        "pos_fraction": 0.5,
        "margin": 1.0,
        "rotation": 5.0,     # RandomRotation(degrees)
        "translate": 0.03,   # RandomAffine translate=(t, t)
        "scale_min": 1.0,    # RandomAffine scale=(scale_min, scale_max)
        "scale_max": 1.0,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # İster düz, ister "train" altında olsun:
            if isinstance(data, dict) and "train" in data and isinstance(data["train"], dict):
                data = data["train"]

            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = v

            print(f"[INFO] Loaded training config from {config_path}")
        except Exception as e:
            print(f"[WARN] Could not parse config file {config_path}: {e}")
            print("[WARN] Using default training hyperparameters.")
    else:
        print(f"[WARN] Config file not found at {config_path}, using default hyperparameters.")

    return cfg, config_path


# ----------------------------------
# TRAIN FUNCTION
# ----------------------------------
def train_siamese(
    root_dir: str = "../sign_data/split/train",
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 5e-4,
    pos_fraction: float = 0.5,
    margin: float = 1.0,
    rotation: float = 5.0,
    translate: float = 0.03,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    config_path: str | None = None,
    used_config: dict | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # -------------------------------------------------------
    # PATHS: models + logs
    # -------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))       # Scripts klasörü
    model_dir = os.path.abspath(os.path.join(base_dir, "..", "models"))
    log_dir = os.path.abspath(os.path.join(base_dir, "..", "logs"))
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_path = os.path.join(model_dir, "signature_siamese_best.pth")
    last_path = os.path.join(model_dir, "signature_siamese_last.pth")

    print(f"[INFO] Best model save path : {best_path}")
    print(f"[INFO] Last model save path : {last_path}")
    print(f"[INFO] Logs dir              : {log_dir}")
    if config_path is not None:
        print(f"[INFO] Training config      : {config_path}")

    # -------------------------------------------------------
    # Augmentation (preprocess dataset içinde)
    # -------------------------------------------------------
    augment = transforms.Compose([
        transforms.RandomRotation(rotation),
        transforms.RandomAffine(
            degrees=0,
            translate=(translate, translate),
            scale=(scale_min, scale_max),
        ),
        transforms.ColorJitter(
        brightness=0.2,   
        contrast=0.2       
        ),
    ])

    dataset = SignatureSiameseDataset(root_dir, augment=augment, pos_fraction=pos_fraction)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # -------------------------------------------------------
    # Model + Optimizer + Loss
    # -------------------------------------------------------
    model = SignatureNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)

    best_loss = float("inf")

    # Log için listeler
    epoch_losses = []
    epoch_pos_dists = []
    epoch_neg_dists = []

    # -------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # pos/neg distance istatistikleri
        pos_sum = 0.0
        neg_sum = 0.0
        pos_count = 0
        neg_count = 0

        for img1, img2, label in loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()

            o1 = model(img1)
            o2 = model(img2)
            loss = criterion(o1, o2, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # --- Distance istatistikleri (gradient dışında) ---
            with torch.no_grad():
                dist = F.pairwise_distance(o1, o2)  # [B]
                pos_mask = (label == 0)
                neg_mask = (label == 1)

                if pos_mask.any():
                    pos_sum += dist[pos_mask].sum().item()
                    pos_count += pos_mask.sum().item()

                if neg_mask.any():
                    neg_sum += dist[neg_mask].sum().item()
                    neg_count += neg_mask.sum().item()

        avg_loss = total_loss / len(loader)
        avg_pos = pos_sum / pos_count if pos_count > 0 else 0.0
        avg_neg = neg_sum / neg_count if neg_count > 0 else 0.0

        epoch_losses.append(avg_loss)
        epoch_pos_dists.append(avg_pos)
        epoch_neg_dists.append(avg_neg)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Loss {avg_loss:.4f} | PosDist {avg_pos:.4f} | NegDist {avg_neg:.4f}"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)
            print(f"🔥 Best model updated at epoch {epoch+1} - loss {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), last_path)
    print("Training finished.")
    print(f"Best model saved to : {best_path}")
    print(f"Last model saved to : {last_path}")

    # -------------------------------------------------------
    # METRİKLERİ KAYDET (ham veri)
    # -------------------------------------------------------
    metrics = {
        "epochs": list(range(1, epochs + 1)),
        "loss": epoch_losses,
        "pos_dist": epoch_pos_dists,
        "neg_dist": epoch_neg_dists,
        "margin": criterion.margin,
        "batch_size": batch_size,
        "lr": lr,
        "pos_fraction": pos_fraction,
        "root_dir": root_dir,
        "rotation": rotation,
        "translate": translate,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "config_path": config_path,
        "config": used_config or {},
    }

    metrics_path = os.path.join(log_dir, "siamese_train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Training metrics saved to {metrics_path}")

    # -------------------------------------------------------
    # GRAFİKLERİ KAYDET
    # -------------------------------------------------------
    try:
        # Loss
        plt.figure()
        plt.plot(metrics["epochs"], metrics["loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Siamese Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        loss_fig_path = os.path.join(log_dir, "siamese_train_loss.png")
        plt.savefig(loss_fig_path, dpi=200)
        plt.close()

        # Pos/Neg distance
        plt.figure()
        plt.plot(metrics["epochs"], metrics["pos_dist"], label="Positive Dist", marker="o", markersize=2)
        plt.plot(metrics["epochs"], metrics["neg_dist"], label="Negative Dist", marker="o", markersize=2)
        plt.xlabel("Epoch")
        plt.ylabel("Distance")
        plt.title("Average Positive / Negative Distances per Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        dist_fig_path = os.path.join(log_dir, "siamese_train_pos_neg_dist.png")
        plt.savefig(dist_fig_path, dpi=200)
        plt.close()

        print(f"[INFO] Training plots saved to:")
        print(f"       {loss_fig_path}")
        print(f"       {dist_fig_path}")
    except Exception as e:
        print(f"[WARN] Could not generate training plots: {e}")


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":
    # YAML config'i yükle
    cfg, cfg_path = load_train_config()

    train_siamese(
        root_dir=cfg["root_dir"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        pos_fraction=cfg["pos_fraction"],
        margin=cfg["margin"],
        rotation=cfg["rotation"],
        translate=cfg["translate"],
        scale_min=cfg["scale_min"],
        scale_max=cfg["scale_max"],
        config_path=cfg_path,
        used_config=cfg,
    )

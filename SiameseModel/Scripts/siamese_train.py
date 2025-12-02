# siamese_train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

from model import SignatureNet
from siamese_dataset import SignatureSiameseDataset


# ----------------------------------
# Contrastive Loss
# ----------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, o1, o2, label):
        dist = F.pairwise_distance(o1, o2)
        loss = (1 - label) * dist.pow(2) + label * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


# ----------------------------------
# TRAIN FUNCTION
# ----------------------------------
def train_siamese(
    root_dir="../sign_data/split/train",
    epochs=150,
    batch_size=16,
    lr=5e-4,
    pos_fraction=0.5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # -------------------------------------------------------
    # FIXED MODEL SAVE PATHS — absolute, guaranteed valid
    # -------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Scripts klasörü
    MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_path = os.path.join(MODEL_DIR, "signature_siamese_best.pth")
    last_path = os.path.join(MODEL_DIR, "signature_siamese_last.pth")

    print(f"[INFO] Best model save path : {best_path}")
    print(f"[INFO] Last model save path : {last_path}")

    # -------------------------------------------------------
    # Augmentation (preprocess yok, dataset içinde)
    # -------------------------------------------------------
    augment = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(1, 1)),
    ])

    dataset = SignatureSiameseDataset(root_dir, augment=augment, pos_fraction=pos_fraction)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # -------------------------------------------------------
    # Model + Optimizer + Loss
    # -------------------------------------------------------
    model = SignatureNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=2.0)

    best_loss = float("inf")

    # -------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

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

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss {avg:.4f}")

        # Save best model
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), best_path)
            print(f"🔥 Best model updated at epoch {epoch+1} - loss {avg:.4f}")

    # Save final model
    torch.save(model.state_dict(), last_path)
    print("Training finished.")
    print(f"Best model saved to : {best_path}")
    print(f"Last model saved to : {last_path}")


# ----------------------------------
# MAIN
# ----------------------------------
if __name__ == "__main__":
    train_siamese()

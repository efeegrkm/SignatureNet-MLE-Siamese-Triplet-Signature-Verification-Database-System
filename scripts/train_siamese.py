# scripts/train_siamese.py
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from datasets import load_items, PairDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_DIM = 256
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
MODEL_PATH = "models/siamese_resnet18.pt"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--emb-dim", type=int, default=EMB_DIM)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--pairs-per-epoch", type=int, default=20000)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--run-name", default="siamese")           # TensorBoard klasörü
    ap.add_argument("--pretrained", action="store_true", help="ResNet18'i önceden eğitilmiş ağırlıklarla başlat (internet gerekir)")
    return ap.parse_args()

class EmbeddingNet(nn.Module):
    def __init__(self, emb_dim=256, use_pretrained=False):
        super().__init__()
        # torchvision sürümünden bağımsız şekilde pretrained kapat/aç
        if use_pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # 1-kanal girişe uyum
            w = base.conv1.weight.data.mean(dim=1, keepdim=True)  # [64,1,7,7]
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            base.conv1.weight.data = w
        else:
            base = models.resnet18(weights=None)  # eski sürümde pretrained=False eşdeğeri
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(base.conv1.weight, nonlinearity="relu")

        self.backbone = nn.Sequential(*(list(base.children())[:-1]))  # global avg pool'a kadar
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.backbone(x)          # [B,512,1,1]
        x = x.view(x.size(0), -1)     # [B,512]
        x = self.fc(x)                # [B,emb_dim]
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalize
        return x

class ContrastiveLoss(nn.Module):
    """Hadsell et al. contrastive loss.
       y=1 (aynı) -> d^2
       y=0 (farklı) -> max(0, margin - d)^2
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    def forward(self, z1, z2, y):
        d = torch.nn.functional.pairwise_distance(z1, z2)  # [B]
        y = y.squeeze()
        pos = y * (d**2)
        neg = (1 - y) * torch.clamp(self.margin - d, min=0.0)**2
        return (pos + neg).mean()

def train():
    args = parse_args()
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- Data
    items = load_items()
    ds = PairDataset(items, input_size=args.input_size, pairs_per_epoch=args.pairs_per_epoch)
    dl = DataLoader(ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,                         # 0 -> 2
        pin_memory=(DEVICE == "cuda"),         # GPU varsa daha hızlı aktarım
        drop_last=True)

    # ---- Model & Opt
    net = EmbeddingNet(args.emb_dim, use_pretrained=args.pretrained).to(DEVICE)
    opt = optim.AdamW(net.parameters(), lr=args.lr)
    crit = ContrastiveLoss(margin=args.margin)

    # ---- TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{args.run_name}")
    global_step = 0

    net.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{args.epochs}")
        for img1, img2, same in pbar:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            same = same.to(DEVICE)

            opt.zero_grad()
            z1 = net(img1)
            z2 = net(img2)
            loss = crit(z1, z2, same)
            loss.backward()
            opt.step()

            running += loss.item()
            writer.add_scalar("loss/step", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / len(dl)
        writer.add_scalar("loss/epoch", avg, epoch)
        print(f"[EPOCH {epoch}] loss={avg:.4f}")

        # Ara kayıt
        torch.save(net.state_dict(), MODEL_PATH)

    print(f"[SAVED] {MODEL_PATH}")
    writer.close()

if __name__ == "__main__":
    train()

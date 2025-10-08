# src/model_training/smoke_test.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import random
import numpy as np
import torch
from tqdm import tqdm
# Adjust imports to your project structure
from modal_training.Data.SignaturePairDataset import make_train_val_loaders   # path to your datasets helper
from modal_training.Modals.siamese_modal import EmbeddingNet, ContrastiveLoss, pairwise_distance, compute_auc_eer

# ----------------- hyperparams (smoke test) -----------------
DATA_ROOT = "data/final_data/train"   # senin data root
BATCH_SIZE = 8
PAIRS_PER_EPOCH = 2000
EPOCHS = 1
LR = 1e-4
MARGIN = 1.0
NUM_WORKERS = 0   # smoke test için 0 (Windows friendly)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_smoke"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# seeds (optional)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(seed)

# ----------------- loaders -----------------
train_loader, val_loader, train_ds, val_ds = make_train_val_loaders(
    final_train_root=DATA_ROOT,
    val_fraction=0.1,
    batch_size=BATCH_SIZE,
    pairs_per_epoch=PAIRS_PER_EPOCH,
    positive_prob=0.5,
    num_workers=NUM_WORKERS,
    seed=seed,
    pin_memory=True
)

print("[INFO] Dataset stats:", train_ds.stats(), val_ds.stats())

# quick batch sanity
imgs_a, imgs_b, labels = next(iter(train_loader))
print("Batch shapes:", imgs_a.shape, imgs_b.shape, labels.shape)
print("Labels sample:", labels[:20])

# ----------------- model, loss, opt -----------------
model = EmbeddingNet(out_dim=128, pretrained=True).to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

# ----------------- tiny training loop (smoke) -----------------
model.train()
running_losses = []
n_steps = 0
max_steps = 50  # en fazla 50 batch ile sınırla
pbar = tqdm(train_loader, desc="SmokeTrain")
for imgs_a, imgs_b, labels in pbar:
    imgs_a = imgs_a.to(DEVICE)
    imgs_b = imgs_b.to(DEVICE)
    labels = labels.to(DEVICE)

    opt.zero_grad()
    if scaler is not None:
        with torch.cuda.amp.autocast():
            emb_a = model(imgs_a)
            emb_b = model(imgs_b)
            loss = criterion(emb_a, emb_b, labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        emb_a = model(imgs_a)
        emb_b = model(imgs_b)
        loss = criterion(emb_a, emb_b, labels)
        loss.backward()
        opt.step()

    running_losses.append(loss.item())
    n_steps += 1
    pbar.set_postfix({"loss": np.mean(running_losses)})
    if n_steps >= max_steps:
        break

print("Average train loss (last steps):", np.mean(running_losses[-10:]))

# ----------------- quick validation -----------------
model.eval()
dists = []
ys = []
with torch.no_grad():
    for imgs_a, imgs_b, labels in tqdm(val_loader, desc="SmokeVal"):
        imgs_a = imgs_a.to(DEVICE)
        imgs_b = imgs_b.to(DEVICE)
        emb_a = model(imgs_a)
        emb_b = model(imgs_b)
        dist = pairwise_distance(emb_a, emb_b).cpu().numpy()
        dists.extend(dist.tolist())
        ys.extend(labels.numpy().tolist())
        if len(dists) >= 2000:  # küçük subset
            break

roc_auc, eer, thr = compute_auc_eer(np.array(ys), np.array(dists))
print(f"SMOKE VAL: ROC-AUC={roc_auc:.4f}, EER={eer:.4f}, thr={thr:.4f}")

# ----------------- save a quick checkpoint -----------------
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": opt.state_dict(),
    "roc_auc": roc_auc
}, os.path.join(CHECKPOINT_DIR, "smoke_checkpoint.pth"))

print("Smoke test done. Checkpoint saved.")

#!/usr/bin/env python3
"""
siamese_train.py

Train a Siamese network (ResNet18 -> 128-d embedding) with Contrastive Loss.

Usage:
    python src/modal_training/Train/siamese_train.py --config configs/main_train.yaml
"""
import os
import sys
import argparse
import yaml
import random
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- ensure project root / src in sys.path so relative modules resolve ---
# This file is expected at: <project_root>/src/modal_training/Train/siamese_train.py
# We want to add <project_root>/src to sys.path (two parents up).
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[2]  # .../src
_PROJECT_ROOT = _THIS_FILE.parents[3] if len(_THIS_FILE.parents) > 3 else _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Try to import dataset loader and model from possible locations (robust)
_imported = False
_import_errors = []
_try_paths = [
    ("modal_training.Data.SignaturePairDataset", "make_train_val_loaders"),
    ("model_training.Data.SignaturePairDataset", "make_train_val_loaders"),
    ("SignaturePairDataset", "make_train_val_loaders"),
    ("src.modal_training.Data.SignaturePairDataset", "make_train_val_loaders"),
]
for mod_path, symbol in _try_paths:
    try:
        mod = __import__(mod_path, fromlist=[symbol])
        make_train_val_loaders = getattr(mod, symbol)
        _imported = True
        break
    except Exception as e:
        _import_errors.append((mod_path, repr(e)))

if not _imported:
    raise ImportError(
        "Could not import make_train_val_loaders from your project. Tried:\n"
        + "\n".join(f"{m}: {err}" for m, err in _import_errors)
        + "\n\nMake sure SignaturePairDataset.py contains make_train_val_loaders and is importable."
    )

# Import Siamese model definitions (EmbeddingNet, ContrastiveLoss, helpers)
_imported = False
_import_errors = []
_try_model_paths = [
    ("modal_training.Modals.siamese_modal", ["EmbeddingNet", "ContrastiveLoss", "pairwise_distance", "compute_auc_eer"]),
    ("model_training.Modals.siamese_modal", ["EmbeddingNet", "ContrastiveLoss", "pairwise_distance", "compute_auc_eer"]),
    ("siamese_modal", ["EmbeddingNet", "ContrastiveLoss", "pairwise_distance", "compute_auc_eer"]),
    ("src.modal_training.Modals.siamese_modal", ["EmbeddingNet", "ContrastiveLoss", "pairwise_distance", "compute_auc_eer"]),
    ("src.model_training.Modals.siamese_modal", ["EmbeddingNet", "ContrastiveLoss", "pairwise_distance", "compute_auc_eer"]),
]
for mod_path, symbols in _try_model_paths:
    try:
        m = __import__(mod_path, fromlist=symbols)
        EmbeddingNet = getattr(m, "EmbeddingNet")
        ContrastiveLoss = getattr(m, "ContrastiveLoss")
        pairwise_distance = getattr(m, "pairwise_distance")
        compute_auc_eer = getattr(m, "compute_auc_eer")
        _imported = True
        break
    except Exception as e:
        _import_errors.append((mod_path, repr(e)))

if not _imported:
    raise ImportError(
        "Could not import Siamese model symbols. Tried:\n"
        + "\n".join(f"{m}: {err}" for m, err in _import_errors)
        + "\n\nMake sure siamese_modal.py defines EmbeddingNet, ContrastiveLoss, pairwise_distance, compute_auc_eer."
    )

# -------------------------
# Default configuration
# -------------------------
DEFAULT_CONFIG = {
    "data": {
        "final_train_root": "data/final_data/train",
        "val_fraction": 0.1,
        "pairs_per_epoch": 50000
    },
    "training": {
        "epochs": 40,
        "batch_size": 32,
        "num_workers": 4,
        "positive_prob": 0.5,
        "pin_memory": True,
        "seed": 42
    },
    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 1e-6
    },
    "loss": {
        "margin": 1.0
    },
    "checkpointing": {
        "outputs_dir": str(_PROJECT_ROOT / "outputs"),
        "save_every_n_epochs": 5,
        "max_keep": 5
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-7
    },
    "early_stopping": {
        "patience": 8
    },
    "eval": {
        "val_pairs_limit": 5000
    },
    "model": {
        "out_dim": 128,
        "pretrained": True
    }
}

# -------------------------
# Utility functions
# -------------------------
def load_config(path: str):
    cfg = DEFAULT_CONFIG.copy()
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    deep_update(d[k], v)
                else:
                    d[k] = v
        deep_update(cfg, user_cfg)
    else:
        print(f"[WARN] Config file not found at {path}. Using default config.")
    return cfg

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def plot_roc_tensorboard(y_true, distances):
    """
    distances: numpy array (higher -> more different)
    convert to scores = -distances then compute ROC curve and plot
    returns matplotlib Figure
    """
    from sklearn.metrics import roc_curve
    scores = -np.array(distances)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0,1],[0,1], '--', color='gray')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    return fig

# -------------------------
# Training function
# -------------------------
def train_main(config, resume_checkpoint=None):
    # prepare outputs directories
    outputs_dir = Path(config["checkpointing"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outputs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = outputs_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # save config used
    save_yaml(config, str(outputs_dir / "config_used.yaml"))

    # seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # create loaders
    data_cfg = config["data"]
    train_loader, val_loader, train_ds, val_ds = make_train_val_loaders(
        final_train_root=data_cfg["final_train_root"],
        val_fraction=data_cfg.get("val_fraction", 0.1),
        batch_size=config["training"]["batch_size"],
        pairs_per_epoch=data_cfg.get("pairs_per_epoch", 50000),
        positive_prob=config["training"].get("positive_prob", 0.5),
        num_workers=config["training"].get("num_workers", 4),
        seed=seed,
        pin_memory=config["training"].get("pin_memory", True),
    )

    print("[INFO] Dataset stats:", train_ds.stats(), val_ds.stats())

    # model, loss, optimizer
    model_cfg = config["model"]
    model = EmbeddingNet(out_dim=safe_int(model_cfg.get("out_dim", 128)), pretrained=bool(model_cfg.get("pretrained", True))).to(device)
    criterion = ContrastiveLoss(margin=safe_float(config["loss"].get("margin", 1.0)))
    lr = safe_float(config["optimizer"].get("lr", 1e-4))
    weight_decay = safe_float(config["optimizer"].get("weight_decay", 1e-6))
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # scheduler
    sch_cfg = config.get("scheduler", {})
    scheduler = None
    if sch_cfg.get("type", "ReduceLROnPlateau") == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                               factor=safe_float(sch_cfg.get("factor", 0.5)),
                                                               patience=safe_int(sch_cfg.get("patience", 3)),
                                                               min_lr=safe_float(sch_cfg.get("min_lr", 1e-7)))
    # AMP
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # logging
    tb = SummaryWriter(log_dir=str(tb_dir))
    log_file = outputs_dir / "train_log.txt"
    lf = open(log_file, "a", buffering=1)
    def log(s):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lf.write(f"[{ts}] {s}\n")
        print(s)

    # resume support
    start_epoch = 1
    best_auc = -1.0
    epochs = safe_int(config["training"].get("epochs", 40))
    if resume_checkpoint:
        ck = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            try:
                opt.load_state_dict(ck["optimizer_state"])
            except Exception:
                log("[WARN] Failed to load optimizer state (shape mismatch). Continuing without optimizer restore.")
        start_epoch = ck.get("epoch", 0) + 1
        best_auc = ck.get("best_auc", best_auc)
        log(f"[INFO] Resumed from {resume_checkpoint}, start_epoch={start_epoch}, best_auc={best_auc}")

    # training loop
    patience = safe_int(config.get("early_stopping", {}).get("patience", 8))
    no_improve_epochs = 0
    val_pairs_limit = safe_int(config.get("eval", {}).get("val_pairs_limit", 5000))

    total_steps = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_iter = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train")
        for imgs_a, imgs_b, labels in pbar:
            imgs_a = imgs_a.to(device)
            imgs_b = imgs_b.to(device)
            labels = labels.to(device)

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

            running_loss += float(loss.item())
            n_iter += 1
            total_steps += 1
            pbar.set_postfix({"loss": running_loss / n_iter, "lr": opt.param_groups[0]["lr"]})

            # log to tensorboard every 100 steps
            if total_steps % 100 == 0:
                tb.add_scalar("train/loss", running_loss / n_iter, total_steps)

        avg_train_loss = running_loss / max(1, n_iter)
        log(f"[Epoch {epoch}] train_loss={avg_train_loss:.6f}")
        tb.add_scalar("train/epoch_loss", avg_train_loss, epoch)

        # --- validation ---
        model.eval()
        dists = []
        ys = []
        with torch.no_grad():
            for imgs_a, imgs_b, labels in tqdm(val_loader, desc="Validation"):
                imgs_a = imgs_a.to(device)
                imgs_b = imgs_b.to(device)
                emb_a = model(imgs_a)
                emb_b = model(imgs_b)
                dist = pairwise_distance(emb_a, emb_b).cpu().numpy()
                dists.extend(dist.tolist())
                ys.extend(labels.cpu().numpy().tolist())
                if len(dists) >= val_pairs_limit:
                    break

        roc_auc, eer, thr = compute_auc_eer(np.array(ys), np.array(dists))
        log(f"[Epoch {epoch}] val ROC-AUC={roc_auc:.6f} EER={eer:.6f} thr={thr:.6f}")
        tb.add_scalar("val/roc_auc", float(roc_auc), epoch)
        tb.add_scalar("val/eer", float(eer), epoch)

        # add ROC curve figure to tensorboard
        try:
            fig = plot_roc_tensorboard(np.array(ys), np.array(dists))
            tb.add_figure("val/roc_curve", fig, epoch)
            plt.close(fig)
        except Exception as e:
            log(f"[WARN] Could not plot ROC curve for tensorboard: {e}")

        # scheduler step (if ReduceLROnPlateau)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(roc_auc)
        elif scheduler is not None:
            scheduler.step()

        # checkpointing
        if roc_auc > best_auc:
            best_auc = roc_auc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "best_auc": best_auc,
                "roc_auc": roc_auc
            }
            torch.save(ckpt, ckpt_dir / f"best_model_epoch{epoch}.pth")
            log(f"[INFO] Saved new best model at epoch {epoch}, AUC={roc_auc:.6f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # periodic save
        if epoch % safe_int(config["checkpointing"].get("save_every_n_epochs", 5)) == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "roc_auc": roc_auc
            }, ckpt_dir / f"snapshot_epoch{epoch}.pth")

        # early stopping
        if no_improve_epochs >= patience:
            log(f"[INFO] Early stopping triggered (no improvement in {patience} epochs).")
            break

    # final save
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "best_auc": best_auc
    }, ckpt_dir / f"final_epoch{epoch}.pth")
    log("[INFO] Training finished.")
    lf.close()
    tb.close()

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/main_train.yaml", help="Path to YAML config")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    # ensure absolute outputs dir
    cfg["checkpointing"]["outputs_dir"] = os.path.abspath(cfg["checkpointing"]["outputs_dir"])
    print("[INFO] Effective config:\n", pformat(cfg))
    train_main(cfg, resume_checkpoint=args.resume)

if __name__ == "__main__":
    main()

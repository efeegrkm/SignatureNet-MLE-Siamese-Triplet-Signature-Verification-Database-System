import os
import random
from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class SignaturePairDataset(Dataset):
    """
    Dataset that returns pairs (img_a, img_b, label)
    - label = 1 -> positive (same writer, both genuine)
    - label = 0 -> negative (genuine vs forgery OR genuine vs genuine-of-different-writer)

    Expected folder structure:
      root/genuine/{writer_id}/*.png
      root/forgery/{writer_id}/*.png   (optional)

    Parameters:
    - root_dir: Path-like to the folder containing 'genuine' and optionally 'forgery'
    - writers: optional list of writer ids to restrict dataset (useful for train/val split)
    - transform: torchvision transform applied to PIL images
    - pairs_per_epoch: __len__ returns this (controls epoch size)
    - positive_prob: probability that a sampled pair is positive (0..1)
    - prefer_forgery_neg: if True and forgery pool exists, half of negatives will come from forgery pool
    """
    def __init__(self,
                 root_dir,
                 writers: Optional[List[str]] = None,
                 transform=None,
                 pairs_per_epoch: int = 50000,
                 positive_prob: float = 0.5,
                 prefer_forgery_neg: bool = True,
                 seed: int = 42):
        self.root = Path(root_dir)
        self.genuine_dir = self.root / "genuine"
        self.forgery_dir = self.root / "forgery"
        self.transform = transform or T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.positive_prob = float(positive_prob)
        self.prefer_forgery_neg = prefer_forgery_neg

        random.seed(seed)

        # Build writer -> list of genuine image paths
        self.writer2imgs = {}
        if not self.genuine_dir.exists():
            raise FileNotFoundError(f"Expected genuine dir at {self.genuine_dir}")

        for w in sorted(os.listdir(self.genuine_dir)):
            wdir = self.genuine_dir / w
            if not wdir.is_dir():
                continue
            imgs = [str(wdir / f) for f in os.listdir(wdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
            if len(imgs) > 0:
                self.writer2imgs[w] = imgs

        # Optionally filter writers
        if writers is not None:
            writers = [str(x) for x in writers]
            self.writer2imgs = {w: self.writer2imgs[w] for w in writers if w in self.writer2imgs}

        self.writers = list(self.writer2imgs.keys())
        if len(self.writers) < 2:
            raise RuntimeError("Need at least 2 writers with genuine images for negative sampling.")

        # Flatten forgery pool if exists
        self.forgery_pool = []
        if self.forgery_dir.exists():
            for w in sorted(os.listdir(self.forgery_dir)):
                wdir = self.forgery_dir / w
                if not wdir.is_dir():
                    continue
                for f in os.listdir(wdir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                        self.forgery_pool.append(str(wdir / f))

    def __len__(self):
        return self.pairs_per_epoch

    def _load_img(self, pth):
        # load grayscale image as PIL and return single-channel PIL image
        img = Image.open(pth).convert('L')
        return img

    def __getitem__(self, idx):
        # Balanced sampling by probability
        is_positive = random.random() < self.positive_prob

        if is_positive:
            # positive: sample a writer, then two different genuine images if possible
            writer = random.choice(self.writers)
            imgs = self.writer2imgs[writer]
            if len(imgs) >= 2:
                a, b = random.sample(imgs, 2)
            else:
                # fallback: same image twice (rare if dataset small)
                a = b = imgs[0]
            label = 1.0
        else:
            # negative: either use forgery pool (if prefer and available) or sample genuine from different writers
            use_forgery = self.prefer_forgery_neg and (len(self.forgery_pool) > 0) and (random.random() < 0.5)
            if use_forgery:
                writer = random.choice(self.writers)
                a = random.choice(self.writer2imgs[writer])
                b = random.choice(self.forgery_pool)
            else:
                w1, w2 = random.sample(self.writers, 2)
                a = random.choice(self.writer2imgs[w1])
                b = random.choice(self.writer2imgs[w2])
            label = 0.0

        img_a = self._load_img(a)
        img_b = self._load_img(b)

        if self.transform is not None:
            img_a = self.transform(img_a)  # Tensor CxHxW
            img_b = self.transform(img_b)

        return img_a, img_b, torch.tensor(label, dtype=torch.float32)

    # helper utilities
    def get_writer_list(self):
        return list(self.writers)

    def stats(self):
        n_writers = len(self.writers)
        avg_per_writer = sum(len(v) for v in self.writer2imgs.values()) / max(1, n_writers)
        n_forgery = len(self.forgery_pool)
        return {"n_writers": n_writers, "avg_imgs_per_writer": avg_per_writer, "n_forgery": n_forgery}


# ------------------------------
# Helper: create writer-disjoint train/val split and dataloaders
# ------------------------------
def make_train_val_loaders(final_train_root: str,
                           val_fraction: float = 0.1,
                           batch_size: int = 32,
                           pairs_per_epoch: int = 50000,
                           positive_prob: float = 0.5,
                           num_workers: int = 4,
                           seed: int = 42,
                           pin_memory: bool = True):
    """
    Returns: train_loader, val_loader, train_dataset, val_dataset
    - final_train_root: path to final_train_data directory (contains genuine/ and forgery/)
    - This will split writers in genuine/ into train and val sets (writer-disjoint).
    """
    final_train_root = Path(final_train_root)
    genuine_dir = final_train_root / "genuine"
    writers = sorted([d for d in os.listdir(genuine_dir) if (genuine_dir / d).is_dir()])
    random.Random(seed).shuffle(writers)

    n_val = max(1, int(len(writers) * val_fraction))
    val_writers = writers[:n_val]
    train_writers = writers[n_val:]

    print(f"[INFO] total_writers={len(writers)} train={len(train_writers)} val={len(val_writers)}")

    transform = T.Compose([
        T.ToTensor(),            # 0..1
        T.Normalize([0.5], [0.5])  # -1..1
    ])

    train_ds = SignaturePairDataset(final_train_root, writers=train_writers, transform=transform,
                                    pairs_per_epoch=pairs_per_epoch, positive_prob=positive_prob, seed=seed)
    val_ds = SignaturePairDataset(final_train_root, writers=val_writers, transform=transform,
                                  pairs_per_epoch=max(2000, int(pairs_per_epoch*0.02)), positive_prob=0.5, seed=seed+1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers>0))

    return train_loader, val_loader, train_ds, val_ds

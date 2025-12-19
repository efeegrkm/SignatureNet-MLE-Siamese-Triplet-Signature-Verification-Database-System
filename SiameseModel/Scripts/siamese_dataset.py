# siamese_dataset.py

import os
import random
from typing import Dict, List, Tuple

from torch.utils.data import Dataset

from preprocess import preprocess_image  


class SignatureSiameseDataset(Dataset):
    def __init__(self, root_dir: str, augment=None, pos_fraction: float = 0.5):
        self.root_dir = root_dir
        self.augment = augment
        self.pos_fraction = pos_fraction

        self.user_real_images: Dict[str, List[str]] = {}
        self.user_forg_images: Dict[str, List[str]] = {}

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')

        for entry in os.listdir(self.root_dir):
            entry_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            if entry.endswith("_forg"):
                user = entry[:-5]
                imgs = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if f.lower().endswith(valid_exts)
                ]
                if imgs:
                    self.user_forg_images[user] = imgs
            else:
                user = entry
                imgs = [
                    os.path.join(entry_path, f)
                    for f in os.listdir(entry_path)
                    if f.lower().endswith(valid_exts)
                ]
                if imgs:
                    self.user_real_images[user] = imgs

        self.users_with_multiple_real = [
            u for u, imgs in self.user_real_images.items() if len(imgs) >= 2
        ]

        self._length = sum(len(v) for v in self.user_real_images.values())

    def __len__(self):
        return self._length

    def _get_positive(self):
        user = random.choice(self.users_with_multiple_real)
        img1, img2 = random.sample(self.user_real_images[user], 2)
        label = 0
        return img1, img2, label

    def _get_negative(self):
        # Forgery varsa %50 ihtimalle real–forge
        users_with_forg = [u for u in self.user_forg_images if len(self.user_forg_images[u])]

        if users_with_forg and random.random() < 0.5:
            user = random.choice(users_with_forg)
            img1 = random.choice(self.user_real_images[user])
            img2 = random.choice(self.user_forg_images[user])
            return img1, img2, 1

        # Cross-user
        u1, u2 = random.sample(list(self.user_real_images.keys()), 2)
        img1 = random.choice(self.user_real_images[u1])
        img2 = random.choice(self.user_real_images[u2])
        return img1, img2, 1

    def __getitem__(self, idx):
        if random.random() < self.pos_fraction:
            p1, p2, label = self._get_positive()
        else:
            p1, p2, label = self._get_negative()

        # --- PREPROCESSING BURADA YAPILIYOR ARTIK ---
        img1 = preprocess_image(p1)
        img2 = preprocess_image(p2)

        # Opsiyonel augment
        if self.augment:
            img1 = self.augment(img1)
            img2 = self.augment(img2)

        return img1, img2, label

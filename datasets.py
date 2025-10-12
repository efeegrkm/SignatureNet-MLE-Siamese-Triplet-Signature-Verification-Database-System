# scripts/datasets.py
import csv, os, cv2, random
import numpy as np
from torch.utils.data import Dataset
import torch

DATA_CSV = "data/labels.csv"
PROC_ROOT = "data/processed"

def _proc_path(raw_path):
    rel = os.path.relpath(raw_path, start="data").replace("\\","/")
    return os.path.join("data/processed", rel).replace("\\","/")

def load_items():
    items = []
    with open(DATA_CSV, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p = _proc_path(r["filepath"])
            if os.path.exists(p):
                items.append({"path": p, "pid": r["person_id"], "label": r["label"]})
    return items

class PairDataset(Dataset):
    """
    Contrastive loss için çift (img1, img2, same_flag) üretir.
    Pozitif: aynı kişi (genuine-genuine)
    Negatif: farklı kişi (genuine-genuine) veya (genuine-forgery aynı pid) -> negative sayılır
    """
    def __init__(self, items, input_size=224, pairs_per_epoch=20000):
        self.items = items
        self.size = input_size
        # kişiye göre grupla
        self.by_pid = {}
        for it in items:
            self.by_pid.setdefault(it["pid"], {"g":[], "f":[]})
            if it["label"]=="genuine":
                self.by_pid[it["pid"]]["g"].append(it["path"])
            else:
                self.by_pid[it["pid"]]["f"].append(it["path"])
        self.pids = [p for p in self.by_pid.keys() if len(self.by_pid[p]["g"])>=2]
        self.pairs_per_epoch = pairs_per_epoch
        # negatifte kullanılacak “farklı pid” listesi
        self.pid_list = list(self.by_pid.keys())

    def __len__(self):
        return self.pairs_per_epoch

    def _read(self, p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:  # güvenlik
            raise FileNotFoundError(p)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # tensör: [1,H,W], [0..1]
        t = torch.from_numpy(img).float().unsqueeze(0)/255.0
        return t

    def __getitem__(self, idx):
        # %50 pozitif, %50 negatif
        if random.random() < 0.5:
            # pozitif: aynı pid, iki farklı genuine
            pid = random.choice(self.pids)
            g = self.by_pid[pid]["g"]
            a, b = random.sample(g, 2)
            return self._read(a), self._read(b), torch.tensor([1.0], dtype=torch.float32)
        else:
            # negatif: ya farklı pid (genuine-genuine) ya da aynı pid (genuine-forgery)
            if random.random() < 0.5:
                # farklı pid
                pid_a, pid_b = random.sample(self.pid_list, 2)
                a = random.choice(self.by_pid[pid_a]["g"]) if self.by_pid[pid_a]["g"] else None
                b = random.choice(self.by_pid[pid_b]["g"]) if self.by_pid[pid_b]["g"] else None
                # fallback
                if a is None: a = random.choice(self.by_pid[pid_a]["f"]) if self.by_pid[pid_a]["f"] else random.choice(self.by_pid[pid_a]["g"])
                if b is None: b = random.choice(self.by_pid[pid_b]["f"]) if self.by_pid[pid_b]["f"] else random.choice(self.by_pid[pid_b]["g"])
            else:
                # aynı pid: genuine vs forgery (negatif)
                pid = random.choice(self.pid_list)
                g_list = self.by_pid[pid]["g"]
                f_list = self.by_pid[pid]["f"]
                if g_list and f_list:
                    a = random.choice(g_list)
                    b = random.choice(f_list)
                else:
                    # fallback farklı pid
                    pid_a, pid_b = random.sample(self.pid_list, 2)
                    a = random.choice(self.by_pid[pid_a]["g"] or self.by_pid[pid_a]["f"])
                    b = random.choice(self.by_pid[pid_b]["g"] or self.by_pid[pid_b]["f"])
            return self._read(a), self._read(b), torch.tensor([0.0], dtype=torch.float32)

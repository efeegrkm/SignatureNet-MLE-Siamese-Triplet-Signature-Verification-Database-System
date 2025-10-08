import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -------------------------
# Embedding network (ResNet18 -> 128-d)
# -------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)

        # adapt first conv to single channel (grayscale)
        # original conv1 weight shape: (64,3,7,7)
        w = resnet.conv1.weight.data  # (64,3,7,7)
        new_w = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight.data = new_w

        # remove classifier head
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.fc = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        """
        x: (B,1,H,W) float tensor
        returns: L2-normalized embedding (B, out_dim)
        """
        feat = self.backbone(x)       # (B, num_ftrs)
        emb = self.fc(feat)           # (B, out_dim)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

# -------------------------
# Contrastive Loss
# -------------------------
class ContrastiveLoss(nn.Module):
    """
    label: 1 -> positive (same), 0 -> negative (different)
    Loss = 0.5 * [ label * d^2  +  (1-label) * max(0, margin - d)^2 ]
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        # emb1, emb2: (B, D), label: (B,) float (1 or 0)
        dist = F.pairwise_distance(emb1, emb2, p=2)  # (B,)
        loss_pos = label * torch.pow(dist, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = 0.5 * (loss_pos + loss_neg)
        return loss.mean()

# -------------------------
# Helpers: distance, EER / AUC computation
# -------------------------
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def pairwise_distance(emb1, emb2):
    """Return Euclidean distances (B,)"""
    return F.pairwise_distance(emb1, emb2, p=2)

def compute_auc_eer(y_true, distances):
    """
    y_true: array-like (1=positive same, 0=negative)
    distances: array-like (higher -> more different)
    Returns: roc_auc, eer, eer_threshold
    """
    # for ROC AUC we want similarity or score where higher => positive,
    # but distances are higher => negative, so invert distances
    scores = -np.array(distances)
    roc_auc = roc_auc_score(y_true, scores)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # EER: point where FPR ~= FNR
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]
    return roc_auc, eer, eer_threshold

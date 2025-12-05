import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def load_eval_json(path):
    """Load distances & labels from JSON saved by evaluation step."""
    with open(path, "r") as f:
        data = json.load(f)

    distances = np.array(data["distances"])
    labels = np.array(data["labels"])

    return distances, labels


def plot_roc(distances, labels, save_path):
    """ROC curve + AUC"""
    # model predicts "genuine" for smaller distances → invert sign
    scores = -distances

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Siamese Model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] ROC curve saved to {save_path}")
    return roc_auc


def plot_pr(distances, labels, save_path):
    """Precision–Recall curve + PR-AUC"""
    scores = -distances  # same scoring rule

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Siamese Model)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[INFO] PR curve saved to {save_path}")
    return pr_auc


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.abspath(os.path.join(BASE_DIR, "../..", "logs"))

    eval_json = os.path.join(LOG_DIR, "siamese_eval_test.json")

    print(f"[INFO] Loading evaluation JSON: {eval_json}")
    distances, labels = load_eval_json(eval_json)

    roc_path = os.path.join(LOG_DIR, "siamese_roc_curve.png")
    pr_path = os.path.join(LOG_DIR, "siamese_pr_curve.png")

    roc_auc = plot_roc(distances, labels, roc_path)
    pr_auc = plot_pr(distances, labels, pr_path)

    print(f"[RESULT] ROC-AUC: {roc_auc:.4f}")
    print(f"[RESULT] PR-AUC:  {pr_auc:.4f}")


if __name__ == "__main__":
    main()

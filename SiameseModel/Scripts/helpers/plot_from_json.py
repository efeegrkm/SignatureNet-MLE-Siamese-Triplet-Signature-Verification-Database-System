import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "../..", "logs")

VAL_JSON = os.path.join(LOG_DIR, "siamese_eval_val.json")
TEST_JSON = os.path.join(LOG_DIR, "siamese_eval_test.json")

OUTPUT_ROC_VAL = os.path.join(LOG_DIR, "roc_curve_val.png")
OUTPUT_ROC_TEST = os.path.join(LOG_DIR, "roc_curve_test.png")


def load_eval_json(path):
    """Load distances + labels from evaluation JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    distances = np.array(data["distances"])
    labels = np.array(data["labels"])
    return distances, labels


def plot_roc(distances, labels, save_path, title="ROC Curve"):
    """
    Draw ROC curve using distances vs labels.
    In Siamese networks:
        - label = 0 → SAME person
        - label = 1 → DIFFERENT person
      So distances should be considered as "score" for ROC.
    """

    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)

    print(f"[INFO] AUC for {os.path.basename(save_path)} = {roc_auc:.4f}")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("[PLOT SAVED]", save_path)


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Loading JSON eval results...")

    # ---- Validation ROC ----
    if os.path.exists(VAL_JSON):
        val_dist, val_labels = load_eval_json(VAL_JSON)
        plot_roc(val_dist, val_labels, OUTPUT_ROC_VAL, "ROC Curve (Validation Set)")
    else:
        print("[WARN] Validation JSON not found:", VAL_JSON)

    # ---- Test ROC ----
    if os.path.exists(TEST_JSON):
        test_dist, test_labels = load_eval_json(TEST_JSON)
        plot_roc(test_dist, test_labels, OUTPUT_ROC_TEST, "ROC Curve (Test Set)")
    else:
        print("[WARN] Test JSON not found:", TEST_JSON)

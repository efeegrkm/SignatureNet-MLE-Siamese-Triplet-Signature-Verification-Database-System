import json
import matplotlib.pyplot as plt
import os

log_path = "training_logs/metrics.json"

with open(log_path, "r") as f:
    metrics = json.load(f)

epochs = range(1, metrics["epochs"] + 1)

train_loss = metrics["train_loss"]
pos_dist = metrics["pos_dist"]
neg_dist = metrics["neg_dist"]

# 1) Train Loss Curve
plt.figure()
plt.plot(epochs, train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.legend()
plt.savefig("training_logs/train_loss_curve.png")

# 2) Positive Distance Curve
plt.figure()
plt.plot(epochs, pos_dist, label="Positive Distance")
plt.xlabel("Epoch")
plt.ylabel("Distance")
plt.title("Positive Pair Distance Curve")
plt.grid(True)
plt.legend()
plt.savefig("training_logs/positive_dist_curve.png")

# 3) Negative Distance Curve
plt.figure()
plt.plot(epochs, neg_dist, label="Negative Distance")
plt.xlabel("Epoch")
plt.ylabel("Distance")
plt.title("Negative Pair Distance Curve")
plt.grid(True)
plt.legend()
plt.savefig("training_logs/negative_dist_curve.png")

print("Plotlar kaydedildi: training_logs/")

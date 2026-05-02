import pandas as pd
import matplotlib.pyplot as plt

csv_path = "runs/detect/train2/results.csv"  # change if needed
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

metrics = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
axes = axes.flatten()

for ax, metric in zip(axes, metrics):
    ax.plot(df["epoch"], df[metric])
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.grid(True)

plt.tight_layout()
plt.savefig("training_metrics_over_epochs.png", dpi=300)
plt.show()
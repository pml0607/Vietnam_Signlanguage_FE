import torch
from torch.utils.data import DataLoader
from dataset import PreprocessedClipDataset
from model import build_s3d_model
from train_utils import clip_collate_fn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Config
val_dir = "preprocessed_clips_6ch_v2/val"
batch_size = 4
num_classes = 15
model_path = "best_s3d_model_6ch_v2.pt"
cm_output_path = "confusion_matrix_v4.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
val_dataset = PreprocessedClipDataset(val_dir)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=clip_collate_fn)

# Load model
model = build_s3d_model(num_classes=num_classes, pretrained=False)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Evaluation
all_preds = []
all_labels = []
correct, total = 0, 0

with torch.no_grad():
    for clips, labels, _, _ in val_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"\n✅ Accuracy: {acc:.2%}")
print("📊 Classification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(cm_output_path)
print(f"📥 Confusion matrix saved to {cm_output_path}")

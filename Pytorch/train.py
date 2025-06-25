import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from dataset import PreprocessedClipDataset
from model import build_s3d_model
from train_utils import clip_collate_fn

# Config
train_dir = "preprocessed_clips/train"
val_dir = "preprocessed_clips/val"
batch_size = 4
epochs = 50
lr = 1e-4
log_dir = "runs/s3d_experiment_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets & Dataloaders
train_dataset = PreprocessedClipDataset(train_dir)
val_dataset = PreprocessedClipDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=clip_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=clip_collate_fn)

# Model
model = build_s3d_model(num_classes=15, pretrained=True, freeze_until_layer=12).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Training loop
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for clips, labels, _, _ in tqdm(loader, desc="Training", leave=False):
        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * clips.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return running_loss / total, acc

# Validation loop
def validate(model, loader, criterion, epoch=None):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for clips, labels, _, _ in tqdm(loader, desc="Validation", leave=False):
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * clips.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total

    # Confusion matrix
    if epoch is not None:
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(15)))
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(15), yticklabels=range(15))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Epoch {epoch}')
        writer.add_figure('Confusion_Matrix', fig, global_step=epoch)
        plt.close(fig)

    return running_loss / total, acc

# Main training
best_val_acc = 0.0
for epoch in range(1, epochs + 1):
    print(f"\n Epoch {epoch}/{epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate(model, val_loader, criterion, epoch=epoch)

    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

    # TensorBoard logging
    writer.add_scalars('Loss', {
        'Train': train_loss,
        'Val': val_loss
    }, epoch)

    writer.add_scalars('Accuracy', {
        'Train': train_acc,
        'Val': val_acc
    }, epoch)


    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_s3d_model.pt")
        print("Saved best model!")

writer.close()
print("\nTraining finished.")

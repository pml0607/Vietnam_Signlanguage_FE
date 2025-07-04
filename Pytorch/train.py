import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from dataset import PreprocessedClipDataset
from model import build_s3d_model
from train_utils import clip_collate_fn
from dataset import SingleStreamAugmentation, DualStreamAugmentation
import yaml

def load_config(config_path="../Configurate/train.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()


# Config
train_dir = config['paths']['train_dir']
val_dir = config['paths']['val_dir']
log_dir = config['paths']['log_dir']
best_model_path = config['paths']['best_model_path']

batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
lr = config['training']['learning_rate']
num_classes = config['training']['num_classes']
freeze_until = config['training']['freeze_until_layer']
num_workers = config['dataloader']['num_workers']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config['augmentation']['single_stream']['enable']:
    transform = SingleStreamAugmentation()
elif config['augmentation']['dual_stream']['enable']:
    transform = DualStreamAugmentation()
else:
    transform = None 

# Datasets & Dataloaders
train_dataset = PreprocessedClipDataset(train_dir, transform)
val_dataset = PreprocessedClipDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=clip_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=clip_collate_fn)

# Model
model = build_s3d_model(num_classes=15, pretrained=True, freeze_until_layer=12)
# import pdb;pdb.set_trace()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
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

    # scheduler.step(val_loss)
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print("Saved best model!")

writer.close()
print("\nTraining finished.")

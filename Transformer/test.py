print("Loading libraries...")
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, Lambda, Resize, RandomCrop, RandomHorizontalFlip
from pytorchvideo.transforms import UniformTemporalSubsample, RandomShortSideScale
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# ====== Load model checkpoint and label map ======
print("Loading model and image processor...")
model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)

def load_label_map(label_map_path):
    label2id, id2label = {}, {}
    with open(label_map_path, "r") as f:
        for line in f:
            label, idx = line.strip().split()
            idx = int(idx)
            label2id[label] = idx
            id2label[idx] = label
    return label2id, id2label

label2id, id2label = load_label_map("../vietnamsignlanguage/label_map.txt")

# ====== Video Dataset (6-channel RGB+heatmap) ======
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        clip = data['clip']
        label = data['label']
        video_id = data['video_id']
        clip_idx = data['clip_idx']
        if self.transform:
            clip = self.transform(clip)
        else:
            clip = clip.float()
        return clip, label, video_id, clip_idx

# ====== Collate Function ======
def clip_collate_fn(batch):
    clips = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    video_ids = [item[2] for item in batch]
    clip_idxs = [item[3] for item in batch]
    return clips, labels, video_ids, clip_idxs

# ====== RGB + Heatmap Normalization ======
class RGBHeatmapNormalize:
    def __init__(self, heatmap_mean=None, heatmap_std=None):
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        if heatmap_mean is None or heatmap_std is None:
            self.heatmap_mean = [0.5, 0.5, 0.5]
            self.heatmap_std = [0.5, 0.5, 0.5]
        else:
            self.heatmap_mean = heatmap_mean
            self.heatmap_std = heatmap_std

    def __call__(self, video):
        rgb = video[:3]
        heat = video[3:]
        for i in range(3):
            rgb[i] = (rgb[i] - self.rgb_mean[i]) / self.rgb_std[i]
            heat[i] = (heat[i] - self.heatmap_mean[i]) / self.heatmap_std[i]
        return torch.cat([rgb, heat], dim=0)

# ====== Heatmap Statistics ======
def compute_heatmap_statistics(data_loader):
    print("Computing heatmap statistics...")
    heatmap_sum = torch.zeros(3)
    heatmap_sum_sq = torch.zeros(3)
    total_pixels = 0
    for batch_idx, (clips, _, _, _) in enumerate(data_loader):
        if batch_idx >= 100:
            break
        clips = clips / 255.0
        heat = clips[:, 3:]
        flat = heat.view(-1, 3, heat.size(2) * heat.size(3) * heat.size(4))
        heatmap_sum += flat.sum(dim=[0, 2])
        heatmap_sum_sq += (flat ** 2).sum(dim=[0, 2])
        total_pixels += flat.size(0) * flat.size(2)
    mean = heatmap_sum / total_pixels
    std = torch.sqrt(heatmap_sum_sq / total_pixels - mean ** 2)
    print(f"Heatmap mean: {mean.tolist()}")
    print(f"Heatmap std: {std.tolist()}")
    return mean.tolist(), std.tolist()

# ====== Projection Layer Initialization ======
def initialize_rgb_heatmap_projection(model, strategy='duplicate'):
    old_proj = model.videomae.embeddings.patch_embeddings.projection
    new_proj = nn.Conv3d(
        in_channels=6,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    with torch.no_grad():
        if strategy == 'duplicate':
            new_proj.weight[:, :3] = old_proj.weight
            new_proj.weight[:, 3:] = old_proj.weight
        if old_proj.bias is not None:
            new_proj.bias = old_proj.bias.clone()
    model.videomae.embeddings.patch_embeddings.projection = new_proj
    return model

# ====== Setup Model and Transforms ======
print("Preparing temporary loader...")
temp_dataset = VideoDataset("../S3D/preprocessed_clips_6ch_v3/train")
temp_loader = DataLoader(temp_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=clip_collate_fn)

heatmap_mean, heatmap_std = compute_heatmap_statistics(temp_loader)
normalize = RGBHeatmapNormalize(heatmap_mean, heatmap_std)

model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
)
model.config.num_channels = 6
model.videomae.embeddings.patch_embeddings.num_channels = 6
model = initialize_rgb_heatmap_projection(model, strategy='duplicate')

height = image_processor.size.get("height", image_processor.size.get("shortest_edge", 224))
width = image_processor.size.get("width", image_processor.size.get("shortest_edge", 224))
resize_to = (height, width)

train_transform = Compose([
    UniformTemporalSubsample(model.config.num_frames),
    Lambda(lambda x: x / 255.0),
    normalize,
    RandomShortSideScale(min_size=256, max_size=320),
    RandomCrop(resize_to),
    RandomHorizontalFlip(p=0.5)
])

val_transform = Compose([
    UniformTemporalSubsample(model.config.num_frames),
    Lambda(lambda x: x / 255.0),
    normalize,
    Resize(resize_to)
])

train_dataset = VideoDataset("../S3D/preprocessed_clips_6ch_v3/train", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=clip_collate_fn)
val_dataset = VideoDataset("../S3D/preprocessed_clips_6ch_v3/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=clip_collate_fn)

# ====== Training and Evaluation ======
def train(model, train_loader, optimizer, scheduler=None, device="cuda"):
    model.to(device).train()
    total_loss = 0
    for clips, labels, _, _ in tqdm(train_loader, desc="Training"):
        clips = clips.to(device).permute(0, 2, 1, 3, 4)
        labels = labels.to(device)
        outputs = model(pixel_values=clips)
        loss = CrossEntropyLoss()(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device="cuda"):
    model.to(device).eval()
    correct = total = total_loss = 0
    with torch.no_grad():
        for clips, labels, _, _ in tqdm(val_loader, desc="Evaluating"):
            clips = clips.to(device).permute(0, 2, 1, 3, 4)
            labels = labels.to(device)
            outputs = model(pixel_values=clips)
            loss = CrossEntropyLoss()(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

# ====== Training Loop ======
best_val_acc = 0.0
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device=device)
    val_loss, val_acc = evaluate(model, val_loader, device=device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"../Weight/Transformer/best_transformers_model_(rgb+heatmap).pth")
        print(f"Saved best model at epoch {epoch + 1} with accuracy {best_val_acc:.4f}")

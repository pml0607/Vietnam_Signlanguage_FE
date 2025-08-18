# train.py

import os
import torch
import random
import numpy as np
import torch.optim as optim
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, get_cosine_with_hard_restarts_schedule_with_warmup
from torchvision.transforms import Compose, Lambda, Resize, RandomCrop, RandomHorizontalFlip
from pytorchvideo.transforms import UniformTemporalSubsample, RandomShortSideScale
import torchvision.transforms.functional as TF
from dataset import VideoDataset, clip_collate_fn, RGBHeatmapNormalize, load_label_map
from train_utils import train, evaluate, initialize_rgb_heatmap_projection, compute_heatmap_statistics

# ====== Config ======
model_ckpt = "MCG-NJU/videomae-base"
train_dir = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/S3D/preprocessed_clips_6ch_v3/train"
val_dir = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/S3D/preprocessed_clips_6ch_v3/val"
label_map_path = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/vietnamsignlanguage/label_map.txt"
weight_save_path = "../Weight/Transformer/best_transformers_model_(rgb+heatmap)_v3.pth"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
batch_size = 4
num_epochs = 50
lr = 1e-4
warmup_ratio = 0.1

# ====== Load Label Map ======
label2id, id2label = load_label_map(label_map_path)

# ====== Load Image Processor ======
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
height = image_processor.size.get("height", image_processor.size.get("shortest_edge", 224))
width = image_processor.size.get("width", image_processor.size.get("shortest_edge", 224))
resize_to = (height, width)

# ====== Heatmap Stats from Training Set ======
temp_dataset = VideoDataset(train_dir)
temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=4, shuffle=False, collate_fn=clip_collate_fn)
heatmap_mean, heatmap_std = compute_heatmap_statistics(temp_loader)
normalize = RGBHeatmapNormalize(heatmap_mean, heatmap_std)

model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)
model.config.num_channels = 6
model.videomae.embeddings.patch_embeddings.num_channels = 6
model = initialize_rgb_heatmap_projection(model, strategy='duplicate')
# ====== Define Transforms ======
class HeatmapJitter:
    def __init__(self, std=0.03):  # 3% noise
        self.std = std

    def __call__(self, video):
        rgb, heat = video[:3], video[3:]
        noise = torch.randn_like(heat) * self.std
        heat = heat + noise
        return torch.cat([rgb, heat], dim=0)

class RandomErasingVideo:
    def __init__(self, p=0.5, scale=(0.02, 0.2)):
        self.p = p
        self.scale = scale

    def __call__(self, video):
        if random.random() > self.p:
            return video
        C, T, H, W = video.shape
        for t in range(T):
            area = H * W
            target_area = random.uniform(*self.scale) * area
            h = int(round(np.sqrt(target_area)))
            w = int(round(np.sqrt(target_area)))
            x1 = random.randint(0, H - h)
            y1 = random.randint(0, W - w)
            video[:, t, x1:x1+h, y1:y1+w] = 0
        return video


train_transform = Compose([
    UniformTemporalSubsample(model.config.num_frames),
    Lambda(lambda x: x / 255.0),
    normalize,
    HeatmapJitter(std=0.03),
    RandomErasingVideo(p=0.5, scale=(0.02, 0.2)),
    RandomShortSideScale(min_size=256, max_size=320),
    RandomCrop(resize_to),
    RandomHorizontalFlip(p=0.3)
])

val_transform = Compose([
    UniformTemporalSubsample(model.config.num_frames),
    Lambda(lambda x: x / 255.0),
    normalize,
    Resize(resize_to)
])

# ====== Dataset & Dataloader ======
train_dataset = VideoDataset(train_dir, transform=train_transform)
val_dataset = VideoDataset(val_dir, transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=clip_collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=clip_collate_fn)


# ====== Optimizer & Scheduler ======
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = num_epochs * len(train_loader)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.2 * total_steps),
    num_training_steps=total_steps,
    num_cycles=2
)

# ====== Training Loop ======
best_val_acc = 0.0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, scheduler=scheduler, device=device)
    val_loss, val_acc = evaluate(model, val_loader, device=device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), weight_save_path)
        print(f"Saved best model at epoch {epoch + 1} with accuracy {best_val_acc:.4f}")

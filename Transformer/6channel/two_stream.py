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

# ====== Video Dataset (Separate RGB and Heatmap) ======
class VideoDatasetTwoStream(Dataset):
    def __init__(self, root_dir, transform_rgb=None, transform_heat=None):
        self.files = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.pt')
        ])
        self.transform_rgb = transform_rgb
        self.transform_heat = transform_heat

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        clip = data['clip']
        rgb = clip[:3]
        heat = clip[3:]
        label = data['label']
        video_id = data['video_id']
        clip_idx = data['clip_idx']
        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)
        if self.transform_heat:
            heat = self.transform_heat(heat)
        return rgb, heat, label, video_id, clip_idx

# ====== Collate Function ======
def twostream_collate_fn(batch):
    rgbs = torch.stack([item[0] for item in batch], dim=0)
    heats = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    video_ids = [item[3] for item in batch]
    clip_idxs = [item[4] for item in batch]
    return rgbs, heats, labels, video_ids, clip_idxs

# ====== Normalize transforms ======
rgb_normalize = Compose([
    Lambda(lambda x: x / 255.0),
    Lambda(lambda x: torch.stack([(x[i] - m) / s for i, (m, s) in enumerate(zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))]))
])

heatmap_normalize = Compose([
    Lambda(lambda x: x / 255.0),
    Lambda(lambda x: torch.stack([(x[i] - 0.5) / 0.5 for i in range(3)]))
])

# ====== Resize config ======
height = image_processor.size.get("height", image_processor.size.get("shortest_edge", 224))
width = image_processor.size.get("width", image_processor.size.get("shortest_edge", 224))
resize_to = (height, width)

transform_common = Compose([
    UniformTemporalSubsample(16),
    Resize(resize_to)
])

# ====== Loaders ======
train_dataset = VideoDatasetTwoStream(
    "../S3D/preprocessed_clips_6ch_v3/train",
    transform_rgb=Compose([
        transform_common,
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(resize_to),
        RandomHorizontalFlip(p=0.5),
        rgb_normalize
    ]),
    transform_heat=Compose([
        transform_common,
        RandomShortSideScale(min_size=256, max_size=320),
        RandomCrop(resize_to),
        RandomHorizontalFlip(p=0.5),
        heatmap_normalize
    ])
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=twostream_collate_fn)

val_dataset = VideoDatasetTwoStream(
    "../S3D/preprocessed_clips_6ch_v3/val",
    transform_rgb=Compose([
        transform_common,
        rgb_normalize
    ]),
    transform_heat=Compose([
        transform_common,
        heatmap_normalize
    ])
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=1, collate_fn=twostream_collate_fn)

# ====== Two-stream Model ======
class RGBHeatmapTwoStream(nn.Module):
    def __init__(self, model_ckpt, label2id, id2label, fusion_type='concat'):
        super().__init__()
        self.rgb_stream = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
        self.heatmap_stream = VideoMAEForVideoClassification.from_pretrained(model_ckpt, label2id=label2id, id2label=id2label)
        self.fusion_type = fusion_type
        hidden_size = self.rgb_stream.config.hidden_size

        if fusion_type == 'concat':
            self.classifier = nn.Linear(hidden_size * 2, len(label2id))
        elif fusion_type == 'add':
            self.classifier = nn.Linear(hidden_size, len(label2id))
        elif fusion_type == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)
            self.classifier = nn.Linear(hidden_size, len(label2id))

    def forward(self, rgb, heat):
        rgb = rgb.permute(0, 2, 1, 3, 4)
        heat = heat.permute(0, 2, 1, 3, 4)
        rgb_feat = self.rgb_stream.videomae(rgb).last_hidden_state.mean(dim=1)
        heat_feat = self.heatmap_stream.videomae(heat).last_hidden_state.mean(dim=1)

        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_feat, heat_feat], dim=1)
            logits = self.classifier(fused)
        elif self.fusion_type == 'add':
            fused = rgb_feat + heat_feat
            logits = self.classifier(fused)
        elif self.fusion_type == 'attention':
            # Reshape for attention: [B, 1, H]
            rgb_feat = rgb_feat.unsqueeze(1)
            heat_feat = heat_feat.unsqueeze(1)
            attn_out, _ = self.attn(rgb_feat, heat_feat, heat_feat)
            logits = self.classifier(attn_out.squeeze(1))
        return logits

model = RGBHeatmapTwoStream(model_ckpt, label2id, id2label, fusion_type='attention')  # or 'concat', 'add'

# ====== Training and Evaluation ======
#apply scheduler if needed
def train(model, train_loader, optimizer, scheduler=None, device="cuda"):
    model.to(device).train()
    total_loss = 0
    for rgb, heat, labels, _, _ in tqdm(train_loader, desc="Training"):
        rgb, heat, labels = rgb.to(device), heat.to(device), labels.to(device)
        logits = model(rgb, heat)
        loss = CrossEntropyLoss()(logits, labels)
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
        for rgb, heat, labels, _, _ in tqdm(val_loader, desc="Evaluating"):
            rgb, heat, labels = rgb.to(device), heat.to(device), labels.to(device)
            logits = model(rgb, heat)
            loss = CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(val_loader), correct / total

# ====== Training Loop ======
best_val_acc = 0.0
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train(model, train_loader, optimizer, device=device)
    val_loss, val_acc = evaluate(model, val_loader, device=device)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"../Weight/Transformer/best_twostream_model.pth")
        print(f"Saved best model at epoch {epoch + 1} with accuracy {best_val_acc:.4f}")

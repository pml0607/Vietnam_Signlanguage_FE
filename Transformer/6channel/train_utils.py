# train_utils.py (updated to use dataset_utils)

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, VideoMAEForVideoClassification
from torchvision.transforms import Compose, Lambda, Resize, RandomCrop, RandomHorizontalFlip
from pytorchvideo.transforms import UniformTemporalSubsample, RandomShortSideScale
from dataset import VideoDataset, clip_collate_fn, RGBHeatmapNormalize, load_label_map


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


def initialize_rgb_heatmap_projection(model, strategy='duplicate'):
    old_proj = model.videomae.embeddings.patch_embeddings.projection
    new_proj = torch.nn.Conv3d(
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
            new_proj.bias = torch.nn.Parameter(old_proj.bias.clone())
    model.videomae.embeddings.patch_embeddings.projection = new_proj
    return model


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

    avg_loss = total_loss / len(train_loader)
    return avg_loss


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

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

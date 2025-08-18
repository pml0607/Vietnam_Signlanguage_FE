# dataset_utils.py

import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

# ====== Load label map ======
def load_label_map(label_map_path):
    label2id, id2label = {}, {}
    with open(label_map_path, "r") as f:
        for line in f:
            label, idx = line.strip().split()
            idx = int(idx)
            label2id[label] = idx
            id2label[idx] = label
    return label2id, id2label

# ====== Custom Normalize for RGB + Heatmap ======
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

# ====== Dataset Class for RGB+Heatmap video ======
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

# ====== Collate Function for Video Clip Loader ======
def clip_collate_fn(batch):
    clips = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    video_ids = [item[2] for item in batch]
    clip_idxs = [item[3] for item in batch]
    return clips, labels, video_ids, clip_idxs

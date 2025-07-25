# validate.py

import os
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
from dataset import VideoDataset, clip_collate_fn, RGBHeatmapNormalize, load_label_map
from train_utils import initialize_rgb_heatmap_projection, compute_heatmap_statistics

# ====== Load configs ======
model_ckpt = "MCG-NJU/videomae-base"
weight_path = "../Weight/Transformer/best_transformers_model_(rgb+heatmap).pth"
val_dir = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/S3D/preprocessed_clips_6ch_v3/val"
label_map_path = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/vietnamsignlanguage/label_map.txt"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# ====== Load label maps ======
label2id, id2label = load_label_map(label_map_path)

# ====== Load model ======
print("Loading model and weights...")
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
)
model.config.num_channels = 6
model.videomae.embeddings.patch_embeddings.num_channels = 6
model = initialize_rgb_heatmap_projection(model, strategy='duplicate')
model.load_state_dict(torch.load(weight_path, map_location="cpu"))
model.to(device)
model.eval()

# ====== Image processor and transform ======
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
height = image_processor.size.get("height", image_processor.size.get("shortest_edge", 224))
width = image_processor.size.get("width", image_processor.size.get("shortest_edge", 224))
resize_to = (height, width)

# ====== Estimate heatmap stats from training set ======
print("Computing heatmap stats from training set...")
temp_dataset = VideoDataset(
    "/home/21013187/work/linh/Vietnam_Signlanguage_FE/S3D/preprocessed_clips_6ch_v3/train"
)
temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=4, shuffle=False, collate_fn=clip_collate_fn)
heatmap_mean, heatmap_std = compute_heatmap_statistics(temp_loader)
normalize = RGBHeatmapNormalize(heatmap_mean, heatmap_std)

# ====== Build val transform ======
from torchvision.transforms import Compose, Lambda, Resize
from pytorchvideo.transforms import UniformTemporalSubsample

val_transform = Compose([
    UniformTemporalSubsample(model.config.num_frames),
    Lambda(lambda x: x / 255.0),
    normalize,
    Resize(resize_to)
])

# ====== Load val dataset ======
val_dataset = VideoDataset(val_dir, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=clip_collate_fn)

# ====== Validate ======
print("Running validation...")
all_preds = []
all_labels = []

with torch.no_grad():
    for clips, labels, _, _ in tqdm(val_loader):
        clips = clips.to(device).permute(0, 2, 1, 3, 4)
        labels = labels.to(device)
        outputs = model(pixel_values=clips)
        preds = outputs.logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ====== Report ======
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(id2label))]))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

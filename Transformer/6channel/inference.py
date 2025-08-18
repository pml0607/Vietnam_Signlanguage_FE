import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, Lambda
from pytorchvideo.transforms import UniformTemporalSubsample
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# ==== Cấu hình ====
model_ckpt = "MCG-NJU/videomae-base"
model_weights_path = "../Weight/Transformer/best_transformers_model_(rgb+heatmap).pth"
label_map_path = "/home/21013187/work/linh/Vietnam_Signlanguage_FE/vietnamsignlanguage/label_map.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==== Hàm load label map ====
def load_label_map(path):
    label2id, id2label = {}, {}
    with open(path, "r") as f:
        for line in f:
            label, idx = line.strip().split()
            label2id[label] = int(idx)
            id2label[int(idx)] = label
    return label2id, id2label

label2id, id2label = load_label_map(label_map_path)

# ==== Load mô hình ====
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model.config.num_channels = 6
model.videomae.embeddings.patch_embeddings.num_channels = 6

# ==== Thay thế lớp projection cho 6 kênh (RGB + Heatmap) ====
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
            new_proj.bias = nn.Parameter(old_proj.bias.clone())
    model.videomae.embeddings.patch_embeddings.projection = new_proj
    return model

model = initialize_rgb_heatmap_projection(model)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# ==== Chuẩn hóa giống lúc train ====
class RGBHeatmapNormalize:
    def __init__(self, heatmap_mean, heatmap_std):
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]
        self.heatmap_mean = heatmap_mean
        self.heatmap_std = heatmap_std

    def __call__(self, video):
        rgb = video[:3]
        heat = video[3:]
        for i in range(3):
            rgb[i] = (rgb[i] - self.rgb_mean[i]) / self.rgb_std[i]
            heat[i] = (heat[i] - self.heatmap_mean[i]) / self.heatmap_std[i]
        return torch.cat([rgb, heat], dim=0)

# ==== Load file clip .pt ====
def load_clip(clip_path, transform):
    data = torch.load(clip_path)
    clip = data['clip'].float() / 255.0  # Normalize về [0,1]
    clip = transform(clip)               # Apply transform
    clip = clip.unsqueeze(0)             # [1, C, T, H, W]
    return clip.to(device)

# ==== Transform ====
# Cập nhật mean/std tương ứng heatmap của bạn
heatmap_mean = [0.5, 0.5, 0.5]
heatmap_std = [0.5, 0.5, 0.5]
normalize = RGBHeatmapNormalize(heatmap_mean, heatmap_std)

num_frames = model.config.num_frames
height = image_processor.size.get("height", image_processor.size.get("shortest_edge", 224))
width = image_processor.size.get("width", image_processor.size.get("shortest_edge", 224))
resize_to = (height, width)

inference_transform = Compose([
    UniformTemporalSubsample(num_frames),
    normalize,
    Resize(resize_to),
])

# ==== Dự đoán clip ====
@torch.no_grad()
def predict(clip_path):
    clip = load_clip(clip_path, inference_transform)  # shape: [1, C, T, H, W]
    clip = clip.permute(0, 2, 1, 3, 4)                # [B, T, C, H, W] → [B, C, T, H, W]
    outputs = model(pixel_values=clip)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    pred_label = id2label[pred_idx]
    confidence = probs[0, pred_idx].item()
    return pred_label, confidence

# ==== Gọi thử ====
if __name__ == "__main__":
    clip_path = "path/to/your_clip.pt"  # ví dụ: "./sample_clip.pt"
    pred_label, confidence = predict(clip_path)
    print(f"Predicted: {pred_label} (Confidence: {confidence:.2f})")

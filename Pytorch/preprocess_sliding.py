import os
import torch
import pandas as pd
import torchvision
from torchvision.transforms import functional as F
from tqdm import tqdm
import yaml

def load_config(path="../Configurate/data_preprocess.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def normalize_clip(clip):
    mean = torch.tensor([0.43216, 0.394666, 0.37645])[:, None, None, None]
    std = torch.tensor([0.22803, 0.22145, 0.216989])[:, None, None, None]
    return (clip / 255.0 - mean) / std

def extract_clips(video_tensor, num_frames=64, stride=32):
    C, T, H, W = video_tensor.shape
    clips = []
    for start in range(0, max(1, T - num_frames + 1), stride):
        end = start + num_frames
        if end <= T:
            clip = video_tensor[:, start:end]
        else:
            pad_len = end - T
            last_frame = video_tensor[:, -1:, :, :]
            clip = torch.cat([video_tensor[:, start:], last_frame.repeat(1, pad_len, 1, 1)], dim=1)
        clips.append((clip, start))
    return clips

def resize_clip(clip, size=(224, 224)):
    return torch.stack([
        F.resize(clip[:, t], size, antialias=True) for t in range(clip.shape[1])
    ], dim=1)

def get_video_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def process_csv(csv_path, split_name, output_dir, num_frames=64, stride=32, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    clip_idx_global = 0
    for row in tqdm(df.itertuples(), total=len(df), desc=f"[{split_name}] Processing videos"):
        video_path = row.video_path
        label = int(row.label)
        video_id = get_video_id(video_path)

        try:
            video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
            video = video.permute(3, 0, 1, 2)  # C x T x H x W

            clips = extract_clips(video, num_frames=num_frames, stride=stride)

            for clip_tensor, clip_idx in clips:
                clip_tensor = resize_clip(clip_tensor, size)
                # clip_tensor = normalize_clip(clip_tensor)

                save_data = {
                    'clip': clip_tensor,
                    'label': label,
                    'video_id': video_id,
                    'clip_idx': clip_idx
                }

                save_path = os.path.join(output_dir, f"{clip_idx_global:06d}.pt")
                torch.save(save_data, save_path)
                clip_idx_global += 1

        except Exception as e:
            print(f"Error in processing {video_path}: {e}")

if __name__ == "__main__":
    config = load_config()
    output_root = config['output_root']
    os.makedirs(output_root, exist_ok=True)

    for split_name, split_info in config['splits'].items():
        csv_file = split_info['csv_path']
        out_dir = os.path.join(output_root, split_name)

        process_csv(
            csv_path=csv_file,
            split_name=split_name,
            output_dir=out_dir,
            num_frames=config['video']['num_frames'],
            stride=config['video']['stride'],
            size=tuple(config['video']['size'])
        )


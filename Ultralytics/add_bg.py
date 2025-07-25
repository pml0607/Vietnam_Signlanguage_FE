import os
import cv2
import torch
import numpy as np
from glob import glob
from torchvision.transforms.functional import resize
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import yaml
from multiprocessing import Process

# === Load config ===
with open("../Configurate/segmentation.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bitwised_root = cfg["add_bg"]["paths"]["bitwised_root"]
mask_root = cfg["add_bg"]["paths"]["mask_root"]
background_folder = cfg["add_bg"]["paths"]["background_folder"]
output_root = cfg["add_bg"]["paths"]["output_root"]
target_size = (640, 640)

# === Load all videos ===
video_paths = glob(os.path.join(bitwised_root, "*", "rgb", "*.avi"))

# === Load background once globally ===
bg_paths = glob(os.path.join(background_folder, "**", "*.jpg"), recursive=True)
backgrounds = [cv2.resize(cv2.imread(p), target_size) for p in bg_paths]

def process_video_gpu(video_path, device_id, backgrounds):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    relative_path = os.path.relpath(video_path, bitwised_root)
    mask_path = os.path.join(mask_root, relative_path)

    if not os.path.exists(mask_path):
        print(f"[SKIP] No mask for {relative_path}")
        return

    masked_cap = cv2.VideoCapture(video_path)
    mask_cap = cv2.VideoCapture(mask_path)
    fps = masked_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    bg_tensor = random.choice(backgrounds)
    bg_tensor = torch.from_numpy(cv2.cvtColor(bg_tensor, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(device)

    output_path = os.path.join(output_root, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    while True:
        ret1, frame_person = masked_cap.read()
        ret2, frame_mask = mask_cap.read()
        if not ret1 or not ret2:
            break

        person = cv2.resize(frame_person, target_size)
        mask = cv2.resize(frame_mask, target_size)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        person_tensor = torch.from_numpy(cv2.cvtColor(person, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().to(device)
        mask_tensor = torch.from_numpy(mask_gray).float().to(device)

        mask_bin = (mask_tensor > 10).float()
        mask_bin_3ch = mask_bin.unsqueeze(0).repeat(3, 1, 1)

        fg = person_tensor * mask_bin_3ch
        bg_part = bg_tensor * (1 - mask_bin_3ch)
        final = fg + bg_part

        final_np = final.byte().permute(1, 2, 0).cpu().numpy()
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
        out.write(final_bgr)

    out.release()
    masked_cap.release()
    mask_cap.release()
    print(f"[GPU {device_id}] Done: {output_path}")


def worker(device_id, video_batch):
    print(f"[GPU {device_id}] Start {len(video_batch)} videos")
    for video_path in video_batch:
        process_video_gpu(video_path, device_id, backgrounds)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Using {num_gpus} GPUs")

    # Chia video thành các batch cho từng GPU
    video_batches = [[] for _ in range(num_gpus)]
    for idx, video_path in enumerate(video_paths):
        video_batches[idx % num_gpus].append(video_path)

    # Khởi chạy từng tiến trình cho mỗi GPU
    processes = []
    for gpu_id in range(num_gpus):
        p = Process(target=worker, args=(gpu_id, video_batches[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All done!")

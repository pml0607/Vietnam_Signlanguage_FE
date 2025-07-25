import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
import yaml

# === Load config ===
with open("../Configurate/segmentation.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    

video_root = cfg["segmentation"]["paths"]["video_root"]
output_root = cfg["segmentation"]["paths"]["output_root"]
bitwise_dir = os.path.join(output_root, "bitwised")
mask_dir = os.path.join(output_root, "mask")
model_path = cfg["segmentation"]["model"]["path"]

model = YOLO(model_path)

# Get all video files
video_list = glob(os.path.join(video_root, "*", "rgb", "*.avi"))

# Split into batches
batch_size = 10
batches = [video_list[i:i+batch_size] for i in range(0, len(video_list), batch_size)]

# Function to process each video
def process_video(video_path):
    try:
        rel_path = os.path.relpath(video_path, video_root)
        bitwise_output_path = os.path.join(bitwise_dir, rel_path)
        mask_output_path = os.path.join(mask_dir, rel_path)

        os.makedirs(os.path.dirname(bitwise_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = 640, 640
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        out_bitwise = cv2.VideoWriter(bitwise_output_path, fourcc, fps, (w, h))
        out_mask = cv2.VideoWriter(mask_output_path, fourcc, fps, (w, h), isColor=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (w, h))
            results = model(frame)
            masks = results[0].masks

            if masks is not None and masks.data.shape[0] > 0:
                mask = masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (w, h))
                out_mask.write(mask)

                mask_3ch = cv2.merge([mask, mask, mask]).astype(frame.dtype)
                person_only = cv2.bitwise_and(frame, mask_3ch)
            else:
                person_only = np.zeros_like(frame)
                out_mask.write(np.zeros((h, w), dtype=np.uint8))

            out_bitwise.write(person_only)

        cap.release()
        out_bitwise.release()
        out_mask.release()
        return True
    except Exception as e:
        print(f"Error in video {video_path}: {e}")
        return False

for batch_idx, batch in enumerate(batches):
    print(f"\nProessing batch {batch_idx+1}/{len(batches)} ({len(batch)} videos)")
    for video_path in tqdm(batch, desc=f"Batch {batch_idx+1}"):
        process_video(video_path)

print("\nProcessing complete!")

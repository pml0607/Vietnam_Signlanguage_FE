import os
import cv2
import numpy as np
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm

# === CẤU HÌNH ===
video_root = "/home/21013187/Vietnam_Signlanguage_FE/vietnamsignlanguage/mediapipe"
output_root = "/home/21013187/Vietnam_Signlanguage_FE/segmented_videos_v2"
bitwise_dir = os.path.join(output_root, "bitwised")
mask_dir = os.path.join(output_root, "mask")

model_path = "/home/21013187/Vietnam_Signlanguage_FE/Ultralytics/yolo11l-seg.pt"
model = YOLO(model_path)

# === LẤY DANH SÁCH VIDEO ===
video_list = glob(os.path.join(video_root, "**", "*.avi"), recursive=True)

# === CHIA THÀNH BATCH ===
batch_size = 10
batches = [video_list[i:i+batch_size] for i in range(0, len(video_list), batch_size)]

# === HÀM XỬ LÝ 1 VIDEO ===
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
        print(f"❌ Lỗi với video {video_path}: {e}")
        return False

# === XỬ LÝ THEO BATCH ===
for batch_idx, batch in enumerate(batches):
    print(f"\n🚀 Đang xử lý batch {batch_idx+1}/{len(batches)} ({len(batch)} videos)")
    for video_path in tqdm(batch, desc=f"Batch {batch_idx+1}"):
        process_video(video_path)

print("\n✅ Xử lý toàn bộ video hoàn tất.")

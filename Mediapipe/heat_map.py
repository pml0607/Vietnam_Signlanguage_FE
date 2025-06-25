import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

# === Color Map ===
POSE_COLOR = (255, 200, 255)
POSE_HAND_COLOR = (225,178,51)
FINGER_COLORS = {
    'thumb':  (255, 0, 0),    # Blue
    'index':  (0, 255, 0),    # Green
    'middle': (0, 0, 255),    # Red
    'ring':   (0, 0, 0),      # Black
    'pinky':  (255, 255, 0),  # Cyan
    'sen':    (255, 200, 255)  # Coral
}

# === Connections ===
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,23),(12,24),(23,24),(23,25),
    (24,26)
]

POSE_HAND_CONECTIONS = [
    (12,14),(14,16),(16,18),(16,22),(18,20),(16,20),
    (11,13),(13,15),(15,17),(15,21),(17,19),(15,19)
]
FINGER_CONNECTIONS = {
    'thumb':  [(0,1), (1,2), (2,3), (3,4)],
    'index':  [(0,5), (5,6), (6,7), (7,8)],
    'middle': [(9,10), (10,11), (11,12)],
    'ring':   [(13,14), (14,15), (15,16)],
    'pinky':  [(0,17), (17,18), (18,19), (19,20)],
    'sen':    [(5,9), (9,13), (13,17)]
}

def draw_pose_lines(image, keypoints_pixel, color):
    for i, j in POSE_CONNECTIONS:
        if i < len(keypoints_pixel) and j < len(keypoints_pixel):
            pt1, pt2 = tuple(keypoints_pixel[i]), tuple(keypoints_pixel[j])
            cv2.line(image, pt1, pt2, color=color, thickness=2, lineType=cv2.LINE_AA)
    return image

def draw_hand_pose_lines(image, keypoints_pixel, color):
    for i, j in POSE_HAND_CONECTIONS:
        if i < len(keypoints_pixel) and j < len(keypoints_pixel):
            pt1, pt2 = tuple(keypoints_pixel[i]), tuple(keypoints_pixel[j])
            cv2.line(image, pt1, pt2, color=color, thickness=2, lineType=cv2.LINE_AA)
    return image

def draw_hand_lines(image, keypoints_pixel, offset):
    for finger, pairs in FINGER_CONNECTIONS.items():
        for i, j in pairs:
            i += offset
            j += offset
            if i < len(keypoints_pixel) and j < len(keypoints_pixel):
                pt1 = tuple(keypoints_pixel[i])
                pt2 = tuple(keypoints_pixel[j])
                cv2.line(image, pt1, pt2, color=FINGER_COLORS[finger], thickness=2, lineType=cv2.LINE_AA)
    return image

def visualize_on_video(video_path, npy_path, label, output_path,
                       font_scale=1.0, font_color=(0, 0, 255)):
    keypoints_seq = np.load(npy_path)  # (T, 75, 3)

    # Lấy thông số video gốc
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()  # không dùng frame nữa

    T = min(len(keypoints_seq), frame_count)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for t in range(T):
        # Tạo nền đen
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        keypoints = keypoints_seq[t]  # (75, 3)
        keypoints_xy = keypoints[:, :2]
        keypoints_pixel = np.round(keypoints_xy * [w, h]).astype(int)

        # Vẽ pose và tay
        draw_pose_lines(frame, keypoints_pixel, color=POSE_COLOR)
        draw_hand_pose_lines(frame, keypoints_pixel, color=POSE_HAND_COLOR)
        draw_hand_lines(frame, keypoints_pixel, offset=33)  # left
        draw_hand_lines(frame, keypoints_pixel, offset=54)  # right

        # Ghi nhãn
        cv2.putText(frame, f'Label: {label}', (20, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=font_color, thickness=2)

        out_writer.write(frame)

    out_writer.release()
    print(f"✅ Đã lưu video nền đen có skeleton: {output_path}")
    
def process_csv(csv_path, output_dir='output'):
    #hien thi so luong video trong csv
    print(f"📂 Đang xử lý file CSV: {csv_path}"
          f"\n📂 Tổng số video: {sum(1 for _ in open(csv_path)) - 1}")  # trừ header
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Đang xử lý batch"):
            video_path = row['video_path']
            npy_path = row['file_path']
            label = row['label']

            # Đặt tên file output theo tên file video
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(output_dir, f"{video_name}_skeleton.mp4")

            try:
                visualize_on_video(video_path, npy_path, label, out_path)
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {video_path}: {e}")

if __name__ == "__main__":
    csv_path = '../cnn_val_1.corpus.csv' 
    process_csv(csv_path, output_dir='../heat_map_data/val')
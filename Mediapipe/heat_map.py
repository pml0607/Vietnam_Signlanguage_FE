import cv2
import numpy as np
import os
import csv
from tqdm import tqdm
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config('../Configurate/landmark_config.yaml')
# Color Map
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

# Connections
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

def visualize_on_video(video_path, npy_path, output_path):
    keypoints_seq = np.load(npy_path)  # (T, 75, 3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Couldn't open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    T = min(len(keypoints_seq), frame_count)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for t in range(T):
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        keypoints = keypoints_seq[t]  # (75, 3)
        keypoints_xy = keypoints[:, :2]
        keypoints_pixel = np.round(keypoints_xy * [w, h]).astype(int)

        draw_pose_lines(frame, keypoints_pixel, color=POSE_COLOR)
        draw_hand_pose_lines(frame, keypoints_pixel, color=POSE_HAND_COLOR)
        draw_hand_lines(frame, keypoints_pixel, offset=33)  # left
        draw_hand_lines(frame, keypoints_pixel, offset=54)  # right
        out_writer.write(frame)

    out_writer.release()
    print(f"Saved skeleton video: {output_path}")
    
def process_csv(csv_path, output_dir='output'):
    print(f"Processing csv file: {csv_path}"
          f"\nNumber of videos: {sum(1 for _ in open(csv_path)) - 1}") 
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Processing batch"):
            video_path = row['video_path']
            npy_path = row['file_path']
            label = row['label']

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(output_dir, f"{video_name}_skeleton.mp4")

            try:
                visualize_on_video(video_path, npy_path, out_path)
            except Exception as e:
                print(f"Error in processing {video_path}: {e}")
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        os.makedirs(output_dir, exist_ok=True)
        # Create output CSV file to store video and skeleton paths
        # If it doesn't exist, create it
        output_csv_path = os.path.join(output_dir, 'skeleton_paths.csv' ) 
        os.makedirs(output_dir, exist_ok=True)
        with open(output_csv_path, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(['video_path', 'skeleton_path', 'label'])
            for row in tqdm(reader, desc="Processing CSV for output"):
                video_path = row['video_path']
                npy_path = row['file_path']
                label = row['label']

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(output_dir, f"{video_name}_skeleton.mp4")

                try:
                    writer.writerow([video_path, out_path, label])
                except Exception as e:
                    print(f"Error in processing {video_path}: {e}")
            


if __name__ == "__main__":
    train_csv_path = config['input']['train_csv_path']
    val_csv_path = config['input']['val_csv_path']
    train_output_dir = config['output']['train_dir']
    val_output_dir = config['output']['val_dir']
    process_csv(train_csv_path, output_dir=train_output_dir)
    process_csv(val_csv_path, output_dir=val_output_dir)
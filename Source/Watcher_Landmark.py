import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import từ file gốc
from model.pose_hrnet import get_pose_net
from utils.skeleton import (
    pose_process, norm_numpy_totensor,
    stack_flip, merge_hm, load_config_from_yaml
)

class LandmarkWatcher(FileSystemEventHandler):
    def __init__(self, input_dir, output_dir, config_path, checkpoint_path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.processing = set()  
        
        # Khởi tạo model
        self.model = self._load_model(config_path, checkpoint_path)
        print(f"[INIT] HRNet model loaded successfully")
        
    def _load_model(self, config_path, checkpoint_path):
        """Load HRNet model từ config và checkpoint"""
        print(f"[INIT] Loading model from {checkpoint_path}")
        
        # Load config
        cfg = load_config_from_yaml(config_path)
        model = get_pose_net(cfg, is_train=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = checkpoint['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('backbone.', '').replace('keypoint_head.', '')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("[INIT] Model moved to GPU")
        else:
            print("[INIT] Model running on CPU")
            
        model.eval()
        return model
        
    def on_created(self, event):
        """Chỉ xử lý video mới được tạo"""
        if not event.is_directory and event.src_path.endswith('.avi'):
            time.sleep(2)  # Tăng thời gian chờ để đảm bảo file đã được ghi xong
            
            if self._is_video_ready(event.src_path):
                self.process_video(event.src_path)
            else:
                print(f"[WARNING] Video file not ready: {event.src_path}")
    
    def _is_video_ready(self, video_path):
        """Kiểm tra video file đã sẵn sàng để xử lý"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception as e:
            print(f"[ERROR] Cannot validate video {video_path}: {e}")
            return False
    
    def process_video(self, video_path):
        """Xử lý video và trích xuất landmarks - chỉ xử lý video mới"""
        video_file = Path(video_path)
        
        if video_file.name in self.processing:
            print(f"[SKIP] Already processing: {video_file.name}")
            return
        
        output_file = self.output_dir / f"{video_file.stem}.npy"
        if output_file.exists():
            print(f"[SKIP] Landmark already exists: {output_file}")
            return
            
        self.processing.add(video_file.name)
        
        try:
            print(f"[NEW VIDEO] Processing: {video_file.name}")
            
            landmarks = self.extract_landmarks_hrnet(video_path)
            
            np.save(output_file, landmarks)
            
            print(f"[SUCCESS] Landmarks saved: {output_file.name} - Shape: {landmarks.shape}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {video_path}: {e}")
            if output_file.exists():
                try:
                    output_file.unlink()
                    print(f"[CLEANUP] Removed incomplete file: {output_file}")
                except:
                    pass
        finally:
            self.processing.discard(video_file.name)
    
    def extract_landmarks_hrnet(self, video_path):
        """Trích xuất landmarks sử dụng HRNet model"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
            
        output_list = []
        frame_count = 0
        
        with torch.no_grad():  # Không cần gradient cho inference
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                frame_count += 1
                
                img = cv2.resize(img, (256, 256))
                img = cv2.flip(img, flipCode=1)  # Flip horizontal
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Multi-scale inference
                multi_scales = [512, 640]
                out_list = []
                
                for scale in multi_scales:
                    # Resize theo scale
                    if scale != 512:
                        img_temp = cv2.resize(img, (scale, scale))
                    else:
                        img_temp = img.copy()
                    
                    # Stack và flip để tăng cường dữ liệu
                    img_temp = stack_flip(img_temp)
                    
                    # Normalize và convert to tensor
                    img_tensor = norm_numpy_totensor(img_temp)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        img_tensor = img_tensor.cuda()

                    # Forward pass
                    hms = self.model(img_tensor)
                    
                    # Resize heatmap nếu cần
                    if scale != 512:
                        hms = torch.nn.functional.interpolate(
                            hms, 
                            (img.shape[0] // 4, img.shape[1] // 4), 
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    out_list.append(hms)

                # Merge all heatmaps
                out = merge_hm(out_list)
                
                # Convert heatmap to keypoint coordinates
                result = out.reshape((133, -1))  # 133 keypoints cho wholebody
                result = torch.argmax(result, dim=1).cpu().numpy().squeeze()

                # Convert 1D index to (x, y) coordinates
                h, w = img.shape[0] // 4, img.shape[1] // 4
                y = result // w
                x = result % w
                
                # Initialize prediction array
                pred = np.zeros((133, 3), dtype=np.float32)
                pred[:, 0], pred[:, 1] = x, y

                # Post-processing
                heatmap_np = out.cpu().detach().numpy().reshape((133, h, w))
                pred = pose_process(pred, heatmap_np)
                
                # Scale back to original resolution
                pred[:, :2] *= 4.0

                # Accumulate results
                output_list.append(pred)
                
                # Log progress mỗi 30 frames
                if frame_count % 30 == 0:
                    print(f"[PROGRESS] Processed {frame_count} frames...")

        cap.release()
        
        if len(output_list) == 0:
            raise ValueError(f"No frames processed from video: {video_path}")
            
        landmarks_array = np.array(output_list)
        print(f"[INFO] Total frames processed: {len(output_list)}")
        
        return landmarks_array

def start_landmark_watcher(input_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_video", 
                          output_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_landmark",
                          config_path="/work/21013187/phuoc/visl-i3d/src/config/wholebody_w48_384x288.yaml",
                          checkpoint_path="/work/21013187/phuoc/visl-i3d/src/checkpoint/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"):
    """
    Khởi động landmark watcher
    
    Args:
        input_dir: Thư mục chứa video input
        output_dir: Thư mục lưu landmarks (.npy)
        config_path: Đường dẫn đến file config YAML
        checkpoint_path: Đường dẫn đến checkpoint model
    """
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[START] Starting Landmark Watcher...")
    print(f"[START] Input directory: {input_dir}")
    print(f"[START] Output directory: {output_dir}")
    print(f"[START] Config: {config_path}")
    print(f"[START] Checkpoint: {checkpoint_path}")
    
    # Khởi tạo watcher
    watcher = LandmarkWatcher(input_dir, output_dir, config_path, checkpoint_path)
    
    print("[START] Ready to process new videos only...")

    observer = Observer()
    observer.schedule(watcher, input_dir, recursive=False)
    observer.start()
    
    print(f"[WATCH] Watching for NEW videos in: {input_dir}")
    print(f"[WATCH] Will save landmarks to: {output_dir}")
    print(f"[WATCH] Ready to process incoming videos...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[STOP] Stopping landmark watcher...")
        observer.stop()
    
    observer.join()
    print("[STOP] Landmark watcher stopped")

if __name__ == "__main__":
    start_landmark_watcher(
        input_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_video",
        output_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_landmark", 
        config_path="/work/21013187/phuoc/visl-i3d/src/config/wholebody_w48_384x288.yaml",
        checkpoint_path="/work/21013187/phuoc/visl-i3d/src/checkpoint/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
    )
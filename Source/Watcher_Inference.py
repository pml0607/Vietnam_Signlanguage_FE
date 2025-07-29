import os
import json
import time
import uuid
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import torch
import cv2

# Import model và utils
from Transformer.rgb_landmark.Model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from Transformer.rgb_landmark.dataset import collate_fn
from Transformer.rgb_landmark.dataset.utils import get_label_map
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Resize
from Transformer.rgb_landmark.dataset.video_transforms import (
    ApplyTransformToKey,
    Normalize as VideoNormalize,
    UniformTemporalSubsample,
)
from Transformer.rgb_landmark.dataset.landmark_transforms import (
    Normalize as LandmarkNormalize,
    UniformTemporalSubsample as LandmarkUniformTemporalSubsample,
    Resize as LandmarkResize,
)

class InferenceSampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class InferenceWatcher:
    def __init__(self, video_dir, landmark_dir, result_dir, model_ckpt_path, dataset_root_path):
        self.video_dir = Path(video_dir)
        self.landmark_dir = Path(landmark_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        self.processed_pairs = set()
        self.processing_pairs = set()  
        
        # Load model và label mapping
        self.model, self.trainer, self.id2label = self._load_model(model_ckpt_path, dataset_root_path)
        
        # Theo dõi cả 2 thư mục
        self.video_observer = Observer()
        self.landmark_observer = Observer()
        
        video_handler = FileHandler(self, 'video')
        landmark_handler = FileHandler(self, 'landmark')
        
        self.video_observer.schedule(video_handler, str(video_dir), recursive=False)
        self.landmark_observer.schedule(landmark_handler, str(landmark_dir), recursive=False)
        
        print(f"[INIT] Inference Watcher with VideoMAE model initialized")
        print(f"[INIT] Video dir: {video_dir}")
        print(f"[INIT] Landmark dir: {landmark_dir}")
        print(f"[INIT] Result dir: {result_dir}")
        print(f"[INIT] Model loaded with {len(self.id2label)} classes")
    
    def _load_model(self, model_ckpt_path, dataset_root_path):
        """Load VideoMAE model và setup trainer"""
        print(f"[MODEL] Loading model from {model_ckpt_path}")        
        label2id, id2label = get_label_map(dataset_root_path)
        # Load model
        model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        )
        
        # Setup trainer for inference
        args = TrainingArguments(
            output_dir="./tmp_inference",
            per_device_eval_batch_size=1,  
            dataloader_drop_last=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collate_fn,
        )
        
        print(f"[MODEL] VideoMAE model loaded successfully")
        return model, trainer, id2label
    
    def on_file_created(self, file_type):
        """Được gọi khi có file mới được tạo"""
        print(f"[EVENT] New {file_type} file detected, checking for pairs...")
        self.check_and_process_pairs()
    
    def check_and_process_pairs(self):
        """Kiểm tra và xử lý các cặp {video, landmark} hoàn chỉnh - chỉ xử lý 1 cặp mới nhất"""
        try:
            # Lấy danh sách file (không có extension)
            video_files = {f.stem for f in self.video_dir.glob('*.avi') if f.is_file()}
            landmark_files = {f.stem for f in self.landmark_dir.glob('*.npy') if f.is_file()}
            
            # Tìm các cặp hoàn chỉnh chưa được xử lý
            complete_pairs = video_files & landmark_files
            new_pairs = complete_pairs - self.processed_pairs - self.processing_pairs
            
            if new_pairs:
                for pair_id in new_pairs:
                    landmark_path = self.landmark_dir / f"{pair_id}.npy"
                    try:
                        landmark_data = np.load(landmark_path)
                        print(f"[SHAPE CHECK] {pair_id}: {landmark_data.shape}")
                    except Exception as e:
                        print(f"[ERROR] Could not load landmark {pair_id}: {e}")
                # Chỉ xử lý 1 cặp mới nhất (theo thời gian tạo file)
                latest_pair = max(new_pairs, key=lambda x: max(
                    (self.video_dir / f"{x}.avi").stat().st_mtime,
                    (self.landmark_dir / f"{x}.npy").stat().st_mtime
                ))
                
                print(f"[PAIR] Processing latest pair: {latest_pair}")
                self.process_pair(latest_pair)
            else:
                if len(video_files) > 0 or len(landmark_files) > 0:
                    print(f"[WAITING] Video files: {len(video_files)}, Landmark files: {len(landmark_files)}, Waiting for pairs...")
                
        except Exception as e:
            print(f"[ERROR] Error checking pairs: {e}")
    
    def process_pair(self, pair_id):
        """Xử lý 1 cặp {video, landmark}"""
        if pair_id in self.processing_pairs:
            return
            
        self.processing_pairs.add(pair_id)
        
        video_path = self.video_dir / f"{pair_id}.avi"
        landmark_path = self.landmark_dir / f"{pair_id}.npy"
        result_path = self.result_dir / f"{pair_id}.json"
        
        # Kiểm tra file tồn tại
        if not video_path.exists():
            print(f"[ERROR] Video file not found: {video_path}")
            self.processing_pairs.discard(pair_id)
            return
            
        if not landmark_path.exists():
            print(f"[ERROR] Landmark file not found: {landmark_path}")
            self.processing_pairs.discard(pair_id)
            return
        
        # Skip nếu đã có kết quả
        if result_path.exists():
            print(f"[SKIP] Result already exists: {result_path}")
            self.processed_pairs.add(pair_id)
            self.processing_pairs.discard(pair_id)
            return

        print(f"[PROCESSING] Starting inference for pair: {pair_id}")
        
        # try:
        start_time = time.time()
        
        # Load landmarks
        landmarks = np.load(landmark_path)
        print(f"[DATA] Loaded landmarks shape: {landmarks.shape}")
        
        # TODO: Thay bằng model AI thực tế
        result = self.run_inference(video_path, landmarks)
        
        # Thêm metadata
        result.update({
            "processing_time": time.time() - start_time,
            "timestamp": time.time(),
            "video_file": str(video_path.name),
            "landmark_file": str(landmark_path.name),
            "landmark_shape": list(landmarks.shape)
        })
        
        # Lưu kết quả
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"[DONE] Result saved: {result_path}")
        print(f"[DONE] Processing time: {result['processing_time']:.2f}s")
        
        self.processed_pairs.add(pair_id)
            
        # except Exception as e:
        #     print(f"[ERROR] Error processing pair {pair_id}: {e}")
            
        #     # Lưu error result
        #     error_result = {
        #         "id": pair_id,
        #         "status": "error",
        #         "error": str(e),
        #         "timestamp": time.time(),
        #         "video_file": str(video_path.name),
        #         "landmark_file": str(landmark_path.name)
        #     }
            
        #     try:
        #         with open(result_path, 'w', encoding='utf-8') as f:
        #             json.dump(error_result, f, indent=2, ensure_ascii=False)
        #         print(f"[ERROR] Error result saved: {result_path}")
        #     except Exception as save_error:
        #         print(f"[ERROR] Failed to save error result: {save_error}")
                
        # finally:
        #     self.processing_pairs.discard(pair_id)
    
    def run_inference(self, video_path, landmarks):
        """
        Chạy VideoMAE model để dự đoán sign language - chỉ trả về 1 kết quả tốt nhất
        
        Args:
            video_path: Đường dẫn đến video
            landmarks: Numpy array chứa landmarks
            
        Returns:
            dict: Kết quả inference với 1 prediction duy nhất
        """
        print(f"[AI] Running VideoMAE inference on {video_path.name}")
        print(f"[AI] Landmarks shape: {landmarks.shape}")
        
        # try:
        samples = self._prepare_video_data(video_path, landmarks)
        if not samples:
            return {"id": video_path.stem, "status": "error", "error": "No valid samples"}

        dataset = InferenceSampleDataset(samples)
        outputs = self.trainer.predict(dataset)
        logits = outputs.predictions  # shape: [num_clips, num_classes]

        # Get softmax
        probs = torch.softmax(torch.tensor(logits), dim=1)  # [N, C]

        # Option 1: Voting
        voted_idx = torch.mode(torch.argmax(probs, dim=1)).values.item()

        # Option 2: Max confidence
        max_conf_idx = torch.argmax(probs.max(dim=1).values).item()
        best_idx = torch.argmax(probs[max_conf_idx]).item()
        best_class = self.id2label[best_idx]
        best_conf = probs[max_conf_idx][best_idx].item()

        return {
            "id": video_path.stem,
            "status": "completed",
            "prediction": {
                "class": best_class,
                "confidence": float(best_conf),
                "voted_class": self.id2label[voted_idx]
            },
            "num_windows": len(samples)
        }

        # except Exception as e:
        #     print(f"[ERROR] Inference failed: {e}")
        #     return {"id": video_path.stem, "status": "error", "error": str(e)}
        
    def _prepare_video_data(self, video_path, landmarks, num_frames=16, stride=8):
        """Chuẩn bị dữ liệu video cho VideoMAE model"""
        try:
            # Load video frames
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[ERROR] Failed to open video: {video_path}")
                return None
            frames = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (224, 224))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
            if len(frames) < num_frames:
                print(f"[WARNING] Not enough frames for sliding window: {len(frames)}")
                return []
            
            if len(frames) == 0:
                print(f"[ERROR] No frames extracted from {video_path}")
                return None
            
            # Convert to numpy array và normalize
            frames = np.array(frames, dtype=np.float32) / 255.0
            samples = []
            for start in range(0, len(frames) - num_frames + 1, stride):
                video_clip = frames[start:start + num_frames]
                landmark_clip = landmarks[:, start:start + num_frames] if landmarks.ndim == 3 else landmarks[start:start + num_frames]
            # VideoMAE expects specific format - có thể cần điều chỉnh theo dataset format

                video_tensor = torch.tensor(video_clip).permute(3, 0, 1, 2)  # [C, T, H, W]
                landmark_tensor = torch.tensor(landmark_clip)               # [T, D]
                sample = {
                    'video':  video_tensor,
                    'landmark': landmark_tensor,
                    'label': 0  # Dummy label for inference
                }
            
            transformed = self.get_inference_transform()(sample)
            samples.append(transformed)
            
            print(f"[DATA] Prepared video data: frames={sample['video'].shape}, landmarks={sample['landmark'].shape}")
            return sample
            
        except Exception as e:
            print(f"[ERROR] Failed to prepare video data: {e}")
            return None
    @staticmethod
    def get_inference_transform(img_size=(224, 224), mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], num_frames=16):
        return Compose([
            UniformTemporalSubsample(num_frames),
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    Lambda(lambda x: x / 255.0),
                    VideoNormalize(mean, std),
                    Resize(img_size),
                ])
            ),
            ApplyTransformToKey(
                key="landmark",
                transform=Compose([
                    LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),
                    LandmarkUniformTemporalSubsample(num_frames),
                    LandmarkResize(size=img_size, original_size=(256, 256)),
                ])
            ),
        ])
        
    def start(self):
        """Bắt đầu theo dõi"""
        print("[START] Starting Inference Watcher...")
        
        # Xử lý các cặp file có sẵn
        print("[START] Processing existing pairs...")
        self.check_and_process_pairs()
        
        # Bắt đầu theo dõi
        self.video_observer.start()
        self.landmark_observer.start()
        
        print("[WATCH] Watching for new files...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Dừng theo dõi"""
        print("[STOP] Stopping inference watcher...")
        self.video_observer.stop()
        self.landmark_observer.stop()
        self.video_observer.join()
        self.landmark_observer.join()
        print("[STOP] Inference watcher stopped")

class FileHandler(FileSystemEventHandler):
    def __init__(self, watcher, file_type):
        self.watcher = watcher
        self.file_type = file_type
    
    def on_created(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Kiểm tra extension
        if (self.file_type == 'video' and file_path.suffix == '.avi') or \
           (self.file_type == 'landmark' and file_path.suffix == '.npy'):
            
            print(f"[EVENT] New {self.file_type} file: {file_path.name}")
            
            # Đợi file được ghi hoàn toàn
            time.sleep(0.5)
            
            # Kiểm tra file có readable không
            try:
                if self.file_type == 'video':
                    # Kiểm tra video có mở được không
                    import cv2
                    cap = cv2.VideoCapture(str(file_path))
                    if cap.isOpened():
                        cap.release()
                        self.watcher.on_file_created(self.file_type)
                    else:
                        print(f"[WARNING] Cannot open video file: {file_path}")
                        
                elif self.file_type == 'landmark':
                    # Kiểm tra npy file có load được không
                    np.load(str(file_path))
                    self.watcher.on_file_created(self.file_type)
                    
            except Exception as e:
                print(f"[WARNING] File not ready yet: {file_path} - {e}")
                # Có thể retry sau

def start_inference_watcher(video_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_video", 
                           landmark_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_landmark", 
                           result_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/results",
                           model_ckpt_path="/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb_landmark/videomae-base-finetuned-47classes/checkpoint-1324",
                           dataset_root_path="/work/21013187/SAM-SLR-v2/data/person_with_backgrounds"):
    """
    Khởi động inference watcher với VideoMAE model
    
    Args:
        video_dir: Thư mục chứa video files
        landmark_dir: Thư mục chứa landmark files (.npy)
        result_dir: Thư mục lưu kết quả JSON
        model_ckpt_path: Đường dẫn đến checkpoint của VideoMAE model
        dataset_root_path: Đường dẫn đến dataset để load label mapping
    """
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # Kiểm tra model checkpoint tồn tại
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
    
    if not os.path.exists(dataset_root_path):
        raise FileNotFoundError(f"Dataset root path not found: {dataset_root_path}")
    
    print(f"[START] Starting VideoMAE Inference Watcher...")
    print(f"[START] Model checkpoint: {model_ckpt_path}")
    print(f"[START] Dataset root: {dataset_root_path}")
    
    watcher = InferenceWatcher(video_dir, landmark_dir, result_dir, model_ckpt_path, dataset_root_path)
    watcher.start()

if __name__ == "__main__":
    # Cập nhật đường dẫn theo môi trường của bạn
    start_inference_watcher(
        video_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_video",
        landmark_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/data/input_landmark", 
        result_dir="/work/21013187/linh/Vietnam_Signlanguage_FE/Source/results",
        model_ckpt_path="/work/21013187/linh/Vietnam_Signlanguage_FE/Transformer/rgb_landmark/videomae-base-finetuned-47classes/checkpoint-1324",
        dataset_root_path="/work/21013187/SAM-SLR-v2/data/person_with_backgrounds"
    )
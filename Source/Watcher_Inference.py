import os
import json
import time
import uuid
import shutil
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import torch
import cv2

# Import model và utils
from Transformer.rgb_landmark_v2.Model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from Source.data.dataset import get_dataset, collate_fn
from Source.data.utils import get_label_map
from Source.data.utils import video_loader, save_video_as_npy
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Lambda, Resize
from Transformer.rgb_landmark_v2.dataset.video_transforms import (
    ApplyTransformToKey,
    Normalize as VideoNormalize,
    UniformTemporalSubsample,
)
from Transformer.rgb_landmark_v2.dataset.landmark_transforms import (
    Normalize as LandmarkNormalize,
    UniformTemporalSubsample as LandmarkUniformTemporalSubsample,
    Resize as LandmarkResize,
)

class InferenceWatcher:
    def __init__(self, data_dir, result_dir, model_ckpt_path, dataset_root_path):
        self.video_dir = Path(data_dir) / "rgb"
        self.landmark_dir = Path(data_dir) / "npy"
        self.cache_dir = Path(data_dir) / "cache"
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
        
        # Tạo thư mục history
        self.history_dir = Path(data_dir) / "history"
        self.history_rgb_dir = self.history_dir / "rgb"
        self.history_npy_dir = self.history_dir / "npy"
        self.history_cache_dir = self.history_dir / "cache"
        
        # Tạo các thư mục nếu chưa tồn tại
        self.history_dir.mkdir(exist_ok=True)
        self.history_rgb_dir.mkdir(exist_ok=True)
        self.history_npy_dir.mkdir(exist_ok=True)
        self.history_cache_dir.mkdir(exist_ok=True)
        
        self.data_dir = data_dir
        self.processed_pairs = set()
        self.processing_pairs = set()  
        
        # Load model và label mapping
        self.model, self.trainer, self.id2label = self._load_model(model_ckpt_path, dataset_root_path)
        
        # Theo dõi cả 2 thư mục
        self.video_observer = Observer()
        self.landmark_observer = Observer()
        
        video_handler = FileHandler(self, 'video')
        landmark_handler = FileHandler(self, 'landmark')
        
        self.video_observer.schedule(video_handler, str(self.video_dir), recursive=False)
        self.landmark_observer.schedule(landmark_handler, str(self.landmark_dir), recursive=False)
        
        print(f"[INIT] Inference Watcher with VideoMAE model initialized")
        print(f"[INIT] Video dir: {self.video_dir}")
        print(f"[INIT] Landmark dir: {self.landmark_dir}")
        print(f"[INIT] Data dir: {self.data_dir}")
        print(f"[INIT] Result dir: {result_dir}")
        print(f"[INIT] Model loaded with {len(self.id2label)} classes")
    
    def _load_model(self, model_ckpt_path, dataset_root_path):   
        dataset_root_path = Path(dataset_root_path)
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
            per_device_eval_batch_size= 20,  
            dataloader_drop_last=False,
            remove_unused_columns=False,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collate_fn,
        )
        
        return model, trainer, id2label
    
    # def on_file_created(self, file_type):
    #     """Được gọi khi có file mới được tạo"""
    #     print(f"[EVENT] New {file_type} file detected, checking for pairs...")
    #     self.check_and_process_pairs()
    
    # def check_and_process_pairs(self):
    #     """Kiểm tra và xử lý các cặp {video, landmark} hoàn chỉnh - chỉ xử lý 1 cặp mới nhất"""
    #     try:
    #         # Lấy danh sách file (không có extension)
    #         video_files = {f.stem for f in self.cache_dir.glob('*.npy') if f.is_file()}
    #         landmark_files = {f.stem for f in self.landmark_dir.glob('*.npy') if f.is_file()}
            
    #         # Tìm các cặp hoàn chỉnh chưa được xử lý
    #         complete_pairs = video_files & landmark_files
    #         new_pairs = complete_pairs - self.processed_pairs - self.processing_pairs
            
    #         if new_pairs:
    #             for pair_id in new_pairs:
    #                 landmark_path = self.landmark_dir / f"{pair_id}.npy"
    #                 try:
    #                     landmark_data = np.load(landmark_path)
    #                     print(f"[SHAPE CHECK] {pair_id}: {landmark_data.shape}")
    #                 except Exception as e:
    #                     print(f"[ERROR] Could not load landmark {pair_id}: {e}")
    #             # Chỉ xử lý 1 cặp mới nhất (theo thời gian tạo file)
    #             latest_pair = max(new_pairs, key=lambda x: max(
    #                 (self.video_dir / f"{x}.avi").stat().st_mtime,
    #                 (self.landmark_dir / f"{x}.npy").stat().st_mtime
    #             ))
                
    #             print(f"[PAIR] Processing latest pair: {latest_pair}")
    #             self.process_pair(latest_pair)
            
    #     except Exception as e:
    #         print(f"[ERROR] Error checking pairs: {e}")
    
    def generate_rgb_cache(self, video_path: Path):
        try:
            tensor = video_loader(video_path)
            cache_path = self.cache_dir / f"{video_path.stem}.npy"
            save_video_as_npy(tensor, cache_path)
            print(f"[CACHE] Saved RGB tensor to: {cache_path}")
        except Exception as e:
            print(f"[ERROR] Failed to generate RGB tensor: {e}")
    
    def move_processed_files(self, pair_id):
        """Di chuyển các file đã xử lý vào thư mục history"""
        try:
            # Đường dẫn file gốc
            rgb_file = self.video_dir / f"{pair_id}.avi"
            npy_file = self.landmark_dir / f"{pair_id}.npy"
            cache_file = self.cache_dir / f"{pair_id}.npy"
            
            # Đường dẫn đích trong history
            rgb_dest = self.history_rgb_dir / f"{pair_id}.avi"
            npy_dest = self.history_npy_dir / f"{pair_id}.npy"
            cache_dest = self.history_cache_dir / f"{pair_id}.npy"
            
            # Di chuyển file rgb nếu tồn tại
            if rgb_file.exists():
                shutil.move(str(rgb_file), str(rgb_dest))
                print(f"[MOVE] Moved RGB file: {rgb_file} -> {rgb_dest}")
            
            # Di chuyển file npy nếu tồn tại
            if npy_file.exists():
                shutil.move(str(npy_file), str(npy_dest))
                print(f"[MOVE] Moved NPY file: {npy_file} -> {npy_dest}")
            
            # Di chuyển file cache nếu tồn tại
            if cache_file.exists():
                shutil.move(str(cache_file), str(cache_dest))
                print(f"[MOVE] Moved cache file: {cache_file} -> {cache_dest}")
            
            print(f"[MOVE] Successfully moved all files for pair: {pair_id}")
            
        except Exception as e:
            print(f"[ERROR] Failed to move files for pair {pair_id}: {e}")
    
    def process_pair(self, pair_id):
        if pair_id in self.processing_pairs:
            return

        self.processing_pairs.add(pair_id)

        video_path = self.cache_dir / f"{pair_id}.npy"         # RGB tensor
        landmark_path = self.landmark_dir / f"{pair_id}.npy"   # Landmark tensor
        result_path = self.result_dir / f"{pair_id}.json"

        if not video_path.exists():
            print(f"[ERROR] RGB cache not found: {video_path}")
            self.processing_pairs.discard(pair_id)
            return

        if not landmark_path.exists():
            print(f"[ERROR] Landmark file not found: {landmark_path}")
            self.processing_pairs.discard(pair_id)
            return

        if result_path.exists():
            print(f"[SKIP] Result already exists: {result_path}")
            self.processed_pairs.add(pair_id)
            self.processing_pairs.discard(pair_id)
            return

        print(f"[PROCESSING] Starting inference for pair: {pair_id}")
        start_time = time.time()

        landmarks = np.load(landmark_path)

        result = self.run_inference(self.data_dir, video_path)

        result.update({
            "processing_time": time.time() - start_time,
            "timestamp": time.time(),
            "video_tensor_file": str(video_path.name),
            "landmark_file": str(landmark_path.name),
            "landmark_shape": list(landmarks.shape)
        })

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[DONE] Result saved: {result_path}")
        print(f"[DONE] Processing time: {result['processing_time']:.2f}s")

        # Di chuyển các file đã xử lý vào thư mục history
        self.move_processed_files(pair_id)

        self.processed_pairs.add(pair_id)
        self.processing_pairs.discard(pair_id)
    
    def run_inference(self, data_path, video_path):
        """
        Chạy VideoMAE model để dự đoán sign language trên một batch chứa toàn bộ video
        
        Args:
            data_path (str): Đường dẫn đến data file
            
        Returns:
            dict: Kết quả inference tổng hợp từ batch
        """
        try:

            dataset = self._prepare_video_data(data_path)
            outputs = self.trainer.predict(dataset)
            y_pred = np.argmax(outputs.predictions, axis=1)
            label_id = int(y_pred[0])
            label_name = self.id2label[label_id]

            
            # Tổng hợp kết quả bằng voting
            if not y_pred:
                return {
                    "id": video_path.stem,
                    "status": "error",
                    "error": "No predictions generated"
                }
            
            result = {
                "id": video_path.stem,
                "status": "completed",
                "prediction": {
                    "class_id": label_id,
                    "class_name": label_name
                }
            }

            
            return result
        
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return {
                "id": video_path.stem,
                "status": "error",
                "error": str(e)
            }
        
    def _prepare_video_data(self, data_path):
        try:
            dataset = get_dataset(data_path)
            print(f"[DEBUG] Dataset loaded: {type(dataset)}")
            return dataset
        
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
        
        # # Xử lý các cặp file có sẵn
        # print("[START] Processing existing pairs...")
        # self.check_and_process_pairs()
        
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
                        self.watcher.generate_rgb_cache(file_path)
                        # self.watcher.on_file_created(self.file_type)
                    else:
                        print(f"[WARNING] Cannot open video file: {file_path}")
                        
                elif self.file_type == 'landmark':
                    # Kiểm tra npy file có load được không
                    np.load(str(file_path))
                    # self.watcher.on_file_created(self.file_type)
                    pair_id = file_path.stem
                    self.watcher.process_pair(pair_id)
                    
            except Exception as e:
                print(f"[WARNING] File not ready yet: {file_path} - {e}")
                # Có thể retry sau

def start_inference_watcher(data_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data", 
                           result_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/results",
                           model_ckpt_path="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Transformer/rgb_landmark_v2/videomae-base-finetuned_47classes/checkpoint-330",
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
    
    # Kiểm tra model checkpoint tồn tại
    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
    
    if not os.path.exists(dataset_root_path):
        raise FileNotFoundError(f"Dataset root path not found: {dataset_root_path}")
    
    print(f"[START] Starting VideoMAE Inference Watcher...")
    print(f"[START] Model checkpoint: {model_ckpt_path}")
    print(f"[START] Dataset root: {dataset_root_path}")
    
    watcher = InferenceWatcher(data_dir, result_dir, model_ckpt_path, dataset_root_path)
    watcher.start()

if __name__ == "__main__":
    # Cập nhật đường dẫn theo môi trường của bạn
    start_inference_watcher(
        data_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data",
        result_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/results",
        model_ckpt_path="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Transformer/rgb_landmark_v2/videomae-base-finetuned_47classes/checkpoint-330",
        dataset_root_path="/work/21013187/SAM-SLR-v2/data/person_with_backgrounds"
    )
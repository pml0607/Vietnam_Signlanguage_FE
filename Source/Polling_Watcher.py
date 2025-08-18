import os
import time
import threading
import logging
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class PollingWatcher:
    """
    Polling-based file watcher cho SLURM environment
    Thay thế watchdog để tránh filesystem event issues
    """
    
    def __init__(self, landmark_watcher, inference_watcher, poll_interval=3):
        self.landmark_watcher = landmark_watcher
        self.inference_watcher = inference_watcher
        self.poll_interval = poll_interval
        
        # Track processed files
        self.processed_videos = set()
        self.processed_landmarks = set()
        
        # Directories
        self.video_dir = Path(landmark_watcher.input_dir)
        self.landmark_dir = Path(landmark_watcher.output_dir)
        self.cache_dir = Path(inference_watcher.cache_dir)
        
        # Control flags
        self.running = False
        self.threads = []
        
        logger.info(f"Polling Watcher initialized")
        logger.info(f"Video dir: {self.video_dir}")
        logger.info(f"Landmark dir: {self.landmark_dir}")
        logger.info(f"Cache dir: {self.cache_dir}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        
    def start(self):
        """Start polling system"""
        logger.info("=== Starting Polling Watcher System ===")
        
        self.running = True
        
        # Scan existing files first
        self._scan_existing_files()
        
        # Start polling threads
        video_thread = threading.Thread(target=self._poll_videos, name="VideoPoller", daemon=True)
        landmark_thread = threading.Thread(target=self._poll_landmarks, name="LandmarkPoller", daemon=True)
        
        self.threads.extend([video_thread, landmark_thread])
        
        video_thread.start()
        landmark_thread.start()
        
        logger.info("Polling threads started")
        logger.info("Press Ctrl+C to stop...")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(5)
                self._print_status()
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal...")
        finally:
            self.stop()
            
    def stop(self):
        """Stop polling system"""
        logger.info("Stopping polling watcher...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                logger.info(f"Waiting for {thread.name} to finish...")
                thread.join(timeout=5)
                
        logger.info("Polling watcher stopped")
        
    def _scan_existing_files(self):
        """Scan existing files to avoid reprocessing"""
        logger.info("Scanning existing files...")
        
        # Scan videos
        if self.video_dir.exists():
            for video_file in self.video_dir.glob("*.avi"):
                self.processed_videos.add(video_file.stem)
                
        # Scan landmarks
        if self.landmark_dir.exists():
            for landmark_file in self.landmark_dir.glob("*.npy"):
                self.processed_landmarks.add(landmark_file.stem)
                
        logger.info(f"Found {len(self.processed_videos)} existing videos")
        logger.info(f"Found {len(self.processed_landmarks)} existing landmarks")
        
    def _poll_videos(self):
        """Poll for new video files"""
        logger.info("[VIDEO POLLER] Started")
        
        while self.running:
            try:
                if not self.video_dir.exists():
                    time.sleep(self.poll_interval)
                    continue
                    
                # Get current video files
                current_videos = {f.stem for f in self.video_dir.glob("*.avi") if f.is_file()}
                new_videos = current_videos - self.processed_videos
                
                if new_videos:
                    logger.info(f"[VIDEO POLLER] Found {len(new_videos)} new videos: {list(new_videos)}")
                    
                    for video_id in new_videos:
                        video_path = self.video_dir / f"{video_id}.avi"
                        
                        logger.info(f"[VIDEO POLLER] Processing: {video_id}")
                        
                        # Check if file is complete
                        if self._wait_for_file_complete(video_path, file_type='video'):
                            try:
                                # Process with landmark watcher
                                self.landmark_watcher.process_video(str(video_path))
                                self.processed_videos.add(video_id)
                                logger.info(f"[VIDEO POLLER] ✅ Completed: {video_id}")
                                
                            except Exception as e:
                                logger.error(f"[VIDEO POLLER] ❌ Error processing {video_id}: {e}")
                        else:
                            logger.warning(f"[VIDEO POLLER] ⚠️ File not ready: {video_id}")
                            
            except Exception as e:
                logger.error(f"[VIDEO POLLER] Error in polling loop: {e}")
                
            time.sleep(self.poll_interval)
            
        logger.info("[VIDEO POLLER] Stopped")
        
    def _poll_landmarks(self):
        """Poll for new landmark files"""
        logger.info("[LANDMARK POLLER] Started")
        
        while self.running:
            try:
                if not self.landmark_dir.exists():
                    time.sleep(self.poll_interval)
                    continue
                    
                # Get current landmark files
                current_landmarks = {f.stem for f in self.landmark_dir.glob("*.npy") if f.is_file()}
                new_landmarks = current_landmarks - self.processed_landmarks
                
                if new_landmarks:
                    logger.info(f"[LANDMARK POLLER] Found {len(new_landmarks)} new landmarks: {list(new_landmarks)}")
                    
                    for landmark_id in new_landmarks:
                        landmark_path = self.landmark_dir / f"{landmark_id}.npy"
                        
                        logger.info(f"[LANDMARK POLLER] Processing: {landmark_id}")
                        
                        # Check if file is complete
                        if self._wait_for_file_complete(landmark_path, file_type='landmark'):
                            try:
                                # Generate RGB cache first
                                video_path = self.video_dir / f"{landmark_id}.avi"
                                if video_path.exists():
                                    logger.info(f"[LANDMARK POLLER] Generating RGB cache for: {landmark_id}")
                                    self.inference_watcher.generate_rgb_cache(video_path)
                                    
                                    # Process inference
                                    logger.info(f"[LANDMARK POLLER] Running inference for: {landmark_id}")
                                    self.inference_watcher.process_pair(landmark_id)
                                    
                                    self.processed_landmarks.add(landmark_id)
                                    logger.info(f"[LANDMARK POLLER] ✅ Completed: {landmark_id}")
                                else:
                                    logger.warning(f"[LANDMARK POLLER] Video file not found for: {landmark_id}")
                                    
                            except Exception as e:
                                logger.error(f"[LANDMARK POLLER] ❌ Error processing {landmark_id}: {e}")
                        else:
                            logger.warning(f"[LANDMARK POLLER] ⚠️ File not ready: {landmark_id}")
                            
            except Exception as e:
                logger.error(f"[LANDMARK POLLER] Error in polling loop: {e}")
                
            time.sleep(self.poll_interval)
            
        logger.info("[LANDMARK POLLER] Stopped")
        
    def _wait_for_file_complete(self, file_path, file_type='unknown', stable_time=2, max_wait=30):
        """Wait for file to be completely written"""
        logger.debug(f"[FILE CHECK] Waiting for {file_type}: {file_path.name}")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                if not file_path.exists():
                    logger.debug(f"[FILE CHECK] File disappeared: {file_path.name}")
                    return False
                    
                # Check file size stability
                initial_size = file_path.stat().st_size
                time.sleep(stable_time)
                
                if not file_path.exists():
                    logger.debug(f"[FILE CHECK] File disappeared during check: {file_path.name}")
                    return False
                    
                final_size = file_path.stat().st_size
                
                if initial_size == final_size and final_size > 0:
                    # Additional validation based on file type
                    if file_type == 'video':
                        if self._validate_video_file(file_path):
                            logger.debug(f"[FILE CHECK] ✅ Video ready: {file_path.name} ({final_size} bytes)")
                            return True
                        else:
                            logger.debug(f"[FILE CHECK] Video validation failed: {file_path.name}")
                            
                    elif file_type == 'landmark':
                        if self._validate_landmark_file(file_path):
                            logger.debug(f"[FILE CHECK] ✅ Landmark ready: {file_path.name} ({final_size} bytes)")
                            return True
                        else:
                            logger.debug(f"[FILE CHECK] Landmark validation failed: {file_path.name}")
                    else:
                        # Generic file
                        logger.debug(f"[FILE CHECK] ✅ File ready: {file_path.name} ({final_size} bytes)")
                        return True
                        
                logger.debug(f"[FILE CHECK] File still changing: {file_path.name} ({initial_size} -> {final_size})")
                
            except Exception as e:
                logger.debug(f"[FILE CHECK] Error checking {file_path.name}: {e}")
                
            time.sleep(1)
        
        logger.warning(f"[FILE CHECK] ⏰ Timeout waiting for: {file_path.name}")
        return False
        
    def _validate_video_file(self, video_path):
        """Validate video file can be opened"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception:
            return False
            
    def _validate_landmark_file(self, landmark_path):
        """Validate landmark file can be loaded"""
        try:
            data = np.load(str(landmark_path))
            return len(data) > 0
        except Exception:
            return False
            
    def _print_status(self):
        """Print current status"""
        logger.info(f"[STATUS] Processed - Videos: {len(self.processed_videos)}, Landmarks: {len(self.processed_landmarks)}")
        
        # Check thread health
        for thread in self.threads:
            status = "🟢 Running" if thread.is_alive() else "🔴 Stopped"
            logger.info(f"[STATUS] {thread.name}: {status}")

# Convenience function to start polling watcher
def start_polling_watcher(landmark_watcher, inference_watcher, poll_interval=3):
    """
    Start polling watcher system
    
    Args:
        landmark_watcher: LandmarkWatcher instance
        inference_watcher: InferenceWatcher instance  
        poll_interval: Polling interval in seconds
    """
    watcher = PollingWatcher(landmark_watcher, inference_watcher, poll_interval)
    watcher.start()
    return watcher
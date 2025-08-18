#!/usr/bin/env python3
"""
SLURM-compatible watcher script using polling instead of filesystem events
"""

import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def run_combined_polling_system():
    """Run both watchers using unified polling system"""
    
    logger.info("=== Starting Combined Polling Watcher System ===")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    try:
        # Import watcher classes
        logger.info("Importing watcher modules...")
        from Watcher_Landmark import LandmarkWatcher
        from Watcher_Inference import InferenceWatcher
        
        # Initialize landmark watcher (for GPU processing)
        logger.info("Initializing Landmark Watcher...")
        landmark_watcher = LandmarkWatcher(
            input_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data/rgb",
            output_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data/npy",
            config_path="/work/21013187/phuoc/visl-i3d/src/config/wholebody_w48_384x288.yaml",
            checkpoint_path="/work/21013187/phuoc/visl-i3d/src/checkpoint/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
        )
        
        # Initialize inference watcher (for GPU processing)  
        logger.info("Initializing Inference Watcher...")
        inference_watcher = InferenceWatcher(
            data_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data",
            result_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/results",
            model_ckpt_path="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Transformer/rgb_landmark_v2/videomae-base-finetuned_47classes/checkpoint-330",
            dataset_root_path="/work/21013187/SAM-SLR-v2/data/person_with_backgrounds"
        )
        
        logger.info("Both watchers initialized successfully")
        
        # Start proper multi-threaded polling system
        logger.info("Starting multi-threaded polling system...")
        
        import threading
        
        # Thread 1: LandmarkWatcher polls video directory
        landmark_thread = threading.Thread(
            target=poll_video_directory_for_landmarks,
            args=(landmark_watcher,),
            name="LandmarkGeneration",
            daemon=True
        )
        
        # Thread 2: InferenceWatcher polls video directory for cache generation
        cache_thread = threading.Thread(
            target=poll_video_directory_for_cache,
            args=(inference_watcher,),
            name="CacheGeneration", 
            daemon=True
        )
        
        # Thread 3: InferenceWatcher polls landmark directory for inference
        inference_thread = threading.Thread(
            target=poll_landmark_directory_for_inference,
            args=(inference_watcher,),
            name="InferenceExecution",
            daemon=True
        )
        
        # Start all threads
        landmark_thread.start()
        cache_thread.start() 
        inference_thread.start()
        
        logger.info("All polling threads started")
        logger.info("Press Ctrl+C to stop...")
        
        # Keep main thread alive and monitor
        try:
            while True:
                time.sleep(10)
                
                # Check thread health
                threads_status = {
                    "LandmarkGeneration": landmark_thread.is_alive(),
                    "CacheGeneration": cache_thread.is_alive(), 
                    "InferenceExecution": inference_thread.is_alive()
                }
                
                for name, alive in threads_status.items():
                    status = "🟢 Running" if alive else "🔴 Stopped"
                    logger.info(f"[MONITOR] {name}: {status}")
                
                # Break if any critical thread died
                if not all(threads_status.values()):
                    logger.error("One or more threads died!")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal...")
            
    except Exception as e:
        logger.error(f"Failed to start combined polling system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_single_watcher():
    """Run single watcher with polling based on environment variable"""
    watcher_type = os.environ.get('WATCHER_TYPE', '').lower()
    
    logger.info(f"=== Starting Single Watcher: {watcher_type.upper()} ===")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if watcher_type == 'landmark':
        run_landmark_polling()
    elif watcher_type == 'inference':
        run_inference_polling()
    else:
        logger.error(f"Unknown watcher type: {watcher_type}")
        logger.error("Valid types: landmark, inference")
        sys.exit(1)

def run_landmark_polling():
    """Run landmark watcher with polling"""
    logger.info("Starting LANDMARK watcher with polling...")
    
    try:
        from Watcher_Landmark import LandmarkWatcher
        
        # Initialize watcher - KHÔNG start filesystem watching
        watcher = LandmarkWatcher(
            input_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data/rgb",
            output_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data/npy",
            config_path="/work/21013187/phuoc/visl-i3d/src/config/wholebody_w48_384x288.yaml",
            checkpoint_path="/work/21013187/phuoc/visl-i3d/src/checkpoint/hrnet_w48_coco_wholebody_384x288-6e061c6a_20200922.pth"
        )
        
        logger.info("Landmark watcher initialized successfully")
        
        # Start polling THAY VÌ filesystem watching
        poll_video_directory_for_landmarks(watcher)
        
    except Exception as e:
        logger.error(f"Landmark watcher failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_inference_polling():
    """Run inference watcher with polling - watches BOTH video and landmark directories"""
    logger.info("Starting INFERENCE watcher with dual directory polling...")
    
    try:
        from Watcher_Inference import InferenceWatcher
        
        # Initialize watcher
        watcher = InferenceWatcher(
            data_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/data",
            result_dir="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Source/results",
            model_ckpt_path="/work/huong.nguyenthi2/Vietnam_Signlanguage_FE/Transformer/rgb_landmark_v2/videomae-base-finetuned_47classes/checkpoint-330",
            dataset_root_path="/work/21013187/SAM-SLR-v2/data/person_with_backgrounds"
        )
        
        logger.info("Inference watcher initialized successfully")
        
        # Start BOTH polling threads for InferenceWatcher
        import threading
        
        # Thread 1: Poll video directory for cache generation
        cache_thread = threading.Thread(
            target=poll_video_directory_for_cache,
            args=(watcher,),
            name="VideoCachePolling",
            daemon=True
        )
        
        # Thread 2: Poll landmark directory for inference execution
        inference_thread = threading.Thread(
            target=poll_landmark_directory_for_inference,
            args=(watcher,),
            name="LandmarkInferencePolling",
            daemon=True
        )
        
        # Start both threads
        cache_thread.start()
        inference_thread.start()
        
        logger.info("Both InferenceWatcher polling threads started")
        logger.info("- Video directory polling (for cache generation)")
        logger.info("- Landmark directory polling (for inference execution)")
        
        # Keep main thread alive and monitor
        try:
            while True:
                time.sleep(10)
                
                # Check thread health
                cache_alive = cache_thread.is_alive()
                inference_alive = inference_thread.is_alive()
                logger.info(f"[MONITOR] VideoCachePolling: {'🟢 Running' if cache_alive else '🔴 Stopped'}")
                logger.info(f"[MONITOR] LandmarkInferencePolling: {'🟢 Running' if inference_alive else '🔴 Stopped'}")
                
                # Break if any thread died
                if not cache_alive or not inference_alive:
                    logger.error("One or more inference polling threads died!")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Stopping inference polling...")
        
    except Exception as e:
        logger.error(f"Inference watcher failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def poll_video_directory_for_landmarks(landmark_watcher, poll_interval=3):
    """Poll video directory for landmark generation"""
    video_dir = Path(landmark_watcher.input_dir)
    processed_videos = set()
    
    logger.info(f"[LANDMARK GEN] Watching video dir: {video_dir}")
    
    # Scan existing files
    if video_dir.exists():
        for video_file in video_dir.glob("*.avi"):
            processed_videos.add(video_file.stem)
    
    logger.info(f"[LANDMARK GEN] Found {len(processed_videos)} existing videos")
    
    while True:
        try:
            if not video_dir.exists():
                time.sleep(poll_interval)
                continue
                
            # Get current video files
            current_videos = {f.stem for f in video_dir.glob("*.avi") if f.is_file()}
            new_videos = current_videos - processed_videos
            
            if new_videos:
                logger.info(f"[LANDMARK GEN] Found {len(new_videos)} new videos: {list(new_videos)}")
                
                for video_id in new_videos:
                    video_path = video_dir / f"{video_id}.avi"
                    
                    logger.info(f"[LANDMARK GEN] Processing: {video_id}")
                    
                    if wait_for_file_complete(video_path, 'video'):
                        try:
                            landmark_watcher.process_video(str(video_path))
                            processed_videos.add(video_id)
                            logger.info(f"[LANDMARK GEN] ✅ Landmarks generated: {video_id}")
                        except Exception as e:
                            logger.error(f"[LANDMARK GEN] ❌ Failed {video_id}: {e}")
                    else:
                        logger.warning(f"[LANDMARK GEN] ⚠️ Video not ready: {video_id}")
            
            time.sleep(poll_interval)
            
        except Exception as e:
            logger.error(f"[LANDMARK GEN] Error: {e}")
            time.sleep(poll_interval)

def poll_video_directory_for_cache(inference_watcher, poll_interval=3):
    """Poll video directory for RGB cache generation"""
    video_dir = Path(inference_watcher.video_dir)
    cache_dir = Path(inference_watcher.cache_dir)
    processed_cache = set()
    
    logger.info(f"[CACHE GEN] Watching video dir: {video_dir}")
    logger.info(f"[CACHE GEN] Cache dir: {cache_dir}")
    
    # Ensure cache directory exists
    cache_dir.mkdir(exist_ok=True)
    
    # Scan existing cache files
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.npy"):
            processed_cache.add(cache_file.stem)
    
    logger.info(f"[CACHE GEN] Found {len(processed_cache)} existing cache files")
    
    while True:
        try:
            if not video_dir.exists():
                time.sleep(poll_interval)
                continue
                
            # Get current video files
            current_videos = {f.stem for f in video_dir.glob("*.avi") if f.is_file()}
            uncached_videos = current_videos - processed_cache
            
            if uncached_videos:
                logger.info(f"[CACHE GEN] Found {len(uncached_videos)} uncached videos: {list(uncached_videos)}")
                
                for video_id in uncached_videos:
                    video_path = video_dir / f"{video_id}.avi"
                    
                    logger.info(f"[CACHE GEN] Processing: {video_id}")
                    
                    if wait_for_file_complete(video_path, 'video'):
                        try:
                            inference_watcher.generate_rgb_cache(video_path)
                            processed_cache.add(video_id)
                            logger.info(f"[CACHE GEN] ✅ RGB cache generated: {video_id}")
                        except Exception as e:
                            logger.error(f"[CACHE GEN] ❌ Failed {video_id}: {e}")
                    else:
                        logger.warning(f"[CACHE GEN] ⚠️ Video not ready: {video_id}")
            
            time.sleep(poll_interval)
            
        except Exception as e:
            logger.error(f"[CACHE GEN] Error: {e}")
            time.sleep(poll_interval)

def poll_landmark_directory_for_inference(inference_watcher, poll_interval=3):
    """Poll landmark directory for inference execution"""
    landmark_dir = Path(inference_watcher.landmark_dir)
    cache_dir = Path(inference_watcher.cache_dir)
    result_dir = Path(inference_watcher.result_dir)
    processed_inference = set()
    
    logger.info(f"[INFERENCE] Watching landmark dir: {landmark_dir}")
    logger.info(f"[INFERENCE] Cache dir: {cache_dir}")
    logger.info(f"[INFERENCE] Result dir: {result_dir}")
    
    # Scan existing result files
    if result_dir.exists():
        for result_file in result_dir.glob("*.json"):
            processed_inference.add(result_file.stem)
    
    logger.info(f"[INFERENCE] Found {len(processed_inference)} existing results")
    
    while True:
        try:
            if not landmark_dir.exists():
                time.sleep(poll_interval)
                continue
                
            # Get current landmark files
            current_landmarks = {f.stem for f in landmark_dir.glob("*.npy") if f.is_file()}
            new_landmarks = current_landmarks - processed_inference
            
            if new_landmarks:
                logger.info(f"[INFERENCE] Found {len(new_landmarks)} new landmarks: {list(new_landmarks)}")
                
                for landmark_id in new_landmarks:
                    landmark_path = landmark_dir / f"{landmark_id}.npy"
                    cache_path = cache_dir / f"{landmark_id}.npy"
                    
                    logger.info(f"[INFERENCE] Checking pair: {landmark_id}")
                    
                    # Check if both landmark and cache files exist and are ready
                    if (wait_for_file_complete(landmark_path, 'landmark') and 
                        cache_path.exists() and 
                        wait_for_file_complete(cache_path, 'cache')):
                        
                        logger.info(f"[INFERENCE] Processing pair: {landmark_id}")
                        
                        try:
                            inference_watcher.process_pair(landmark_id)
                            processed_inference.add(landmark_id)
                            logger.info(f"[INFERENCE] ✅ Inference completed: {landmark_id}")
                        except Exception as e:
                            logger.error(f"[INFERENCE] ❌ Failed {landmark_id}: {e}")
                    else:
                        logger.info(f"[INFERENCE] ⏳ Waiting for cache: {landmark_id}")
            
            time.sleep(poll_interval)
            
        except Exception as e:
            logger.error(f"[INFERENCE] Error: {e}")
            time.sleep(poll_interval)

def wait_for_file_complete(file_path, file_type='unknown', stable_time=2, max_wait=30):
    """Wait for file to be completely written"""
    import cv2
    import numpy as np
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            if not file_path.exists():
                return False
                
            initial_size = file_path.stat().st_size
            time.sleep(stable_time)
            
            if not file_path.exists():
                return False
                
            final_size = file_path.stat().st_size
            
            if initial_size == final_size and final_size > 0:
                # Additional validation based on file type
                if file_type == 'video':
                    cap = cv2.VideoCapture(str(file_path))
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        return ret and frame is not None
                elif file_type in ['landmark', 'cache']:
                    try:
                        data = np.load(str(file_path))
                        return len(data) > 0
                    except:
                        return False
                        
                return True
                
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")
            
        time.sleep(1)
    
    return False

def main():
    logger.info("=== SLURM Polling Watcher Starting ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check if running single watcher or combined system
    watcher_type = os.environ.get('WATCHER_TYPE', '').lower()
    
    if watcher_type in ['landmark', 'inference']:
        logger.info(f"Running single watcher mode: {watcher_type}")
        run_single_watcher()
    else:
        logger.info("Running combined watcher mode")
        run_combined_polling_system()

if __name__ == "__main__":
    main()
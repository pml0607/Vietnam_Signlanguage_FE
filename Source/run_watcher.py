import os
from Watcher_Landmark import start_landmark_watcher
from Watcher_Inference import start_inference_watcher

def main():
    watcher_type = os.environ.get("WATCHER_TYPE", "")
    print(f"[INFO] Starting {watcher_type}...")

    if watcher_type == "landmark":
        start_landmark_watcher()
    elif watcher_type == "inference":
        start_inference_watcher()
    else:
        print("[ERROR] WATCHER_TYPE not set or invalid.")

if __name__ == "__main__":
    main()

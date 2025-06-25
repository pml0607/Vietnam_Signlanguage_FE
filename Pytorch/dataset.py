import os
import torch
from torch.utils.data import Dataset

class PreprocessedClipDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir: thư mục chứa các file .pt (ví dụ: 'preprocessed_clips/train')
        """
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        clip = data['clip']       # Tensor: (3, 64, 224, 224)
        label = data['label']     # int
        video_id = data['video_id']
        clip_idx = data['clip_idx']
        return clip, label, video_id, clip_idx

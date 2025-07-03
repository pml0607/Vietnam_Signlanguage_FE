import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as Fu
import torchvision.transforms as A
import random
from PIL import Image

class SingleStreamAugmentation:
    def __init__(self):
        self.rgb_aug = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.rotation_range = 10  # degrees
        self.translation_px = 50

    def __call__(self, clip):
        """
        clip: Tensor (3, T, H, W) – RGB video clip
        return: Tensor (3, T, H, W) – augmented RGB video
        """
        C, T, H, W = clip.shape
        assert C == 3, "output clip must have 3 channels (RGB)"

        aug_frames = []

        do_flip = random.random() < 0.2
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        dx = random.randint(-self.translation_px, self.translation_px)
        dy = random.randint(-self.translation_px, self.translation_px)

        for t in range(T):
            rgb_frame = Fu.to_pil_image(clip[:, t])

            # Color jitter + affine (rotation + translate)
            rgb_frame = self.rgb_aug(rgb_frame)
            rgb_frame = Fu.affine(rgb_frame, angle=angle, translate=(dx, dy), scale=1.0, shear=0)

            if do_flip:
                rgb_frame = Fu.hflip(rgb_frame)

            aug_frames.append(Fu.to_tensor(rgb_frame))

        rgb_clip_aug = torch.stack(aug_frames, dim=1)
        return rgb_clip_aug


class DualStreamAugmentation:
    def __init__(self):
        self.rgb_aug = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        self.rotation_range = 10  # degrees
        self.translation_px = 50

    def __call__(self, clip):
        rgb_clip = clip[:3]
        skel_clip = clip[3:]
        _, T, _, _ = clip.shape

        aug_rgb = []
        aug_skel = []

        do_flip = random.random() < 0.2
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        dx = random.randint(-self.translation_px, self.translation_px)
        dy = random.randint(-self.translation_px, self.translation_px)

        for t in range(T):
            rgb_frame = Fu.to_pil_image(rgb_clip[:, t])
            skel_frame = Fu.to_pil_image(skel_clip[:, t])

            # Apply jitter + affine (rotation + translate)
            rgb_frame = self.rgb_aug(rgb_frame)
            rgb_frame = Fu.affine(rgb_frame, angle=angle, translate=(dx, dy), scale=1.0, shear=0)

            skel_frame = Fu.affine(skel_frame, angle=angle, translate=(dx, dy), scale=1.0, shear=0)

            if do_flip:
                rgb_frame = Fu.hflip(rgb_frame)
                skel_frame = Fu.hflip(skel_frame)

            aug_rgb.append(Fu.to_tensor(rgb_frame))
            aug_skel.append(Fu.to_tensor(skel_frame))

        rgb_clip_aug = torch.stack(aug_rgb, dim=1)
        skel_clip_aug = torch.stack(aug_skel, dim=1)
        return torch.cat([rgb_clip_aug, skel_clip_aug], dim=0)



class PreprocessedClipDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        """
        root_dir: Directory containing preprocessed video clips.
        """
        self.files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.pt')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        clip = data['clip']       # Tensor: (3, 64, 224, 224)
        label = data['label']     # int
        video_id = data['video_id']
        clip_idx = data['clip_idx']
        if self.transform:
            clip = self.transform(clip)
        else:
            clip = clip.float() / 255.0 
        return clip.float(), label, video_id, clip_idx

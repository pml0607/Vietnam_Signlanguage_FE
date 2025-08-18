import os
import torch

from Transformer.rgb_landmark_v2.dataset.video_transforms import (
    ApplyTransformToKey,
    Normalize as VideoNormalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    RandomHorizontalFlip as VideoRandomHorizontalFlip,
    RemoveKey
)

from Transformer.rgb_landmark_v2.dataset.landmark_transforms import (
    Normalize as LandmarkNormalize,
    RandomHorizontalFlip as LandmarkRandomHorizontalFlip,
    UniformTemporalSubsample as LandmarkUniformTemporalSubsample,
    RandomCrop as LandmarkRandomCrop,
    Resize as LandmarkResize
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    Resize,
)

import pytorchvideo
from Source.data.utils import unlabeled_video_dataset

def get_dataset(dataset_root_path,
                img_size = (224, 224),
                mean = [0.5,0.5,0.5],
                std = [0.5,0.5,0.5],
                num_frames = 16):    
    sample_rate = 4
    fps = 30
    clip_duration = num_frames * sample_rate / fps


    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
            UniformTemporalSubsample(num_frames),
            
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Lambda(lambda x: x / 255.0),
                        VideoNormalize(mean, std),
                        Resize(img_size),
                    ]
                ),
            ),
            
            # Landmark transforms for validation
            ApplyTransformToKey(
                key = 'landmark',
                transform=Compose(
                    [
                        LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),
                        LandmarkUniformTemporalSubsample(num_frames),
                        LandmarkResize(size=img_size, original_size=(256, 256)),
                    ]
                ),
            ),
        ]
    )

    test_dataset = unlabeled_video_dataset(
        data_path=os.path.join(dataset_root_path, 'rgb'),
        transform=val_transform,
    )
    
    return test_dataset

def collate_fn(examples):
    """The collation function to be used by `Trainer` to prepare data batches."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    
    # Stack landmarks - shape should be (B, T, N, C) where:
    # B = batch_size, T = num_frames, N = num_keypoints (133), C = coordinates (3)
    landmarks = torch.stack(
        [example["landmark"] for example in examples]
    )
    
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    
    return {
        "pixel_values": pixel_values, 
        "labels": labels,
        "landmarks": landmarks
    }
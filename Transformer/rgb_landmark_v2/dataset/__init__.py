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
from Transformer.rgb_landmark_v2.dataset.utils import labeled_video_dataset

def get_dataset(dataset_root_path,
                img_size = (224, 224),
                mean = [0.5,0.5,0.5],
                std = [0.5,0.5,0.5],
                num_frames = 16):

    video_count_train = len(list(dataset_root_path.glob("train/rgb/*.avi")))
    video_count_val = len(list(dataset_root_path.glob("val/rgb/*.avi")))
    video_count_test = len(list(dataset_root_path.glob("test/rgb/*.avi")))
    video_total = video_count_train + video_count_val + video_count_test

    print(f"Total videos: {video_total}")
    
    sample_rate = 4
    fps = 30
    clip_duration = num_frames * sample_rate / fps

    # Training dataset transformations.
    train_transform = Compose(
        [
            # Remove unused keys first
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
            # Apply temporal subsampling to both video and landmarks
            UniformTemporalSubsample(num_frames),
            
            # Video transforms
            ApplyTransformToKey(
                key = 'video',
                transform=Compose(
                    [
                        Lambda(lambda x: x / 255.0),
                        VideoNormalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(img_size),
                        VideoRandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),

            # Landmark transforms - đồng bộ với video transforms
            ApplyTransformToKey(
                key = 'landmark',
                transform=Compose(
                    [
                        # Normalize landmarks to a reasonable range
                        LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),  
                        # Temporal subsampling should match video
                        LandmarkUniformTemporalSubsample(num_frames),  
                        # Crop landmarks to match video crop
                        LandmarkRandomCrop(output_size=img_size, original_size=(256, 256)),  
                        # Flip landmarks to match video flip
                        LandmarkRandomHorizontalFlip(p=0.5),  
                    ]
                ),
            ),
        ]
    )
    
    # Training dataset.
    train_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "train",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

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

    # Validation and evaluation datasets.
    val_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "val",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )

    test_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "test",'rgb'),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    
    return train_dataset,val_dataset,test_dataset

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
import os
import torch

from dataset.video_transforms import (
    ApplyTransformToKey,
    Normalize as VideoNormalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
    RandomHorizontalFlip as VideoRandomHorizontalFlip,
    RemoveKey
)

from dataset.landmark_transforms import (
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
from dataset.utils import labeled_video_dataset

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
    
    # chua biet cai nay lam gi :)) nhung dung xoa
    sample_rate = 4
    fps = 30
    clip_duration = num_frames * sample_rate / fps

    # -- NOTE -- Tung: transfrom tren vid -> transform tren landmark tuong tu k (cung gia tri so voi vid)
    
    # Training dataset transformations.
    train_transform = Compose(
        [
            # apply all the key
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
            UniformTemporalSubsample(num_frames),
            # apply just only key focus
            #remove unuse key
            RemoveKey(['video_index','clip_index','aug_index','video_name']),
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

            ApplyTransformToKey(
                key = 'landmark',
                transform=Compose(
                    [
                        LandmarkNormalize(mean=(0, 0, 0), std=(1, 1, 1)),  
                        LandmarkUniformTemporalSubsample(num_frames),  
                        LandmarkRandomCrop(output_size=img_size, original_size=(256, 256)),  
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
            # Add transform for landmark - validation set
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
    landmarks = torch.stack(
        [example["landmark"] for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples],dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels,"landmarks":landmarks}

def collate_fn_test(examples):
    """Safe collation function for test mode."""
    batch = {}

    if "video" in examples[0]:
        batch["pixel_values"] = torch.stack([
            example["video"].permute(1, 0, 2, 3) for example in examples
        ])

    if "landmark" in examples[0]:
        batch["landmarks"] = torch.stack([
            example["landmark"] for example in examples
        ])

    batch["labels"] = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return batch
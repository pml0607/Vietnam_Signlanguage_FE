import torch

def clip_collate_fn(batch):
    """
    batch: list of samples, mỗi sample là tuple (clip, label, video_id, clip_idx)
    Returns:
        clips: (B, 3, 64, 224, 224)
        labels: (B,)
        video_ids: list[str]
        clip_idxs: list[int]
    """
    clips = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    video_ids = [item[2] for item in batch]
    clip_idxs = [item[3] for item in batch]
    return clips, labels, video_ids, clip_idxs
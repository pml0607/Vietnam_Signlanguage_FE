
from pathlib import Path
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union, Type
from pytorchvideo.data.utils import MultiProcessSampler
from glob import glob
from torch.utils.data import Dataset
import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler
import numpy as np


def get_label_map(dataset_root_path,
                  dataset_type: str = 'rgb') -> Tuple[Dict[str, int], Dict[int, str]]:
	if not isinstance(dataset_root_path,Path):
		dataset_root_path = Path(dataset_root_path)

	number_classes = max(find_number_classes(os.path.join(dataset_root_path,phase,dataset_type)) for phase in ['train','val','test'])

	# number_classes = 122
	label2id = {label: int(i) for  i,label in enumerate(range(number_classes))}
	id2label = {i: label for label, i in label2id.items()}
	
	#print(f"Unique classes: {list(label2id.keys())}.")
	print("Label 2 ID:\n",'\n'.join([f'{k}: {v}' for k, v in label2id.items()]))
	return label2id,id2label


def get_label_map(dataset_root_path, dataset_type: str = 'rgb') -> Tuple[Dict[int, int], Dict[int, int]]:
    if not isinstance(dataset_root_path, Path):
        dataset_root_path = Path(dataset_root_path)

    class_ids = set()

    for phase in ['train', 'val', 'test']:
        dir_path = dataset_root_path / phase / dataset_type
        for file in dir_path.glob("*.avi"):
            try:
                class_id = int(file.stem.split('P')[0].split('A')[1])
                class_ids.add(class_id)
            except Exception as e:
                print(f"⚠️ Bỏ qua file không hợp lệ: {file.name}")

    class_ids = sorted(class_ids) 

    label2id = {cls: i for i, cls in enumerate(class_ids)}
    id2label = {i: cls for cls, i in label2id.items()}

    print("Đã tạo label2id:")
    for k, v in label2id.items():
        print(f"  Class {k} → ID {v}")

    return label2id, id2label



class LabeledVideoPaths:


	@classmethod
	def from_directory(cls, dir_path: str, class_to_idx: Optional[Dict[int, int]] = None):
		video_paths_and_label = make_dataset(
			dir_path, class_to_idx=class_to_idx, extensions=("mp4", "avi")
		)
		assert len(video_paths_and_label) > 0, f"Failed to load dataset from {dir_path}."
		return cls(video_paths_and_label)

	def __init__(
		self, paths_and_labels: List[Tuple[str, Optional[int]]], path_prefix=""
	) -> None:
		self._paths_and_labels = paths_and_labels
		self._path_prefix = path_prefix

	def path_prefix(self, prefix):
		self._path_prefix = prefix

	path_prefix = property(None, path_prefix)

	def __getitem__(self, index: int) -> Tuple[str, int]:
		path, label = self._paths_and_labels[index]
		return (os.path.join(self._path_prefix, path), {"label": label})

	def __len__(self) -> int:
		return len(self._paths_and_labels)

class UnlabeledVideoDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.video_paths = sorted(
            glob(os.path.join(data_path, "*.avi")) + glob(os.path.join(data_path, "*.mp4"))
        )
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        video_tensor = video_loader(video_path)

        # Tìm landmark tương ứng
        root_dir = os.path.dirname(os.path.dirname(video_path))  # ../data/
        npy_path = os.path.join(root_dir, "npy", str(Path(os.path.basename(video_path)).stem) + ".npy")

        if os.path.exists(npy_path):
            landmark = torch.tensor(np.load(npy_path)).float()
        else:
            print(f"[WARNING] Missing landmark file: {npy_path}")
            landmark = torch.zeros((1, 1))  # dummy để tránh lỗi

        sample = {
            "video": video_tensor,
            "landmark": landmark,
            "label": 0  # dummy label để Trainer không lỗi
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
	"""Checks if a file is an allowed extension.

	Args:
		filename (string): path to a file
		extensions (tuple of strings): extensions to consider (lowercase)

	Returns:
		bool: True if the filename ends with one of given extensions
	"""
	return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
	"""Checks if a file is an allowed image extension.

	Args:
		filename (string): path to a file

	Returns:
		bool: True if the filename ends with a known image extension
	"""
	return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_number_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
	"""Finds the class folders in a dataset.

	See :class:`DatasetFolder` for details.
	"""
	
	max_class = 0
	for entry in os.scandir(directory):
		if entry.is_file():
				try:
					class_id = int(os.path.splitext(entry.name)[0].split('P')[0].split('A')[1])
				except:
					# print(entry, "Not follow the formate _A_P_.avi or .mp4 skip")
					continue
				max_class =  max_class if class_id is None else max(max_class, class_id)
	
	return max_class



def make_dataset(
	directory: Union[str, Path],
	class_to_idx: Optional[Dict[int, int]] = None,
	extensions: Optional[Union[str, Tuple[str, ...]]] = ('.avi', '.mp4'),
	is_valid_file: Optional[Callable[[str], bool]] = None,
	allow_empty: bool = False,
) -> List[Tuple[str, int]]:
	directory = os.path.expanduser(directory)
	is_valid_file = cast(Callable[[str], bool], is_valid_file)

	# 🔧 Auto-generate label2id only if not provided
	if class_to_idx is None:
		class_ids = set()
		for file in Path(directory).glob("*.avi"):
			try:
				class_id = int(file.stem.split("P")[0].split("A")[1])
				class_ids.add(class_id)
			except Exception:
				continue
		class_ids = sorted(class_ids)
		class_to_idx = {cls: i for i, cls in enumerate(class_ids)}

	instances = []
	for file in Path(directory).glob("*.avi"):
		try:
			class_id = int(file.stem.split("P")[0].split("A")[1])
			if class_id not in class_to_idx:
				continue
			label = class_to_idx[class_id]
			instances.append((str(file), label))
		except:
			continue

	return instances

#region RGB Loader

import cv2
import numpy as np
import torch

def video_loader(video_path):
    vidcap = cv2.VideoCapture(str(video_path))
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Assuming 3 channels (RGB)
    frames = np.empty((frame_count, height, width, 3), dtype=np.uint8)

    idx = 0
    success, image = vidcap.read()
    while success and idx < frame_count:
        frames[idx] = image
        success, image = vidcap.read()
        idx += 1

    vidcap.release()

    # Truncate if read fewer frames than expected
    frames = frames[:idx]
    # Convert to tensor: C,T,H,W
    return torch.from_numpy(frames).permute(3, 0, 1, 2).float()

def save_video_as_npy(video_tensor, video_path):
    # Convert tensor from C,T,H,W to T,H,W,C and back to uint8
    video_array = video_tensor.permute(1, 2, 3, 0).to(torch.uint8).numpy()
    # Save as .npy file
    np.save(video_path, video_array)

def load_video_from_npy(video_path):
    # Load .npy file and convert back to tensor with C,T,H,W format
    video_array = np.load(video_path)
    return torch.from_numpy(video_array).permute(3, 0, 1, 2).float()






#region Iter dataset

class LabeledVideoDataset(torch.utils.data.IterableDataset):
	"""
	LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
	video dataset. It assumes each video is stored as either an encoded video
	(e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
	"""

	_MAX_CONSECUTIVE_FAILURES = 10

	def __init__(
		self,
		labeled_video_paths: List[Tuple[str, Optional[dict]]],
		clip_sampler: ClipSampler,
		video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
		transform: Optional[Callable[[dict], Any]] = None,
		decode_audio: bool = True,
		decoder: str = "pyav",
		**kwargs
	) -> None:
		"""
		Args:
			labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
					video file paths and associated labels. If video paths are a folder
					it's interpreted as a frame video, otherwise it must be an encoded
					video.

			clip_sampler (ClipSampler): Defines how clips should be sampled from each
				video. See the clip sampling documentation for more information.

			video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
				video container. This defines the order videos are decoded and,
				if necessary, the distributed split.

			transform (Callable): This callable is evaluated on the clip output before
				the clip is returned. It can be used for user defined preprocessing and
				augmentations on the clips. The clip output format is described in __next__().

			decode_audio (bool): If True, also decode audio from video.

			decoder (str): Defines what type of decoder used to decode a video. Not used for
				frame videos.
		"""
		self._decode_audio = decode_audio
		self._transform = transform
		self._clip_sampler = clip_sampler
		self._labeled_videos = labeled_video_paths
		
		self._preprocess_path()
		
		self._decoder = decoder

		# If a RandomSampler is used we need to pass in a custom random generator that
		# ensures all PyTorch multiprocess workers have the same random seed.
		self._video_random_generator = None
		if video_sampler == torch.utils.data.RandomSampler:
			self._video_random_generator = torch.Generator()
			self._video_sampler = video_sampler(
				self._labeled_videos, generator=self._video_random_generator
			)
		else:
			self._video_sampler = video_sampler(self._labeled_videos)

		self._video_sampler_iter = None  # Initialized on first call to self.__next__()

		# Depending on the clip sampler type, we may want to sample multiple clips
		# from one video. In that case, we keep the store video, label and previous sampled
		# clip time in these variables.
		self._loaded_video_label = None
		self._loaded_clip = None
		self._next_clip_start_time = 0.0
		self.video_path_handler = VideoPathHandler()

	def _preprocess_path(self):
		self.__labeled_videos_landmarks = []
		for video_path,label in self._labeled_videos:
			folder_root,img_name = os.path.split(video_path)
			root_dir = os.path.dirname(folder_root)
			landmark_path = os.path.join(root_dir,'npy',img_name+".npy")
			self.__labeled_videos_landmarks.append(((landmark_path,video_path),label))

			
		assert(len(self.__labeled_videos_landmarks) == len(self.__labeled_videos_landmarks))

	@property
	def video_sampler(self):
		"""
		Returns:
			The video sampler that defines video sample order. Note that you'll need to
			use this property to set the epoch for a torch.utils.data.DistributedSampler.
		"""
		return self._video_sampler

	@property
	def num_videos(self):
		"""
		Returns:
			Number of videos in dataset.
		"""
		return len(self.video_sampler)
#region __next__
	def __next__(self) -> dict:
		"""
		Retrieves the next clip based on the clip sampling strategy and video sampler.

		Returns:
			A dictionary with the following format.

			.. code-block:: text

				{
					'video': <video_tensor>,
					'label': <index_label>,
					'video_label': <index_label>
					'video_index': <video_index>,
					'clip_index': <clip_index>,
					'aug_index': <aug_index>,
				}
		"""
		if not self._video_sampler_iter:
			
			# Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
			self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

		for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
			# Reuse previously stored video if there are still clips to be sampled from
			# the last loaded video.
			
			video_index = next(self._video_sampler_iter)
			
			(landmark_path,video_path), info_dict = self.__labeled_videos_landmarks[video_index]
			cache_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)),"cache")
			id = os.path.splitext(os.path.basename(video_path))[0]
			base_cache_name = id+".npy"
			cache_video_path = os.path.join(cache_dir,base_cache_name)
			
			if os.path.exists(cache_video_path):
				video = load_video_from_npy(cache_video_path)
			else:
				# video = video_loader(video_path)
				# if not os.path.exists(cache_dir): os.mkdir(cache_dir)
				# save_video_as_npy(video,cache_video_path)
				print(f"Video {video_path} not found in cache, loading from disk...")
				continue
				


			if not os.path.exists(landmark_path):
				print(f"Body landmark is not exist: {landmark_path}")
			landmark = torch.tensor(np.load(landmark_path)).float()

			sample_dict = {
				"video": video,
				"video_index": video_index,
				'landmark': landmark,
				**info_dict,
			}
   
			if self._transform is not None:
				sample_dict = self._transform(sample_dict)
				# User can force dataset to continue by returning None in transform.
				if sample_dict is None:
					continue
			
			return sample_dict
		else:
			raise RuntimeError(
				f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
			)


	def __iter__(self):
		self._video_sampler_iter = None  # Reset video sampler

		# If we're in a PyTorch DataLoader multiprocessing context, we need to use the
		# same seed for each worker's RandomSampler generator. The workers at each
		# __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
		# which we can use for this seed.
		worker_info = torch.utils.data.get_worker_info()
		if self._video_random_generator is not None and worker_info is not None:
			base_seed = worker_info.seed - worker_info.id
			self._video_random_generator.manual_seed(base_seed)

		return self


def labeled_video_dataset(
	data_path: str,
	clip_sampler: ClipSampler,
	video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
	transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
	video_path_prefix: str = "",
	decode_audio: bool = True,
	decoder: str = "pyav",
    class_to_idx: Optional[Dict[int, int]] = None,
	**kwrags
) -> LabeledVideoDataset:
	"""
	A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

	Args:
		data_path (str): Path to the data. The path type defines how the data
			should be read:

			* For a file path, the file is read and each line is parsed into a
			  video path and label.
			* For a directory, the directory structure defines the classes
			  (i.e. each subdirectory is a class).

		clip_sampler (ClipSampler): Defines how clips should be sampled from each
				video. See the clip sampling documentation for more information.

		video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
				video container. This defines the order videos are decoded and,
				if necessary, the distributed split.

		transform (Callable): This callable is evaluated on the clip output before
				the clip is returned. It can be used for user defined preprocessing and
				augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
				output format.

		video_path_prefix (str): Path to root directory with the videos that are
				loaded in ``LabeledVideoDataset``. All the video paths before loading
				are prefixed with this path.

		decode_audio (bool): If True, also decode audio from video.

		decoder (str): Defines what type of decoder used to decode a video.

	"""
	labeled_video_paths = LabeledVideoPaths.from_directory(data_path, class_to_idx=class_to_idx)
	labeled_video_paths.path_prefix = video_path_prefix
	
	dataset = LabeledVideoDataset(
		labeled_video_paths,
		clip_sampler,
		video_sampler,
		transform,
		decode_audio=decode_audio,
		decoder=decoder,
		**kwrags
	)
	return dataset

def unlabeled_video_dataset(data_path, transform=None):
    return UnlabeledVideoDataset(data_path, transform=transform)

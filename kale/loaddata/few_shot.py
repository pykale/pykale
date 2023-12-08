"""
Dataset for N-way K-shot setting in few-shot problems.

"""

import os
from typing import Callable, Optional

import numpy as np
import torch

# import torchvision
from PIL import Image
from torch.utils.data import Dataset


class NWayKShotDataset(Dataset):
    """
    It is used to load data for N-way K-shot problems in few-shot learning.

    Note:
        The dataset should be organized as:

        - root
            - train
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...
            - val
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...
            - test (optional)
                - class_name 1
                    - xxx.png
                    - yyy.png
                    - ...
                - class_name 2
                    - xxx.png
                    - yyy.png
                    - ...
                - ...

    Args:
        path (string): The root directory of the data.
        mode (string): The mode of the dataset. It can be 'train', 'val' or 'test'. Default: 'train'.
        k_shot (int): Number of support examples per class in each episode. Default: 5.
        query_samples (int): Number of query examples per class in each episode. Default: 15.
        transform (callable, optional): Optional transform to be applied on a sample. Default: None.
    """

    def __init__(
        self,
        path: str,
        mode: str = "train",
        k_shot: int = 5,
        query_samples: int = 15,
        transform: Optional[Callable] = None,
    ):
        super(NWayKShotDataset, self).__init__()

        self.root = path
        self.k_shot = k_shot
        self.query_samples = query_samples
        self.mode = mode
        self.transform = transform
        self.images = []  # type: list
        self.labels = []  # type: list
        self._load_data()

    def __len__(self):
        # length of the dataset is the number of classes
        return len(self.classes)

    def __getitem__(self, idx):
        # sampling data for one class
        image_idx = self._get_idx(idx)
        images = self._sample_data(image_idx)
        assert isinstance(images, list)
        images = torch.stack(images)
        return images, idx

    def _get_idx(self, idx):
        # get the indices of images for one class
        image_idx = np.random.choice(
            [i for (i, item) in enumerate(self.labels) if item == idx],
            self.k_shot + self.query_samples,
            replace=False,
        )
        return image_idx

    def _sample_data(self, image_idx):
        # sampling data for one class
        images = [self.transform(self.images[index]) if self.transform else self.images[index] for index in image_idx]
        return images

    def _load_data(self):
        # loading data from the root directory
        data_path = os.path.join(self.root, self.mode)
        classes = os.listdir(data_path)
        classes.sort()
        self.classes = classes
        for i, c in enumerate(classes):
            c_dir = os.path.join(data_path, c)
            imgs = os.listdir(c_dir)
            for img in imgs:
                img_path = os.path.join(c_dir, img)
                self.images.append(Image.open(img_path).convert("RGB"))
                self.labels.append(i)

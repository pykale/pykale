"""
Dataset for N-way-K-shot setting in few-shot problems.
Author: Wenrui Fan
Email: winslow.fan@outlook.com
"""

import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NWayKShotDataset(Dataset):
    """
    It loads data for N-way K-shot problems in few-shot learning.

    - N-way: This refers to the number of different classes or categories in one epoch in evaluation. For example, in a 5-way problem, the model is fed with instances from 5 different classes for every epoch.

    - K-shot: It is the number of examples (or "shots") from each class in training and evaluation. In a 1-shot learning task, the model gets only one example per class, while in a 3-shot task, it gets three examples per class.

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
        mode (string): The mode of the dataset. It can be 'train', 'val', or 'test'. Default: 'train'.
        num_support_samples (int): Number of support examples per class in each iteration. It is corresponding to K in N-way-K-shot setting Default: 5.
        num_query_samples (int): Number of query examples per class in each iteration. Default: 15.
        transform (callable, optional): Optional transform to be applied on a sample. Default: None.
    """

    def __init__(
        self,
        path: str,
        mode: str = "train",
        num_support_samples: int = 5,
        num_query_samples: int = 15,
        transform: Optional[Callable] = None,
    ):
        super(NWayKShotDataset, self).__init__()

        self.root = path
        self.num_support_samples = num_support_samples
        self.num_query_samples = num_query_samples
        self.mode = mode
        self.transform = transform
        self.images = []  # type: list
        self.labels = []  # type: list
        self._load_data()

    def __len__(self):
        # the length of the dataset is the number of classes
        return len(self.classes)

    def __getitem__(self, idx):
        # sampling data for one class
        image_idx = self._get_idx(idx)
        images = self._sample_data(image_idx)
        assert isinstance(images, list)
        images = torch.stack(images)
        return images, idx

    def _get_idx(self, idx):
        # getting the indices of images for one class
        image_idx = np.random.choice(
            [i for (i, item) in enumerate(self.labels) if item == idx],
            self.num_support_samples + self.num_query_samples,
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

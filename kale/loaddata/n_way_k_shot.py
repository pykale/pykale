import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NWayKShotDataset(Dataset):
    """
    This is a dataset class for few-shot learning. It is used to load data for N-way K-shot learning for Prototypical Networks.

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
        path (string): A string, which is the root directory of data.
        mode (string): A string, which is the mode of the dataset.
                        It can be 'train', 'val' or 'test'.
        k_shot (int): Number of support examples for each class in each episode.
        query_samples (int): Number of query examples for each class in each episode.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, path: str, mode: str = "train", k_shot: int = 5, query_samples: int = 15, transform: Any = None):
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
        return len(self.classes)

    def __getitem__(self, idx):
        image_idx = self._get_idx(idx)
        images = self._sample_data(image_idx)
        assert isinstance(images, list)
        images = torch.stack(images)
        return images, idx

    def _get_idx(self, idx):
        image_idx = np.random.choice(
            [i for (i, item) in enumerate(self.labels) if item == idx], self.k_shot + self.query_samples, replace=False,
        )
        return image_idx

    def _sample_data(self, image_idx):
        images = [self.transform(self.images[index]) if self.transform else self.images[index] for index in image_idx]
        return images

    def _load_data(self):
        cls_path = os.path.join(self.root, self.mode)
        classes = os.listdir(cls_path)
        classes.sort()
        self.classes = classes
        for i, c in enumerate(classes):
            c_dir = os.path.join(cls_path, c)
            imgs = os.listdir(c_dir)
            for img in imgs:
                img_path = os.path.join(c_dir, img)
                self.images.append(Image.open(img_path).convert("RGB"))
                self.labels.append(i)

import os
from typing import Any
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NWayKShotDataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str = "train",
        k_shot: int = 5,
        query_samples: int = 15,
        transform: Any = None
    ):
        super(NWayKShotDataset, self).__init__()
        """
        This is a dataset class for few-shot learning. The dataset should be organized as:
        root:
            - train:
                - class_1:
                    - img_1
                    - img_2
                    ...
                - class_2:
                    - img_1
                    - img_2
                    ...
                ...
            - val:
                - class_1:
                    - img_1
                    - img_2
                    ...
                - class_2:
                    - img_1
                    - img_2
                    ...
                ...
            - test (optional):
                - class_1:
                    - img_1
                    - img_2
                    ...
                - class_2:
                    - img_1
                    - img_2
                    ...
                ... 

        Args:
            path (string): A string, which is the root directory of data.
            mode (string): A string, which is the mode of the dataset. 
                           It can be 'train', 'val' or 'test'.
            k_shot (int): Number of support examples for each class in each episode.
            query_samples (int): Number of query examples for each class in each episode.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = path
        self.k_shot = k_shot
        self.query_samples = query_samples
        self.mode = mode
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()
        print("Load {} images in {} mode.".format(len(self.labels), self.mode))

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        image_idx = np.random.choice(
            [i for (i, item) in enumerate(self.labels) if item == idx],
            self.k_shot + self.query_samples,
            replace=False,
        )
        images = []
        labels = []
        for index in image_idx:
            labels.append(self.labels[index])
            if self.transform:
                images.append(self.transform(self.images[index]))
            else:
                images.append(self.images[index])
        images = torch.stack(images)
        return images, idx

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

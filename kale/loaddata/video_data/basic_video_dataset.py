import os
import math

import torch
from torch.utils.data import Dataset

from PIL import Image
import pickle
import numpy as np


class BasicVideoDataset(Dataset):
    """
    Dataset for GTEA, ADL, KITCHEN and EPIC-Kitchen.

    Args:
        data_path (string): image directory of dataset
        list_path (string): list file directory of dataset
        mode (string): image type (RGB or Optical Flow)
        dataset_split (string): split type (train or test)
        window_len (int): length of each action sample (the unit is number of frame)
        transforms (Compose): Video transform
    """
    def __init__(self, data_path, list_path, mode, dataset_split, n_classes, window_len=16, transforms=None):
        """Init action video dataset."""
        self.data_path = data_path
        self.list_path = list_path
        self.mode = mode
        self.dataset = dataset_split
        self.n_classes = n_classes
        self.window_len = window_len
        self.transforms = transforms
        self.data = self.make_dataset()

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def __getitem__(self, item):
        """
        Get images from each action video sample and labels for data loader.

        Args:
            item (int): item

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        self.vid, self.start_frame, self.end_frame, self.label = self.data[item]
        self.img_path = os.path.join(self.data_path, self.mode, self.vid)

        if self.mode == 'rgb':
            imgs = self.load_rgb_frames()
        elif self.mode == 'flow':
            imgs = self.load_flow_frames()

        if self.transforms is not None:
            imgs = self.transforms(imgs)
        return self.video_to_tensor(imgs), self.label

    def make_dataset(self):
        """
        Load data from the list file.

        Returns:
            data (list): list of (video_name, start_frame, end_frame, label)
        """
        data = []
        i = 0
        with open(self.list_path, 'rb') as input_file:
            input_file = pickle.load(input_file)
            for line in input_file.values:
                if 0 <= eval(line[5]) < self.n_classes:
                    data.append((line[0], eval(line[1]), eval(line[2]), eval(line[5])))
                    i = i + 1
        print("Number of {:5} action segments: {}".format(self.dataset, i))
        return data

    def load_rgb_frames(self):
        """Load and normalize RGB frames from dataset."""
        frames = []
        for i in range(self.start_frame, self.start_frame + self.window_len):
            dir = os.path.join(self.img_path, 'frame_{:0>10}.jpg'.format(i))
            img = Image.open(dir).convert('RGB')
            w, h = img.size
            if w < 255 or h < 255:
                d = 255. - min(w, h)
                sc = 1 + d / min(w, h)
                img = img.resize(int(w * sc), int(h * sc))
            img = np.asarray(img)
            img = self.linear_norm(img)
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)

    def load_flow_frames(self):
        """Load and normalize Optical Flow frames from dataset."""
        frames = []
        start_frame = math.ceil(self.start_frame / 2) - 1
        if start_frame == 0:
            start_frame = 1
        if self.window_len > 1:
            window_len = self.window_len / 2
        for i in range(int(start_frame), int(start_frame + window_len)):
            diru = os.path.join(self.img_path, 'u', 'frame_{:0>10}.jpg'.format(i))
            dirv = os.path.join(self.img_path, 'v', 'frame_{:0>10}.jpg'.format(i))
            imgu = Image.open(diru).convert("L")
            imgv = Image.open(dirv).convert("L")
            w, h = imgu.size
            if w < 255 or h < 255:
                d = 255. - min(w, h)
                sc = 1 + d / min(w, h)
                imgu = imgu.resize(int(w * sc), int(h * sc))
                imgv = imgv.resize(int(w * sc), int(h * sc))
            img = np.asarray([np.array(imgu), np.array(imgv)]).transpose([1, 2, 0])
            img = self.linear_norm(img)
            frames.append(img)
        return np.asarray(frames, dtype=np.float32)

    @staticmethod
    def video_to_tensor(pic):
        """
        Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
            pic (numpy.ndarray): Video to be converted to tensor.

        Returns:
            Tensor: Converted video.
        """
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

    @staticmethod
    def linear_norm(arr):
        """
        Linear normalize the image.
        
        Args:
            arr (numpy.ndarray): image to be normalized.
        """
        arr = arr.astype('float')
        for i in range(arr.shape[-1]):
            a = arr[..., i]
            b = a / 255.
            c = b * 2
            d = c - 1
            arr[..., i] = (arr[..., i] / 255.) * 2 - 1
        return arr

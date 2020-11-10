import os
import random
import math

import torch
from torch.utils.data import Dataset

from PIL import Image
import pickle
import numpy as np


class EPIC(Dataset):
    def __init__(self, data_path, list_path, mode, dataset_split, window_len=16, transforms=None):
        self.data_path = data_path
        self.list_path = list_path
        self.mode = mode
        self.window_len = window_len
        self.transforms = transforms
        self.dataset = dataset_split
        self.data = self.make_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        self.vid, self.start, self.end, self.label = self.data[item]
        self.start_f = random.randint(self.start, self.end - (self.window_len + 1))

        if self.mode == 'rgb':
            imgs = self.load_rgb_frames()
        elif self.mode == 'flow':
            imgs = self.load_flow_frames()

        if self.transforms is not None:
            # try:
            imgs = self.transforms(imgs)
            # except:
            #     print('Cannot transform image: {} from {} to {}.'.format(vid, start, end))

        return self.video_to_tensor(imgs), self.label

    def make_dataset(self):
        data = []
        class_num = 8
        i = 0
        with open(self.list_path, 'rb') as input_file:
            input_file = pickle.load(input_file)
            # print(type(input_file))
            for line in input_file.values:
                if line[1] in ['P01', 'P08', 'P22']:
                    # Select class 'put', 'take', 'open', 'close', 'wash', 'cut', 'mix', 'pour', 'put down', 'wipe'...
                    if 0 <= line[9] <= 7:
                        if line[7] - line[6] + 1 >= self.window_len:
                            # print(line[0], line[1], line[5], line[6], line[7], line[8])
                            ### matrix label
                            # label = np.zeros((class_num, self.window_len), np.float32)
                            # for fr in range(0, self.window_len):
                            #     label[line[9], fr] = 1
                            ### vector label
                            # label = np.zeros(class_num, np.float32)
                            # label[line[9]] = 1
                            ### index label
                            label = line[9]

                            # label = np.zeros((class_num, 1), np.float32)
                            # label[line[8]] = 1

                            # label = line[8]
                            data.append((os.path.join(line[1], line[2]), line[6], line[7], label))
                            i = i + 1
        print("Number of {:5} action segments: {}".format(self.dataset, i))
        return data

    def load_rgb_frames(self):
        frames = []
        # dataset = 'train'
        for i in range(self.start_f, self.start_f + self.window_len):
            dir = os.path.join(self.data_path, self.mode, self.dataset, self.vid, 'frame_{:0>10}.jpg'.format(i))
            # try:
            img = Image.open(dir).convert('RGB')
            w, h = img.size
            if w < 255 or h < 255:
                d = 255. - min(w, h)
                sc = 1 + d / min(w, h)
                img = img.resize(int(w * sc), int(h * sc))
            img = np.asarray(img)
            img = self.linear_norm(img)
            # img = Image.fromarray(img.astype('uint8'), 'RGB')
            # img = np.asarray(img, dtype=np.float32)
            frames.append(img)
            # except:
            #     print('Cannot read image: {}'.format(dir))
        return np.asarray(frames, dtype=np.float32)

    def load_flow_frames(self):
        frames = []
        # dataset = 'train'
        start_f = math.ceil(self.start_f / 2) - 1
        if start_f == 0:
            start_f = 1
        if self.window_len > 1:
            window_len = self.window_len / 2
        for i in range(int(start_f), int(start_f + window_len)):
            diru = os.path.join(self.data_path, self.mode, self.dataset, self.vid, 'u', 'frame_{:0>10}.jpg'.format(i))
            dirv = os.path.join(self.data_path, self.mode, self.dataset, self.vid, 'v', 'frame_{:0>10}.jpg'.format(i))
            # try:
            imgu = Image.open(diru).convert("L")
            imgv = Image.open(dirv).convert("L")
            w, h = imgu.size
            if w < 255 or h < 255:
                d = 255. - min(w, h)
                sc = 1 + d / min(w, h)
                imgu = imgu.resize(int(w * sc), int(h * sc))
                imgv = imgv.resize(int(w * sc), int(h * sc))
            # imgu = linear_norm(np.array(imgu))
            # imgv = linear_norm(np.array(imgv))
            img = np.asarray([np.array(imgu), np.array(imgv)]).transpose([1, 2, 0])
            img = self.linear_norm(img)
            frames.append(img)
            # except:
            #     print('Cannot read image: {}'.format(
            #         os.path.join(root, mode, dataset, vid, 'u or v', 'frame_{:0>10}.jpg'.format(i))))
        return np.asarray(frames, dtype=np.float32)

    @staticmethod
    def video_to_tensor(pic):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
            pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
            Tensor: Converted video.
        """
        return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

    @staticmethod
    def default_loader(dir):
        try:
            img = Image.open(dir)
            return img.convert('RGB')
        except:
            print('Cannot read image: {}'.format(dir))

    @staticmethod
    def linear_norm(arr):
        # arr = np.asarray(img)
        arr = arr.astype('float')
        for i in range(arr.shape[-1]):
            arr[..., i] = (arr[..., i] / 255.) * 2 - 1
        # img = Image.fromarray(arr.astype('uint8'), 'RGB')
        return arr

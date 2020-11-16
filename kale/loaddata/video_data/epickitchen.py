import os
import random

from .basic_video_dataset import BasicVideoDataset

import pickle


class EPIC(BasicVideoDataset):

    def __getitem__(self, item):
        """
        Get images from each EPIC-Kitchen action video sample and labels for data loader.
        Randomly get a 16-frame sample from each video
        Follow this paper. (https://ieeexplore.ieee.org/document/9156981)
        """
        self.vid, self.start_frame, self.end_frame, self.label = self.data[item]
        self.rand_start_frame = random.randint(self.start_frame, self.end_frame - (self.window_len + 1))
        self.img_path = os.path.join(self.data_path, self.mode, self.dataset, self.vid)

        if self.mode == 'rgb':
            imgs = self.load_rgb_frames()
        elif self.mode == 'flow':
            imgs = self.load_flow_frames()

        if self.transforms is not None:
            imgs = self.transforms(imgs)
        return self.video_to_tensor(imgs), self.label

    def make_dataset(self):
        """
        Load data from the EPIC-Kitchen list file.
        Because the original list files are not the same, inherit from class BasicVideoDataset and be modified.
        """
        data = []
        i = 0
        with open(self.list_path, 'rb') as input_file:
            input_file = pickle.load(input_file)
            for line in input_file.values:
                if line[1] in ['P01', 'P08', 'P22']:
                    if 0 <= line[9] < self.n_classes:
                        if line[7] - line[6] + 1 >= self.window_len:
                            label = line[9]
                            data.append((os.path.join(line[1], line[2]), line[6], line[7], label))
                            i = i + 1
        print("Number of {:5} action segments: {}".format(self.dataset, i))
        return data

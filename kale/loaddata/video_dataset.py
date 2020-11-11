import os
import os.path
import numpy as np
from numpy.random import randint
from PIL import Image
from torchvision import transforms
import torch.utils.data as data

class VideoRecord(object):
    def __init__(self, row, root_path):
        self._data = row
        self._path = os.path.join(root_path, row[0])


    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, annotation_file,
                 num_segments=3, frames_per_segment=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.annotation_file = annotation_file
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.image_tmpl = image_tmpl  # image filename template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '), self.root_path) for x in open(self.annotation_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.frames_per_segment + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        """
        For each segment, returns the center frame indices.
        Returns the list of all indices of all segments.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, p)
                images.extend(seg_img)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

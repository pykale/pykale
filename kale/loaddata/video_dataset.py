import os
import os.path
import numpy as np
from numpy.random import randint
from PIL import Image
from torchvision import transforms
import torch

class VideoRecord(object):
    """
    Helper class for below class VideoFrameDataset. This class
    represents a video sample's metadata.
    """
    def __init__(self, row, root_datapath):
        """
        :param root_datapath: the system path to the root folder
        of the videos.
        :param row: a list with three elements where 1) the first
        element is the path to the video sample's frames excluding
        the root_datapath prefix 2) the  second element is the number
        of frames 3) the third element is the label index.
        """
        self._data = row
        self._path = os.path.join(root_datapath, row[0])


    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

class VideoFrameDataset(torch.utils.data.Dataset):
    """
    A dataset class for videos. Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or (FRAMES x CHANNELS x HEIGHT x WIDTH)
    tensors where FRAMES=x if the right transforms are used.

    More specifically, the frame range [0,N] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT frames are taken from each segment.

    Note:
        This dataset corresponds to the frame sampling technique introduced
        in "Temporal Segment Networks" at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside the ROOT_DATA folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg. Numbering must start at 1.
        For enumeration and annotations, this class expects to receive
        a .txt file where each video sample has a row with three
        space separated values:
        `VIDEO_FOLDER_PATH NUM_FRAMES LABEL_INDEX`
        VIDEO_FOLDER_PATH is expected to be the path of a video folder
        starting inside of ROOT_DATA. For example, ROOT_DATA might
        be `home\data\datasetxyz\videos\` inside of which a VIDEO_FOLDER_PATH
        might be `jumping\0052\` or `sample1\` or `00053\`.

    Note:
        A demonstration of using this class can be seen
        in PyKale/Examples/VideoFrameDataset

    """
    def __init__(self, root_path, annotation_file,
                 num_segments=3, frames_per_segment=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False):
        """

        :param root_path: the root path in which video folders lie.
        this is ROOT_DATA from the description above.
        :param annotation_file: the .txt annotation file containing
        one row per video sample as described above.
        :param num_segments: the number of segments the video should
        be divided into to sample frames from.
        :param frames_per_segment: the number of frames that should
        be loaded per segment. for each segment's frame-range, a
        random start index is chosen, from which frames_per_segment
        consecutive frames are loaded.
        :param image_tmpl: the image filename template that video frame
        files have inside of their video folders as described above.
        :param transform: transforms that act on lists of images/frames.
        :param random_shift: whether to choose a random start index for
        frames inside each segment frame-range, or choose frames
        from the center of each segment.
        :param test_mode: whether this is a test dataset. if so, chooses
        frames from segments with random_shift=False.
        """

        self.root_path = root_path
        self.annotation_file = annotation_file
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.image_tmpl = image_tmpl
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
        For each segment, chooses an index from where frames
        are to be loaded from.
        :param record: VideoRecord denoting a video sample
        :return: list of indices of where the frames of each
        segment are to be loaded from.
        """

        average_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)

        # edge cases for when a video only has a tiny number of frames.
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.frames_per_segment + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        """
        For each segment, finds the center frame index.
        :param record: VideoRecord denoting a video sample
        :return: list of indices of segment center frames.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.zeros((self.num_segments,)) + 1
        return offsets

    def _get_test_indices(self, record):
        """
        For each segment, finds the center frame index.
        :param record: VideoRecord denoting a video sample
        :return: list of indices of segment center frames.
        """

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.
        :param index: video sample index
        :return: a list of PIL images or the result
        of applying self.transform on this list if
        self.transform is not None.
        """
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        """
        Loads the frames of a video at the corresponding
        indices.
        :param record: VideoRecord denoting a video sample
        :param indices: indices at which to load video frames from
        :return:
        1) a list of PIL images or the result
        of applying self.transform on this list if
        self.transform is not None.
        2) an integer denoting the video label.
        """

        images = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                if frame_index < record.num_frames:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)

def imglist_totensor(img_list):
    """
    Converts each PIL image in a list to
    a torch Tensor and stacks them into
    a single tensor.
    :param img_list: list of PIL images.
    :return: tensor of size
    NUM_IMAGES x CHANNELS x HEIGHT x WIDTH
    """
    return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
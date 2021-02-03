import math
import os
import os.path
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive

    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def label(self):
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]


class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``kale.prepdata.video_transform.ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``PyKale/examples/video_loading``
        https://github.com/pykale/pykale/tree/master/examples/video_loading

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see example README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME     END_FRAME     LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        image_modality: Image modality (RGB or Optical Flow).
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.

    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        image_modality: str = "rgb",
        num_segments: int = 3,
        frames_per_segment: int = 1,
        imagefile_template: str = "img_{:05d}.jpg",
        transform=None,
        random_shift: bool = True,
        test_mode: bool = False,
    ):
        super(VideoFrameDataset, self).__init__()

        self.root_path = Path(root_path)
        self.annotationfile_path = Path(annotationfile_path)
        self.image_modality = image_modality
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        if self.image_modality == "flow" and self.frames_per_segment > 1:
            self.frames_per_segment //= 2

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.image_modality == "rgb":
            return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert("RGB")]
        elif self.image_modality == "flow":
            idx = math.ceil(idx / 2) - 1 if idx > 2 else 1
            x_img = Image.open(os.path.join(directory, self.imagefile_template.format("x", idx))).convert("L")
            y_img = Image.open(os.path.join(directory, self.imagefile_template.format("y", idx))).convert("L")
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(" "), self.root_path) for x in open(self.annotationfile_path)]

    def _get_random_indices(self, record):
        """
        For each segment, randomly chooses the start frame indexes.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of segment start frames.
        """

        if record.num_frames > self.num_segments * self.frames_per_segment - 1:
            segment_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
            offsets = np.multiply(list(range(self.num_segments)), segment_duration) + np.random.randint(
                segment_duration, size=self.num_segments
            )
        else:
            offsets = np.sort(random.sample(range(record.num_frames - self.frames_per_segment), self.num_segments))
        return offsets

    def _get_symmetric_indices(self, record):
        """
        For each segment, finds the start frame indexes which are symmetrical.

        Args:
            record: VideoRecord denoting a video sample
        Returns:
            List of indices of segment start frames.
        """

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.

        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """

        record = self.video_list[index]

        if record.num_frames < self.frames_per_segment:
            raise RuntimeError(
                "Path:{}, start:{}, end:{}.\n Video_length is {}, which should be larger than "
                "frame_per_segment {}.".format(
                    record.path, record.start_frame, record.end_frame, record.num_frames, self.frames_per_segment
                )
            )
        elif record.num_frames < self.num_segments:
            raise RuntimeError(
                "Path:{}, start:{}, end:{}.\n Video_length is {}, which should be larger than "
                "num_segments {}.".format(
                    record.path, record.start_frame, record.end_frame, record.num_frames, self.num_segments
                )
            )
        elif record.num_frames < self.num_segments * self.frames_per_segment:
            if self.num_segments > record.num_frames - self.frames_per_segment + 1:
                raise RuntimeError(
                    "Path:{}, start:{}, end:{}.\n Video_length is {}, num_segments is {} and "
                    "frame_per_segment is {}. Please make num_segments<frame_length-frames_per_segment "
                    "to avoid getting too many same segments.".format(
                        record.path,
                        record.start_frame,
                        record.end_frame,
                        record.num_frames,
                        self.num_segments,
                        self.frames_per_segment,
                    )
                )

        if not self.test_mode:
            segment_indices = (
                self._get_random_indices(record) if self.random_shift else self._get_symmetric_indices(record)
            )
        else:
            segment_indices = self._get_symmetric_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """
        Loads the frames of a video at the corresponding indices.

        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from.
        Returns:
            1) A list of PIL images or the result of applying self.transform on this list if self.transform is not None.
            2) An integer denoting the video label.
        """

        indices = indices + record.start_frame
        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index)
                images.extend(seg_img)
                image_indices.append(frame_index)
                if frame_index < record.end_frame:
                    frame_index += 1

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)

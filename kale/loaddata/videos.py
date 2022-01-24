import math
import os
import os.path
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath (Path, optional): the system path to the root folder of the videos.
        row (tuple, optional): A list with four or more elements where
            1) The first element is the path to the video sample's frames excluding the root_datapath prefix.
            2) The second element is the starting frame id of the video.
            3) The third element is the inclusive ending frame id of the video.
            4) The fourth element is the label index.
            5) Any following elements are labels in the case of multi-label classification.
    """

    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def segment_id(self):
        return "{}-{}-{}".format(self._data[0], self._data[1], self._data[2])

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
            return [int(self._data[3])]
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]


class VideoFeatureRecord(object):
    """
    Helper class for class VideoFeatureDataset. This class represents a video feature vector.

    Args:
        index (int): the index of the video feature vector.
        row (pandas.Series, optional): A series with information of feature vector.
        num_segments (int): the number of segments to split the video into.
    """

    def __init__(self, index, row, num_segments):
        self._data = row
        self._index = index
        self._n_seg = num_segments

    @property
    def segment_id(self):
        return self._data.narration_id

    @property
    def num_frames(self):
        return int(self._n_seg)

    @property
    def label(self):
        if ("verb_class" in self._data) and ("noun_class" in self._data):
            return int(self._data.verb_class), int(self._data.noun_class)
        elif ("verb_class" in self._data) and ("noun_class" not in self._data):
            return [int(self._data.verb_class)]
        else:
            return 0, 0


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
        root_path (str, Path): root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path (str, Path): .txt annotation file containing
                             one row per video sample as described above.
        image_modality (str): image modality (RGB or Optical Flow).
        num_segments (int): number of segments the video should be divided into to sample frames from.
                            (Default is 1 in image mode and 5 in feature mode)
        frames_per_segment (int): number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded. (set as 1 in feature vector mode)
        imagefile_template (str): image filename template that video frame files
                            have inside of their video folders as described above.
        transform (Compose, optional): transform pipeline that receives a list of PIL images/frames.
        random_shift (bool): whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode (bool): whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.
        input_type (str): type of input. (options: 'image' or 'feature')
        num_data_load (int): number of the data to load. (only used in feature vector mode)
        total_segments (int): total number of segments a video is divided into. (only used in feature mode)

    """

    def __init__(
        self,
        root_path,
        annotationfile_path,
        image_modality="rgb",
        num_segments=5,
        frames_per_segment=1,
        imagefile_template="img_{:05d}.jpg",
        transform=None,
        random_shift=True,
        test_mode=False,
        input_type="image",
        num_data_load=None,
        total_segments=25,
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
        self.input_type = input_type
        self.num_data_load = num_data_load
        self.total_segments = total_segments
        self._data = None
        if self.input_type == "feature":
            if self.total_segments != 25 or self.frames_per_segment != 1:
                raise ValueError("total_segments and frames_per_segment must be 25 and 1 in feature vector mode")

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.image_modality == "rgb":
            return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert("RGB")]
        elif self.image_modality == "flow":
            idx = math.ceil(idx / 2) - 1 if idx > 2 else 1
            x_img = Image.open(os.path.join(directory, self.imagefile_template.format("x", idx))).convert("L")
            y_img = Image.open(os.path.join(directory, self.imagefile_template.format("y", idx))).convert("L")
            return [x_img, y_img]
        else:
            raise ValueError("Input modality is not in [rgb, flow, joint]. Current is {}".format(self.image_modality))

    def _load_feature_vector(self, idx, segment):
        if self._data is None:
            self._read_feature_vector()
        if (
            self.image_modality == "rgb"
            or self.image_modality == "flow"
            or self.image_modality == "audio"
            or self.image_modality == "all"
        ):
            return torch.from_numpy(np.expand_dims(self._data[segment][idx - 1], axis=0)).float()
        else:
            raise ValueError(
                "Input modality is not in [rgb, flow, audio, all]. Current is {}".format(self.image_modality)
            )

    def _read_feature_vector(self):
        with open(self.root_path, "rb") as f:
            data = pickle.load(f)
            if self.image_modality == "all":
                data_features = np.concatenate(list(data["features"].values()), -1)
            elif self.image_modality == "rgb":
                data_features = data["features"]["RGB"]
            elif self.image_modality == "flow":
                data_features = data["features"]["Flow"]
            elif self.image_modality == "audio":
                data_features = data["features"]["Audio"]
            data_narrations = data["narration_ids"]
            self._data = dict(zip(data_narrations, data_features))

    def _parse_list(self):
        if self.input_type == "image":
            self.video_list = [
                VideoRecord(x.strip().split(" "), self.root_path) for x in open(self.annotationfile_path)
            ]
        elif self.input_type == "feature":
            label_file = pd.read_pickle(self.annotationfile_path).reset_index()
            self.video_list = [
                VideoFeatureRecord(i, row[1], self.total_segments) for i, row in enumerate(label_file.iterrows())
            ]
            # repeat the list if the length is less than num_data_load (especially for target data)
            n_repeat = self.num_data_load // len(self.video_list)
            n_left = self.num_data_load % len(self.video_list)
            self.video_list = self.video_list * n_repeat + self.video_list[:n_left]

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
        if self.input_type == "image":
            indices = indices + record.start_frame

        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                if self.input_type == "image":
                    seg_img = self._load_image(record.path, frame_index)
                else:
                    seg_img = self._load_feature_vector(frame_index, record.segment_id)
                images.extend(seg_img)
                image_indices.append(frame_index)

                if self.input_type == "image":
                    frame_index = frame_index + 1 if frame_index < record.end_frame else frame_index
                else:  # feature vector does not have record.end_frame.
                    frame_index = frame_index + 1 if frame_index < record.num_frames else frame_index

        if self.input_type == "image" and self.transform is not None:
            images = self.transform(images)

        return images, record.label, record.segment_id

    def __len__(self):
        return len(self.video_list)

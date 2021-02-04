import logging
import math
import os
import pickle
from pathlib import Path

from PIL import Image

from kale.loaddata.videos import VideoFrameDataset, VideoRecord


class BasicVideoDataset(VideoFrameDataset):
    """
    Dataset for GTEA, ADL and KITCHEN.

    Args:
        root_path (string): The root path in which video folders lie.
        annotationfile_path (string): The annotation file containing one row per video sample.
        dataset_split (string): Split type (train or test)
        image_modality (string): Image modality (RGB or Optical Flow)
        num_segments (int): The number of segments the video should be divided into to sample frames from.
        frames_per_segment (int): The number of frames that should be loaded per segment.
        imagefile_template (string): The image filename template.
        transform (Compose): Video transform.
        random_shift (bool): Whether the frames from each segment should be taken consecutively starting from
                        the center(False) of the segment, or consecutively starting from
                        a random(True) location inside the segment range.
        test_mode (bool): Whether this is a test dataset. If so, chooses frames from segments with random_shift=False.
        n_classes (int): The number of classes.
    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        dataset_split: str,
        image_modality: str,
        num_segments: int = 1,
        frames_per_segment: int = 16,
        imagefile_template: str = "img_{:010d}.jpg",
        transform=None,
        random_shift: bool = True,
        test_mode: bool = False,
        n_classes: int = 8,
    ):
        self.root_path = Path(root_path)
        self.image_modality = image_modality
        self.dataset = dataset_split
        self.n_classes = n_classes
        self.img_path = self.root_path.joinpath(self.image_modality)
        super(BasicVideoDataset, self).__init__(
            root_path,
            annotationfile_path,
            image_modality,
            num_segments,
            frames_per_segment,
            imagefile_template,
            transform,
            random_shift,
            test_mode,
        )

    def _parse_list(self):
        self.video_list = [VideoRecord(x, self.img_path) for x in list(self.make_dataset())]

    def make_dataset(self):
        """
        Load data from the EPIC-Kitchen list file and make them into the united format.
        Different datasets correspond to a different number of classes.

        Returns:
            data (list): list of (video_name, start_frame, end_frame, label)
        """

        data = []
        i = 0
        with open(self.annotationfile_path, "rb") as input_file:
            input_file = pickle.load(input_file)
            for line in input_file.values:
                if 0 <= eval(line[5]) < self.n_classes:
                    data.append((line[0], eval(line[1]), eval(line[2]), eval(line[5])))
                    i = i + 1
        logging.info("Number of {:5} action segments: {}".format(self.dataset, i))
        return data


class EPIC(VideoFrameDataset):
    """
    Dataset for EPIC-Kitchen.
    """

    def __init__(
        self,
        root_path: str,
        annotationfile_path: str,
        dataset_split: str,
        image_modality: str,
        num_segments: int = 1,
        frames_per_segment: int = 16,
        imagefile_template: str = "img_{:010d}.jpg",
        transform=None,
        random_shift: bool = True,
        test_mode: bool = False,
        n_classes: int = 8,
    ):
        self.root_path = Path(root_path)
        self.image_modality = image_modality
        self.dataset = dataset_split
        self.n_classes = n_classes
        self.img_path = self.root_path.joinpath(self.image_modality, self.dataset)
        super(EPIC, self).__init__(
            root_path,
            annotationfile_path,
            image_modality,
            num_segments,
            frames_per_segment,
            imagefile_template,
            transform,
            random_shift,
            test_mode,
        )

    def _parse_list(self):
        self.video_list = [VideoRecord(x, self.img_path) for x in list(self.make_dataset())]

    def _load_image(self, directory, idx):
        if self.image_modality == "rgb":
            return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert("RGB")]
        elif self.image_modality == "flow":
            idx = math.ceil(idx / 2) - 1 if idx > 2 else 1
            u_img = Image.open(os.path.join(directory, "u", self.imagefile_template.format(idx))).convert("L")
            v_img = Image.open(os.path.join(directory, "v", self.imagefile_template.format(idx))).convert("L")
            return [u_img, v_img]

    def make_dataset(self):
        """
        Load data from the EPIC-Kitchen list file and make them into the united format.
        Because the original list files are not the same, inherit from class BasicVideoDataset and be modified.
        """

        data = []
        i = 0
        with open(self.annotationfile_path, "rb") as input_file:
            input_file = pickle.load(input_file)
            for line in input_file.values:
                if line[1] in ["P01", "P08", "P22"]:
                    if 0 <= line[9] < self.n_classes:
                        if line[7] - line[6] + 1 >= self.frames_per_segment:
                            label = line[9]
                            data.append((os.path.join(line[1], line[2]), line[6], line[7], label))
                            i = i + 1
        logging.info("Number of {:5} action segments: {}".format(self.dataset, i))
        return data

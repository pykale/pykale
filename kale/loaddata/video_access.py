"""
Action video dataset loading for EPIC-Kitchen, ADL, GTEA, KITCHEN. The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os
from copy import deepcopy
from enum import Enum

import kale.prepdata.video_transform as video_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.video_datasets import BasicVideoDataset, EPIC


def get_videodata_config(cfg):
    config_params = {
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_src_name": cfg.DATASET.SOURCE,
            "dataset_src_trainlist": cfg.DATASET.SRC_TRAINLIST,
            "dataset_src_testlist": cfg.DATASET.SRC_TESTLIST,
            "dataset_tar_name": cfg.DATASET.TARGET,
            "dataset_tar_trainlist": cfg.DATASET.TAR_TRAINLIST,
            "dataset_tar_testlist": cfg.DATASET.TAR_TESTLIST,
            "dataset_image_modality": cfg.DATASET.IMAGE_MODALITY,
            "num_classes": cfg.DATASET.NUM_CLASSES,
            "frames_per_segment": cfg.DATASET.FRAMES_PER_SEGMENT,
        }
    }
    return config_params


def generate_list(data_name, data_params_local, domain):
    """

    Args:
        data_name (string): name of dataset
        data_params_local (dict): hyper parameters from configure file
        domain (string): domain type (source or target)

    Returns:
        data_path (string): image directory of dataset
        train_listpath (string): training list file directory of dataset
        test_listpath (string): test list file directory of dataset
    """

    if data_name == "EPIC":
        dataset_path = os.path.join(data_params_local["dataset_root"], data_name, "EPIC_KITCHENS_2018")
        data_path = os.path.join(dataset_path, "frames_rgb_flow")
    elif data_name in ["ADL", "GTEA", "KITCHEN"]:
        dataset_path = os.path.join(data_params_local["dataset_root"], data_name)
        data_path = os.path.join(dataset_path, "frames_rgb_flow")
    else:
        raise RuntimeError("Wrong dataset name. Select from [EPIC, ADL, GTEA, KITCHEN]")

    train_listpath = os.path.join(
        dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_trainlist".format(domain)]
    )
    test_listpath = os.path.join(
        dataset_path, "annotations", "labels_train_test", data_params_local["dataset_{}_testlist".format(domain)]
    )

    return data_path, train_listpath, test_listpath


class VideoDataset(Enum):
    EPIC = "EPIC"
    ADL = "ADL"
    GTEA = "GTEA"
    KITCHEN = "KITCHEN"

    @staticmethod
    def get_source_target(source: "VideoDataset", target: "VideoDataset", params):
        """
        Gets data loaders for source and target datasets

        Args:
            source: (VideoDataset): source dataset name
            target: (VideoDataset): target dataset name
            params: (CfgNode): hyper parameters from configure file

        Examples::
            >>> source, target, num_channel = get_source_target(sourcename, targetname, cfg)
        """
        config_params = get_videodata_config(params)
        data_params = config_params["data_params"]
        data_params_local = deepcopy(data_params)
        data_src_name = data_params_local["dataset_src_name"].upper()
        src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain="src")
        data_tar_name = data_params_local["dataset_tar_name"].upper()
        tar_data_path, tar_tr_listpath, tar_te_listpath = generate_list(data_tar_name, data_params_local, domain="tar")
        image_modality = data_params_local["dataset_image_modality"]
        n_classes = data_params_local["num_classes"]
        frames_per_segment = data_params_local["frames_per_segment"]

        if image_modality == "rgb":
            channel_numbers = {
                VideoDataset.EPIC: 3,
                VideoDataset.GTEA: 3,
                VideoDataset.ADL: 3,
                VideoDataset.KITCHEN: 3,
            }

            transform_names = {
                (VideoDataset.EPIC, 3): "epic",
                (VideoDataset.GTEA, 3): "gtea",
                (VideoDataset.ADL, 3): "adl",
                (VideoDataset.KITCHEN, 3): "kitchen",
            }
        elif image_modality == "flow":
            channel_numbers = {
                VideoDataset.EPIC: 2,
                VideoDataset.GTEA: 2,
                VideoDataset.ADL: 2,
                VideoDataset.KITCHEN: 2,
            }

            transform_names = {
                (VideoDataset.EPIC, 2): "epic",
                (VideoDataset.GTEA, 2): "gtea",
                (VideoDataset.ADL, 2): "adl",
                (VideoDataset.KITCHEN, 2): "kitchen",
            }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
            VideoDataset.GTEA: GTEADatasetAccess,
            VideoDataset.ADL: ADLDatasetAccess,
            VideoDataset.KITCHEN: KITCHENDatasetAccess,
        }

        # handle color/nb channels
        num_channels = max(channel_numbers[source], channel_numbers[target])
        source_tf = transform_names[(source, num_channels)]
        target_tf = transform_names[(target, num_channels)]

        return (
            factories[source](
                src_data_path,
                src_tr_listpath,
                src_te_listpath,
                image_modality,
                frames_per_segment,
                n_classes,
                source_tf,
            ),
            factories[target](
                tar_data_path,
                tar_tr_listpath,
                tar_te_listpath,
                image_modality,
                frames_per_segment,
                n_classes,
                target_tf,
            ),
            num_channels,
        )


class VideoDatasetAccess(DatasetAccess):
    """
    Common API for video dataset access

    Args:
        data_path (string): image directory of dataset
        train_list (string): training list file directory of dataset
        test_list (string): test list file directory of dataset
        image_modality (string): image type (RGB or Optical Flow)
        frames_per_segment (int): length of each action sample (the unit is number of frame)
        n_classes (int): number of class
        transform_kind (string): types of video transforms
    """

    def __init__(self, data_path, train_list, test_list, image_modality, frames_per_segment, n_classes, transform_kind):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._image_modality = image_modality
        self._frames_per_segment = frames_per_segment
        self._transform = video_transform.get_transform(transform_kind, self._image_modality)


class EPICDatasetAccess(VideoDatasetAccess):
    """EPIC data loader"""

    def get_train(self):
        return EPIC(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=True,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return EPIC(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class GTEADatasetAccess(VideoDatasetAccess):
    """GTEA data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class ADLDatasetAccess(VideoDatasetAccess):
    """ADL data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )


class KITCHENDatasetAccess(VideoDatasetAccess):
    """KITCHEN data loader"""

    def get_train(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._train_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["train"],
            random_shift=False,
            test_mode=False,
            image_modality=self._image_modality,
            dataset_split="train",
            n_classes=self._n_classes,
        )

    def get_test(self):
        return BasicVideoDataset(
            root_path=self._data_path,
            annotationfile_path=self._test_list,
            num_segments=1,
            frames_per_segment=self._frames_per_segment,
            imagefile_template="frame_{:010d}.jpg" if self._image_modality in ["rgb"] else "flow_{}_{:010d}.jpg",
            transform=self._transform["test"],
            random_shift=False,
            test_mode=True,
            image_modality=self._image_modality,
            dataset_split="test",
            n_classes=self._n_classes,
        )

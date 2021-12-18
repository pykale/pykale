# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
Action video dataset loading for EPIC-Kitchen, ADL, GTEA, KITCHEN. The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

import os
from copy import deepcopy
from enum import Enum

import torch

import kale.prepdata.video_transform as video_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.video_datasets import BasicVideoDataset, EPIC


def get_image_modality(image_modality):
    """Change image_modality (string) to rgb (bool) and flow (bool) for efficiency"""

    if image_modality == "joint":
        rgb = flow = True
    elif image_modality == "rgb" or image_modality == "flow":
        rgb = image_modality == "rgb"
        flow = image_modality == "flow"
    else:
        raise Exception("Invalid modality option: {}".format(image_modality))
    return rgb, flow


def get_videodata_config(cfg):
    """Get the configure parameters for video data from the cfg files"""

    config_params = {
        "data_params": {
            "dataset_root": cfg.DATASET.ROOT,
            "dataset_src_name": cfg.DATASET.SOURCE,
            "dataset_src_trainlist": cfg.DATASET.SRC_TRAINLIST,
            "dataset_src_testlist": cfg.DATASET.SRC_TESTLIST,
            "dataset_tgt_name": cfg.DATASET.TARGET,
            "dataset_tgt_trainlist": cfg.DATASET.TGT_TRAINLIST,
            "dataset_tgt_testlist": cfg.DATASET.TGT_TESTLIST,
            "dataset_image_modality": cfg.DATASET.IMAGE_MODALITY,
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
        raise ValueError("Wrong dataset name. Select from [EPIC, ADL, GTEA, KITCHEN]")

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
    def get_source_target(source: "VideoDataset", target: "VideoDataset", seed, params):
        """
        Gets data loaders for source and target datasets
        Sets channel_number as 3 for RGB, 2 for flow.
        Sets class_number as 8 for EPIC, 7 for ADL, 6 for both GTEA and KITCHEN.

        Args:
            source: (VideoDataset): source dataset name
            target: (VideoDataset): target dataset name
            seed: (int): seed value set manually.
            params: (CfgNode): hyper parameters from configure file

        Examples::
            >>> source, target, num_classes = get_source_target(source, target, seed, params)
        """
        config_params = get_videodata_config(params)
        data_params = config_params["data_params"]
        data_params_local = deepcopy(data_params)
        data_src_name = data_params_local["dataset_src_name"].upper()
        src_data_path, src_tr_listpath, src_te_listpath = generate_list(data_src_name, data_params_local, domain="src")
        data_tgt_name = data_params_local["dataset_tgt_name"].upper()
        tgt_data_path, tgt_tr_listpath, tgt_te_listpath = generate_list(data_tgt_name, data_params_local, domain="tgt")
        image_modality = data_params_local["dataset_image_modality"]
        frames_per_segment = data_params_local["frames_per_segment"]

        rgb, flow = get_image_modality(image_modality)

        transform_names = {
            VideoDataset.EPIC: "epic",
            VideoDataset.GTEA: "gtea",
            VideoDataset.ADL: "adl",
            VideoDataset.KITCHEN: "kitchen",
        }

        class_numbers = {
            VideoDataset.EPIC: 8,
            VideoDataset.GTEA: 6,
            VideoDataset.ADL: 7,
            VideoDataset.KITCHEN: 6,
        }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
            VideoDataset.GTEA: GTEADatasetAccess,
            VideoDataset.ADL: ADLDatasetAccess,
            VideoDataset.KITCHEN: KITCHENDatasetAccess,
        }

        # handle color/nb classes
        num_classes = min(class_numbers[source], class_numbers[target])
        source_tf = transform_names[source]
        target_tf = transform_names[target]

        rgb_source, rgb_target, flow_source, flow_target = [None] * 4

        if rgb:
            rgb_source = factories[source](
                src_data_path,
                src_tr_listpath,
                src_te_listpath,
                "rgb",
                frames_per_segment,
                num_classes,
                source_tf,
                seed,
            )
            rgb_target = factories[target](
                tgt_data_path,
                tgt_tr_listpath,
                tgt_te_listpath,
                "rgb",
                frames_per_segment,
                num_classes,
                target_tf,
                seed,
            )

        if flow:
            flow_source = factories[source](
                src_data_path,
                src_tr_listpath,
                src_te_listpath,
                "flow",
                frames_per_segment,
                num_classes,
                source_tf,
                seed,
            )
            flow_target = factories[target](
                tgt_data_path,
                tgt_tr_listpath,
                tgt_te_listpath,
                "flow",
                frames_per_segment,
                num_classes,
                target_tf,
                seed,
            )

        return (
            {"rgb": rgb_source, "flow": flow_source},
            {"rgb": rgb_target, "flow": flow_target},
            num_classes,
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
        seed: (int): seed value set manually.
    """

    def __init__(
        self, data_path, train_list, test_list, image_modality, frames_per_segment, n_classes, transform_kind, seed
    ):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._image_modality = image_modality
        self._frames_per_segment = frames_per_segment
        self._transform = video_transform.get_transform(transform_kind, self._image_modality)
        self._seed = seed

    def get_train_valid(self, valid_ratio):
        """Get the train and validation dataset with the fixed random split. This is used for joint input like RGB and
        optical flow, which will call `get_train_valid` twice. Fixing the random seed here can keep the seeds for twice
        the same."""
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - valid_ratio) * ntotal)
        return torch.utils.data.random_split(
            train_dataset, [ntrain, ntotal - ntrain], generator=torch.Generator().manual_seed(self._seed)
        )


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

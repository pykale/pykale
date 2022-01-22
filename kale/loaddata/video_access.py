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
from pathlib import Path

import pandas as pd
import torch

import kale.prepdata.video_transform as video_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.video_datasets import BasicVideoDataset, EPIC
from kale.loaddata.videos import VideoFrameDataset


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


def get_class_type(class_type):
    """Change class_type (string) to verb (bool) and noun (bool) for efficiency. Only noun is NA because we
    work on action recognition."""

    verb = True
    if class_type.lower() == "verb":
        noun = False
    elif class_type.lower() == "verb+noun":
        noun = True
    else:
        raise ValueError("Invalid class type option: {}".format(class_type))
    return verb, noun


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
            "dataset_input_type": cfg.DATASET.INPUT_TYPE,
            "dataset_class_type": cfg.DATASET.CLASS_TYPE,
            "dataset_num_segments": cfg.DATASET.NUM_SEGMENTS,
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
    EPIC100 = "EPIC100"

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
        input_type = data_params_local["dataset_input_type"]
        class_type = data_params_local["dataset_class_type"]
        num_segments = data_params_local["dataset_num_segments"]
        frames_per_segment = data_params_local["frames_per_segment"]

        rgb, flow = get_image_modality(image_modality)
        verb, noun = get_class_type(class_type)

        transform_names = {
            VideoDataset.EPIC: "epic",
            VideoDataset.GTEA: "gtea",
            VideoDataset.ADL: "adl",
            VideoDataset.KITCHEN: "kitchen",
            VideoDataset.EPIC100: None,
        }

        verb_class_numbers = {
            VideoDataset.EPIC: 8,
            VideoDataset.GTEA: 6,
            VideoDataset.ADL: 7,
            VideoDataset.KITCHEN: 6,
            VideoDataset.EPIC100: 97,
        }

        noun_class_numbers = {
            VideoDataset.EPIC100: 300,
        }

        factories = {
            VideoDataset.EPIC: EPICDatasetAccess,
            VideoDataset.GTEA: GTEADatasetAccess,
            VideoDataset.ADL: ADLDatasetAccess,
            VideoDataset.KITCHEN: KITCHENDatasetAccess,
            VideoDataset.EPIC100: EPIC100DatasetAccess,
        }

        rgb_source, rgb_target, flow_source, flow_target = [None] * 4
        num_verb_classes = num_noun_classes = None

        if verb:
            num_verb_classes = min(verb_class_numbers[source], verb_class_numbers[target])
        if noun:
            num_noun_classes = min(noun_class_numbers[source], noun_class_numbers[target])

        source_tf = transform_names[source]
        target_tf = transform_names[target]

        if input_type == "image":

            if rgb:
                rgb_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                rgb_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )

            if flow:
                flow_source = factories[source](
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                )
                flow_target = factories[target](
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                )

        elif input_type == "feature":
            # Input is feature vector, no need to use transform.
            if rgb:
                rgb_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

                rgb_target = factories[source](
                    domain="target",
                    data_path=tgt_data_path,
                    train_list=tgt_tr_listpath,
                    test_list=tgt_te_listpath,
                    image_modality="rgb",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=target_tf,
                    seed=seed,
                    input_type=input_type,
                )
            if flow:
                flow_source = factories[source](
                    domain="source",
                    data_path=src_data_path,
                    train_list=src_tr_listpath,
                    test_list=src_te_listpath,
                    image_modality="flow",
                    num_segments=num_segments,
                    frames_per_segment=frames_per_segment,
                    n_classes=num_verb_classes,
                    transform=source_tf,
                    seed=seed,
                    input_type=input_type,
                )

        else:
            raise Exception("Invalid input type option: {}".format(input_type))

        return (
            {"rgb": rgb_source, "flow": flow_source},
            {"rgb": rgb_target, "flow": flow_target},
            {"verb": num_verb_classes, "noun": num_noun_classes},
        )


class VideoDatasetAccess(DatasetAccess):
    """
    Common API for video dataset access

    Args:
        data_path (string): image directory of dataset
        train_list (string): training list file directory of dataset
        test_list (string): test list file directory of dataset
        image_modality (string): image type (RGB or Optical Flow)
        num_segments (int): number of segments the video should be divided into to sample frames from.
        frames_per_segment (int): length of each action sample (the unit is number of frame)
        n_classes (int): number of class
        transform (string): types of video transforms
        seed: (int): seed value set manually.
    """

    def __init__(
        self,
        data_path,
        train_list,
        test_list,
        image_modality,
        num_segments,
        frames_per_segment,
        n_classes,
        transform,
        seed,
    ):
        super().__init__(n_classes)
        self._data_path = data_path
        self._train_list = train_list
        self._test_list = test_list
        self._image_modality = image_modality
        self._num_segments = num_segments
        self._frames_per_segment = frames_per_segment
        self._transform = video_transform.get_transform(transform, self._image_modality)
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


class EPIC100DatasetAccess(VideoDatasetAccess):
    """EPIC-100 video feature data loader"""

    def __init__(
        self,
        domain,
        data_path,
        train_list,
        test_list,
        image_modality,
        num_segments,
        frames_per_segment,
        n_classes,
        transform,
        seed,
        input_type,
    ):
        super(EPIC100DatasetAccess, self).__init__(
            data_path,
            train_list,
            test_list,
            image_modality,
            num_segments,
            frames_per_segment,
            n_classes,
            transform,
            seed,
        )
        self._input_type = input_type
        self._domain = domain
        self._num_train_dataload = len(pd.read_pickle(self._train_list).index)
        self._num_test_dataload = len(pd.read_pickle(self._test_list).index)

    def get_train(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on train subset for EPIC 2021 challenge
            # data_path=Path(self._data_path, self._input_type, "{}_train.pkl".format(self._domain)),
            annotationfile_path=self._train_list,
            total_segments=25,
            num_segments=self._num_segments,  # 1
            frames_per_segment=self._frames_per_segment,  # 1
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=False,
            test_mode=True,
            input_type="feature",
            num_data_load=self._num_train_dataload,
        )

    def get_test(self):
        return VideoFrameDataset(
            root_path=Path(self._data_path, self._input_type, "{}_val.pkl".format(self._domain)),
            # Uncomment to run on test subset for EPIC 2021 challenge
            # data_path=Path(self._data_path, self._input_type, "{}_test.pkl".format(self._domain)),
            annotationfile_path=self._test_list,
            total_segments=25,
            num_segments=self._num_segments,
            frames_per_segment=self._frames_per_segment,
            image_modality=self._image_modality,
            imagefile_template="img_{:05d}.t7"
            if self._image_modality in ["RGB", "RGBDiff", "RGBDiff2", "RGBDiffplus"]
            else self._input_type + "{}_{:05d}.t7",
            random_shift=True,
            test_mode=True,
            input_type="feature",
            num_data_load=self._num_test_dataload,
        )

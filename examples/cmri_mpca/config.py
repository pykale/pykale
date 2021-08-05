"""
Default configurations for CMR processing and classification
"""

import os

import numpy as np
import pydicom
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "./"  # '/shared/tale2/Shared'
_C.DATASET.IMG_DIR = "DICOM"
_C.DATASET.LANDMARK_FILE = "landmarks.csv"
_C.DATASET.MASK_FILE = "Mask/phase1.dcm"
# ---------------------------------------------------------------------------- #
# Image processing
# ---------------------------------------------------------------------------- #
_C.PROC = CN()
_C.PROC.SCALE = 2

# ---------------------------------------------------------------------------- #
# MPCA Pipeline
# ---------------------------------------------------------------------------- #
_C.MPCA = CN()
_C.MPCA.CLF = ""  # choices=['CDAN', 'CDAN-E', 'DANN']

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir


def get_cfg_defaults():
    return _C.clone()


def read_dicom_imgs(dicom_path):
    sub_dirs = os.listdir(dicom_path)
    all_ds = []
    sub_ids = []
    for sub_dir in sub_dirs:
        sub_ds = []
        sub_path = os.path.join(dicom_path, sub_dir)
        phase_files = os.listdir(sub_path)
        for phase_file in phase_files:
            dataset = pydicom.dcmread(os.path.join(sub_path, phase_file))
            sub_ds.append(dataset)
        sub_ds.sort(key=lambda x: x.InstanceNumber, reverse=False)
        sub_ids.append(int(sub_ds[0].PatientID))
        all_ds.append(sub_ds)

    all_ds.sort(key=lambda x: int(x[0].PatientID), reverse=False)

    # for sub_ds in all_ds:
    n_sub = len(all_ds)
    n_phase = len(all_ds[0])
    img_shape = all_ds[0][0].pixel_array.shape
    imgs = np.zeros((n_sub, n_phase,) + img_shape)
    for i in range(n_sub):
        for j in range(n_phase):
            imgs[i, j, ...] = all_ds[i][j].pixel_array

    return imgs

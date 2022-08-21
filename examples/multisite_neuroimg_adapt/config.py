"""
Default configurations for classification on resting-state fMRI of ABIDE
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../data"
_C.DATASET.PIPELINE = "cpac"  # options: {‘cpac’, ‘css’, ‘dparsf’, ‘niak’}
_C.DATASET.ATLAS = "rois_cc200"
# options: {rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez, rois_ho, rois_tt}
_C.DATASET.SITE_IDS = None  # list of site ids to use, if None, use all sites
_C.DATASET.TARGET = "NYU"  # target site ids, e.g. "UM_1", "UCLA_1", "USM"
# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.KERNEL = "rbf"
_C.MODEL.ALPHA = 0.01
_C.MODEL.LAMBDA_ = 1.0
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir


def get_cfg_defaults():
    return _C.clone()

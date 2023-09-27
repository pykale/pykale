"""
Default configurations for classification on resting-state fMRI of ABIDE
"""

from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "../data"
_C.DATASET.PIPELINE = "cpac"  # options: {‘cpac’, ‘css’, ‘dparsf’, ‘niak’}
_C.DATASET.ATLAS = "rois_cc200"
# options: {rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez, rois_ho, rois_tt}
_C.DATASET.SITE_IDS = None  # list of site ids to use, if None, use all sites
_C.DATASET.TARGET = "NYU"  # target site ids, e.g. "UM_1", "UCLA_1", "USM"
# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.KERNEL = "rbf"
_C.MODEL.ALPHA = 0.01
_C.MODEL.LAMBDA_ = 1.0
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "./outputs"  # output_dir


def get_cfg_defaults():
    return _C.clone()

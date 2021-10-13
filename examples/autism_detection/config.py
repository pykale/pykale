"""
Default configurations for cardiac MRI data (ShefPAH) processing and classification
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
_C.DATASET.PIPELINE = "cpac"
_C.DATASET.ATLAS = "rois_cc200"
_C.DATASET.SITE_IDS = None
_C.DATASET.TARGET = "NYU"
# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.LOSS = "logits"
_C.MODEL.KERNEL = "rbf"
_C.MODEL.ALPHA = 0.01
_C.MODEL.LAMBDA_ = 1.0
_C.MODEL.LR = 0.0001
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir


def get_cfg_defaults():
    return _C.clone()

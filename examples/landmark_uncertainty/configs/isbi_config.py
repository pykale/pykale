"""
Default configurations for uncertainty estimation using Quantile Binning
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

_C.DATASET.SOURCE = "https://github.com/pykale/data/blob/landmark-data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip?raw=true"
_C.DATASET.ROOT = "../../../landmark_data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples"
_C.DATASET.BASE_DIR = "Uncertainty_tuples"

_C.DATASET.UNCERTAINTY_ERROR_PAIRS = [
    ["S-MHA", "S-MHA Error", "S-MHA Uncertainty"],
    ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"],
    ["E-CPV", "E-CPV Error", "E-CPV Uncertainty"],
]
_C.DATASET.FILE_FORMAT = "zip"

_C.DATASET.CONFIDENCE_INVERT = [["S-MHA", True], ["E-MHA", True], ["E-CPV", False]]

_C.DATASET.MODELS = ["U-NET"]
_C.DATASET.DATA = "ISBI"
_C.DATASET.LANDMARKS = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
_C.DATASET.NUM_FOLDS = 4


_C.DATASET.UE_PAIRS_VAL = "uncertainty_pairs_valid"
_C.DATASET.UE_PAIRS_TEST = "uncertainty_pairs_test"


# ---------------------------------------------------------------------------- #
# Uncertainty Estimation Pipeline Parameters
# ---------------------------------------------------------------------------- #
_C.PIPELINE = CN()

_C.PIPELINE.NUM_QUANTILE_BINS = 5


# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
_C.IM_KWARGS = CN()
_C.IM_KWARGS.cmap = "gray"

_C.MARKER_KWARGS = CN()
_C.MARKER_KWARGS.marker = "o"
_C.MARKER_KWARGS.markerfacecolor = (1, 1, 1, 0.1)
_C.MARKER_KWARGS.markeredgewidth = 1.5
_C.MARKER_KWARGS.markeredgecolor = "r"

_C.WEIGHT_KWARGS = CN()
_C.WEIGHT_KWARGS.markersize = 6
_C.WEIGHT_KWARGS.alpha = 0.7


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()


_C.OUTPUT.SAVE_FOLDER = "../../../outputs"
# _C.OUTPUT.SAVE_FILE_EVALUATION= "../../../outputs"


def get_cfg_defaults():
    return _C.clone()

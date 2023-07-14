"""
Default configurations for uncertainty estimation using Quantile Binning.
See the documentation for full information on the configuration file options:
pykale/examples/landmark_uncertainty/README.md
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
_C.DATASET.SOURCE = (
    "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"
)
_C.DATASET.ROOT = "../../../data/landmarks/"
_C.DATASET.BASE_DIR = "Uncertainty_tuples"


_C.DATASET.FILE_FORMAT = "zip"

_C.DATASET.CONFIDENCE_INVERT = [["S-MHA", True], ["E-MHA", True], ["E-CPV", False]]


_C.DATASET.DATA = "4CH"
_C.DATASET.LANDMARKS = [0, 1, 2]
_C.DATASET.NUM_FOLDS = 8
_C.DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE = True


_C.DATASET.UE_PAIRS_VAL = "uncertainty_pairs_valid"
_C.DATASET.UE_PAIRS_TEST = "uncertainty_pairs_test"


# ---------------------------------------------------------------------------- #
# Uncertainty Estimation Pipeline Parameters
# ---------------------------------------------------------------------------- #
_C.PIPELINE = CN()

# Can choose to evaluate over a single value or multiple values for Q (# bins). You can:
# 1) Evaluate over each value of Q (set COMPARE_INDIVIDUAL_Q = True). For each Q it will compare DATASET.MODELS and DATASET.UNCERTAINTY_ERROR_PAIRS against each other.
# 2) Compare results of a single model and a single uncertainty error pair (set COMPARE_Q_VALUES = True).

_C.PIPELINE.NUM_QUANTILE_BINS = [5, 10, 25]

# ~# 1)
# Compare uncertainty measures AND models over each single value of Q?
_C.PIPELINE.COMPARE_INDIVIDUAL_Q = True
# [NAME, KEY (error in csv), KEY (uncertainty in csv)]
_C.PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS = [
    ["S-MHA", "S-MHA Error", "S-MHA Uncertainty"],
    ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"],
    ["E-CPV", "E-CPV Error", "E-CPV Uncertainty"],
]
# Key for model name found in path.
_C.PIPELINE.INDIVIDUAL_Q_MODELS = ["U-NET", "PHD-NET"]
# ~#


# 2)
# Compare a single uncertainty measure on single model through various values of Q bins e.g. 5, 10, 20
_C.PIPELINE.COMPARE_Q_VALUES = True
_C.PIPELINE.COMPARE_Q_MODELS = ["PHD-NET"]  # Which model to compare over values of Q.
_C.PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS = [["E-MHA", "E-MHA Error", "E-MHA Uncertainty"]]
# ~#


_C.PIPELINE.COMBINE_MIDDLE_BINS = False
_C.PIPELINE.PIXEL_TO_MM_SCALE = 1.0
_C.PIPELINE.IND_LANDMARKS_TO_SHOW = [-1]  # -1 means show all landmarks individually, [] means show none
_C.PIPELINE.SHOW_IND_LANDMARKS = True

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

_C.BOXPLOT = CN()
_C.BOXPLOT.SAMPLES_AS_DOTS = True
_C.BOXPLOT.ERROR_LIM = 64
_C.BOXPLOT.SHOW_SAMPLE_INFO_MODE = "Average"  # "None", "All", "Average"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()


_C.OUTPUT.SAVE_FOLDER = "./outputs/"
_C.OUTPUT.SAVE_PREPEND = "example"
_C.OUTPUT.SAVE_FIGURES = True  # True to save, False to visualize in matplotlib


def get_cfg_defaults():
    return _C.clone()

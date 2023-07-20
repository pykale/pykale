"""
Default configurations for the AVMNIST dataset using Multimodal Neural Network (MMNN).
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
_C.DATASET.ROOT = "avmnist"
_C.DATASET.NAME = "avmnist.zip"
_C.DATASET.GDRIVE_ID = "1N5k-LvLwLbPBgn3GdVg6fXMBIR6pYrKb"
_C.DATASET.FILE_FORMAT = "zip"
_C.DATASET.BATCH_SIZE = 40
_C.DATASET.SHUFFLE = True
_C.DATASET.FLATTEN_AUDIO = True
_C.DATASET.FLATTEN_IMAGE = True
_C.DATASET.UNSQUEEZE_CHANNEL = False
_C.DATASET.NORMALIZE_IMAGE = False
_C.DATASET.NORMALIZE_AUDIO = False


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MAX_EPOCHS = 1
_C.SOLVER.EARLY_STOP = True
_C.SOLVER.CLIP_VAL = 8

# ---------------------------------------------------------------------------- #
# Model configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.LENET_IN_CHANNELS = 1
_C.MODEL.LENET_ADD_LAYERS_IMG = 3
_C.MODEL.LENET_ADD_LAYERS_AUD = 5
_C.MODEL.CHANNELS = 6
_C.MODEL.MLP_IN_DIM = _C.MODEL.CHANNELS * 40
_C.MODEL.MLP_LOW_RANK_IN_DIM = _C.MODEL.CHANNELS * 20
_C.MODEL.MLP_HIDDEN_DIM = 100
_C.MODEL.OUT_DIM = 2
_C.MODEL.FUSION = "late"
_C.MODEL.MULTIPLICATIVE_FUSION_IN_DIM = [_C.MODEL.CHANNELS * 8, _C.MODEL.CHANNELS * 32]
_C.MODEL.MULTIPLICATIVE_FUSION_OUT_DIM = _C.MODEL.CHANNELS * 40
_C.MODEL.MULTIPLICATIVE_OUTPUT = "matrix"
_C.MODEL.LOW_RANK_TENSOR_IN_DIM = [_C.MODEL.CHANNELS * 8, _C.MODEL.CHANNELS * 32]
_C.MODEL.LOW_RANK_TENSOR_OUT_DIM = _C.MODEL.CHANNELS * 20
_C.MODEL.LOW_RANK_TENSOR_RANK = 40


# -----------------------------------------------------------------------------
# Comet Logger (optional) - https://www.comet.ml/site/
# -----------------------------------------------------------------------------
_C.COMET = CN()
_C.COMET.ENABLE = False  # Set True to enable Comet logging (requires an API key).
_C.COMET.API_KEY = ""  # Your Comet API key
_C.COMET.PROJECT_NAME = "Multimodal Neural Network"
_C.COMET.EXPERIMENT_NAME = "AVMNIST_MMNN"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.OUT_DIR = "./outputs"
_C.OUTPUT.PB_FRESH = 50  # Number of steps before a new progress bar is printed. Set 0 to disable the progress bar.


def get_cfg_defaults():
    return _C.clone()

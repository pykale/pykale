"""
Default configurations for domain adaptation
"""

import os

from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "./data"  # Root directory of dataset, "data" is in the same directory as this file
_C.DATASET.NAME = "digits"  # Dataset type name
_C.DATASET.SOURCE = "mnist"  # The source dataset name
_C.DATASET.TARGET = "usps"  # The target dataset name
_C.DATASET.NUM_CLASSES = 10  # Number of classes in the dataset
_C.DATASET.NUM_REPEAT = 10  # Number of times to repeat the experiment
_C.DATASET.DIMENSION = 784
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "source"
_C.DATASET.VALID_SPLIT_RATIO = 0.1  # Ratio of validation set to the training dataset

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 120
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20
_C.SOLVER.NUM_WORKERS = 1
_C.SOLVER.TRAIN_BATCH_SIZE = 150
_C.SOLVER.TEST_BATCH_SIZE = 200  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True  # Set True to use adaptive lambda
_C.SOLVER.AD_LR = True  # Set True to use adaptive learning rate
_C.SOLVER.INIT_LAMBDA = 1.0  # Initial value of lambda

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CfgNode()
_C.DAN.METHOD = "CDAN"  # choices=['CDAN', 'CDAN-E', 'DANN', 'DAN', 'JAN']
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.PB_FRESH = 0  # Number of steps before a new progress bar is printed. Set 0 to disable the progress bar
_C.OUTPUT.OUT_DIR = os.path.join("outputs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)

# -----------------------------------------------------------------------------
# Comet Logger (optional) - https://www.comet.ml/site/
# -----------------------------------------------------------------------------
_C.COMET = CfgNode()
_C.COMET.ENABLE = False  # Set True to enable Comet logging (requires an API key).
_C.COMET.API_KEY = ""  # Your Comet API key
_C.COMET.PROJECT_NAME = "Digit DANN"
_C.COMET.EXPERIMENT_NAME = "DigitDANN"


def get_cfg_defaults():
    return _C.clone()

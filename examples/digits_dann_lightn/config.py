"""
Default configurations for domain adaptation
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../data"  # Path to store data and results
_C.DATASET.NAME = "digits"  # Name of datasets
_C.DATASET.SOURCE = "mnist"  # The source dataset name
_C.DATASET.TARGET = "usps"  # The target dataset name
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NUM_REPEAT = 10
_C.DATASET.DIMENSION = 784
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "source"
_C.DATASET.VALID_SPLIT_RATIO = 0.1
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.001  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 120  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 150  # 150
_C.SOLVER.TEST_BATCH_SIZE = 200  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1.0

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = "CDAN"  # choices=['CDAN', 'CDAN-E', 'DANN']
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.TB_DIR = os.path.join("lightning_logs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)


def get_cfg_defaults():
    return _C.clone()

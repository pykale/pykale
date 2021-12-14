"""
Default configurations for action recognition domain adaptation
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
_C.DATASET.ROOT = "I:/Datasets/EgoAction/"  # "/shared/tale2/Shared"
_C.DATASET.SOURCE = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.SRC_TRAINLIST = "epic_D1_train.pkl"
_C.DATASET.SRC_TESTLIST = "epic_D1_test.pkl"
_C.DATASET.TARGET = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.TGT_TRAINLIST = "epic_D2_train.pkl"
_C.DATASET.TGT_TESTLIST = "epic_D2_test.pkl"
_C.DATASET.IMAGE_MODALITY = "rgb"  # mode options=["rgb", "flow", "joint"]
# _C.DATASET.NUM_CLASSES = 8
_C.DATASET.FRAMES_PER_SEGMENT = 16
_C.DATASET.NUM_REPEAT = 5  # 10
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "max"  # options=["source", "max"]
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 30  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 5  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 16  # 150
# _C.SOLVER.TEST_BATCH_SIZE = 32  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1.0

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.METHOD = "i3d"  # options=["r3d_18", "r2plus1d_18", "mc3_18", "i3d"]
_C.MODEL.ATTENTION = "None"  # options=["None", "SELayer"]

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = "CDAN"  # options=["CDAN", "CDAN-E", "DANN", "DAN"]
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.TB_DIR = os.path.join("lightning_logs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)


def get_cfg_defaults():
    return _C.clone()

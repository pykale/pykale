"""
Default configurations for domain adapation
"""

from yacs.config import CfgNode as CN
import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = 'I:/Datasets/EgoAction/'  # '/shared/tale2/Shared'
# _C.DATASET.NAME = 'EPIC'  # dset choices=['office', 'image-clef', 'office-home']
_C.DATASET.SOURCE = 'EPIC'  # s_dset_path  , help="The source dataset path list"
_C.DATASET.SRC_TRAINLIST = 'D1_train.pkl'
_C.DATASET.SRC_TESTLIST = 'D1_test.pkl'
_C.DATASET.TARGET = 'EPIC'  # s_dset_path  , help="The target dataset path list"
_C.DATASET.TAR_TRAINLIST = 'D2_train.pkl'
_C.DATASET.TAR_TESTLIST = 'D2_test.pkl'
_C.DATASET.MODE = 'rgb'  # mode choices=['rgb', 'flow', 'rgb+flow']
_C.DATASET.NUM_CLASSES = 8
_C.DATASET.WINDOW_LEN = 16
_C.DATASET.NUM_REPEAT = 5  # 10
# _C.DATASET.DIMENSION = 784
_C.DATASET.WEIGHT_TYPE = 'natural'
_C.DATASET.SIZE_TYPE = 'source'
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
_C.SOLVER.MAX_EPOCHS = 100  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 16  # 150
_C.SOLVER.TEST_BATCH_SIZE = 32  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.METHOD = "r3d_18"

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = 'CDAN'  # choices=['CDAN', 'CDAN-E', 'DANN']
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = './outputs'  # output_dir
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.DIR = os.path.join(_C.OUTPUT.ROOT, _C.DATASET.SOURCE + '2' + _C.DATASET.TARGET)


def get_cfg_defaults():
    return _C.clone()

"""
Default configurations for image classification using ISONet,
based on https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/config.py
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
_C.DATASET.NAME = "CIFAR10"
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NUM_WORKERS = 1

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.05
_C.SOLVER.LR_MILESTONES = [30, 60, 90]
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.DAMPENING = False
_C.SOLVER.NESTEROV = False

_C.SOLVER.TRAIN_BATCH_SIZE = 128
_C.SOLVER.TEST_BATCH_SIZE = 200

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.WARMUP = False
_C.SOLVER.WARMUP_EPOCH = 5
_C.SOLVER.WARMUP_FACTOR = 0.2
# ---------------------------------------------------------------------------- #
# ISONet configs
# ---------------------------------------------------------------------------- #
_C.ISON = CN()
_C.ISON.DEPTH = 34
_C.ISON.ORTHO_COEFF = 1e-4
_C.ISON.HAS_BN = False
_C.ISON.HAS_ST = False
_C.ISON.SReLU = True
_C.ISON.DIRAC_INIT = True
_C.ISON.HAS_RES_MULTIPLIER = False
_C.ISON.RES_MULTIPLIER = 1.0
_C.ISON.DROPOUT = False
_C.ISON.DROPOUT_RATE = 0.0

_C.ISON.TRANS_FUN = "basic_transform"


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return _C.clone()

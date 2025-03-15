"""
Default configurations for multi-source domain adapation
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
_C.DATASET.ROOT = "../data"
_C.DATASET.NAME = "digits"  # choices=['office', 'digits', 'office_caltech', 'office31']
_C.DATASET.TARGET = "MNIST"
# -----------------------------------------------------------------------------
_C.DATASET.SOURCE = None
# a list of source domain names (e.g. ["SVHN", "USPS_RGB"]) or None. If None, all domains (excluding the target)
# will be used as sources
# -----------------------------------------------------------------------------
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NUM_REPEAT = 10  # 10
_C.DATASET.NUM_CHANNELS = 3
_C.DATASET.DIMENSION = 784
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "source"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2025
_C.SOLVER.BASE_LR = 0.001  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 120  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 100
_C.SOLVER.TEST_BATCH_SIZE = 100
_C.SOLVER.LOG_EVERY_N_STEPS = 10
_C.SOLVER.NUM_WORKERS = 1

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CfgNode()
_C.DAN.METHOD = "M3SDA"  # choices=['M3SDA', 'MFSAN']
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "./outputs"  # output_dir
_C.OUTPUT.VERBOSE = False
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option


def get_cfg_defaults():
    return _C.clone()

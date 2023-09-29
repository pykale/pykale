"""
Default configurations for prototypical networks
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 1397
_C.DEVICE = "cuda"
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "Data/omniglot/"
# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet18"
_C.MODEL.PRETRAIN_WEIGHTS = None
# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.OPTIMIZER = "SGD"
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.N_WAYS = 30
_C.TRAIN.K_SHOTS = 5
_C.TRAIN.K_QUERIES = 15
# ---------------------------------------------------------------------------- #
# Val
# ---------------------------------------------------------------------------- #
_C.VAL = CN()
_C.VAL.N_WAYS = 5
_C.VAL.K_SHOTS = 5
_C.VAL.K_QUERIES = 15
# ---------------------------------------------------------------------------- #
# Logger
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.LOG_DIR = "logs"
_C.OUTPUT.WEIGHT_DIR = "weights"
_C.OUTPUT.SAVE_FREQ = 1
_C.OUTPUT.SAVE_TOP_K = 2
_C.OUTPUT.SAVE_LAST = True


def get_cfg_defaults():
    return _C.clone()

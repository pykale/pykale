"""
Default configurations for prototypical networks
"""
from yacs.config import CfgNode

# ---------------------------------------------------------------------------- #
# Environment settings
# ---------------------------------------------------------------------------- #
_C = CfgNode()
_C.SEED = 1397
_C.GPUS = 1
# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "data/omniglot/"
_C.DATASET.IMG_SIZE = 84
# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.BACKBONE = "ResNet18Feature"
_C.MODEL.PRETRAIN_WEIGHTS = None
# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.OPTIMIZER = "SGD"
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.NUM_CLASSES = 30
_C.TRAIN.NUM_SUPPORT_SAMPLES = 5
_C.TRAIN.NUM_QUERY_SAMPLES = 15
# ---------------------------------------------------------------------------- #
# Validation and Test
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()
_C.VAL.NUM_CLASSES = 5
_C.VAL.NUM_SUPPORT_SAMPLES = 5
_C.VAL.NUM_QUERY_SAMPLES = 15
# ---------------------------------------------------------------------------- #
# Logger
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "outputs"
_C.OUTPUT.LOG_DIR = "logs"
_C.OUTPUT.WEIGHT_DIR = "weights"
_C.OUTPUT.SAVE_FREQ = 1
_C.OUTPUT.SAVE_TOP_K = 1
_C.OUTPUT.SAVE_LAST = True


def get_cfg_defaults():
    return _C.clone()

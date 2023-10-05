"""
Default configurations for prototypical networks
"""
from yacs.config import CfgNode

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CfgNode()
_C.SEED = 1397
_C.DEVICE = "cuda"
# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "Data/omniglot/"
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
_C.TRAIN.N_WAYS = 30
_C.TRAIN.K_SHOTS = 5
_C.TRAIN.K_QUERIES = 15
# ---------------------------------------------------------------------------- #
# Val
# ---------------------------------------------------------------------------- #
_C.VAL = CfgNode()
_C.VAL.N_WAYS = 5
_C.VAL.K_SHOTS = 5
_C.VAL.K_QUERIES = 15
# ---------------------------------------------------------------------------- #
# Logger
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "examples/protonet/outputs/"
_C.OUTPUT.LOG_DIR = "logs"
_C.OUTPUT.WEIGHT_DIR = "weights"
_C.OUTPUT.SAVE_FREQ = 1
_C.OUTPUT.SAVE_TOP_K = 2
_C.OUTPUT.SAVE_LAST = True


def get_cfg_defaults():
    return _C.clone()

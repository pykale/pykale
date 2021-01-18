"""
Hyperparameter configuration file based on the YACS library.
"""
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../data"
_C.DATASET.NAME = "CIFAR10"
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NUM_WORKERS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.BASE_LR = 0.05
_C.SOLVER.LR_MILESTONES = [30, 60, 90]
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.TRAIN_BATCH_SIZE = 128
_C.SOLVER.TEST_BATCH_SIZE = 200

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.WARMUP = False
_C.SOLVER.WARMUP_EPOCHS = 5

# ---------------------------------------------------------------------------- #
# CNN configs
# ---------------------------------------------------------------------------- #
_C.CNN = CN()
_C.CNN.POOL_LOCATIONS = (0, 3)  # After which index of the below
# convolutionial-layer list pooling
# layers should be placed. (0,3) Applies
# 2 pooling layers, resulting in an image
# size of 8x8.

# A tuple for each convolutional layer given as (num_channels, kernel_size)
# (Nested lists log to file prettier than nested tuples do)
_C.CNN.CONV_LAYERS = [[16, 3], [32, 3], [64, 3], [32, 1], [64, 3], [128, 3], [256, 3], [64, 1]]
_C.CNN.USE_BATCHNORM = True
_C.CNN.ACTIVATION_FUN = "relu"  # one of ('relu', 'elu', 'leaky_relu')
_C.CNN.OUTPUT_SHAPE = (-1, 64, 8, 8)
# ---------------------------------------------------------------------------- #
# Transformer configs
# ---------------------------------------------------------------------------- #
_C.TRANSFORMER = CN()
# Will not attach the Transformer on top of the CNN in the model if False
_C.TRANSFORMER.USE_TRANSFORMER = True
_C.TRANSFORMER.NUM_LAYERS = 3
_C.TRANSFORMER.NUM_HEADS = 2
_C.TRANSFORMER.DIM_FEEDFORWARD = 128
_C.TRANSFORMER.DROPOUT = 0.1
_C.TRANSFORMER.OUTPUT_TYPE = "spatial"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return _C.clone()

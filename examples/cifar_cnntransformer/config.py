"""
Hyperparameter configuration file based on the YACS library.
"""
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "./data"  # Root directory of dataset, "data" is in the same directory as this file
_C.DATASET.NAME = "CIFAR10"  # Dataset name
_C.DATASET.NUM_CLASSES = 10  # Number of classes in the dataset
_C.DATASET.NUM_WORKERS = 0  # Number of workers for data loading

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.BASE_LR = 0.05
_C.SOLVER.AD_LR = True  # default to enable multi-step learning rate decay
_C.SOLVER.LR_MILESTONES = [30, 60, 90]
_C.SOLVER.LR_GAMMA = 0.1

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.TRAIN_BATCH_SIZE = 128
_C.SOLVER.TEST_BATCH_SIZE = 200

# ---------------------------------------------------------------------------- #
# CNN configs
# ---------------------------------------------------------------------------- #
_C.CNN = CN()

# A list for each convolutional layer is given as (num_channels, kernel_size). Nested lists log to file prettier
# than nested tuples do.
_C.CNN.CONV_LAYERS = [[16, 3], [32, 3], [64, 3], [32, 1], [64, 3], [128, 3], [256, 3], [64, 1]]

# The index after which pooling layers should be positioned in the convolutional layer list. (0,3) applies 2 pooling
# layers, resulting in an image size of 8x8.
_C.CNN.POOL_LOCATIONS = (0, 3)

_C.CNN.USE_BATCHNORM = True
_C.CNN.ACTIVATION_FUN = "relu"  # one of ('relu', 'elu', 'leaky_relu')
_C.CNN.OUTPUT_SHAPE = (-1, 64, 8, 8)

# ---------------------------------------------------------------------------- #
# Transformer configs
# ---------------------------------------------------------------------------- #
_C.TRANSFORMER = CN()
_C.TRANSFORMER.USE_TRANSFORMER = True  # Set False to use the CNN only
_C.TRANSFORMER.NUM_LAYERS = 3
_C.TRANSFORMER.NUM_HEADS = 2
_C.TRANSFORMER.DIM_FEEDFORWARD = 128
_C.TRANSFORMER.DROPOUT = 0.1
_C.TRANSFORMER.OUTPUT_TYPE = "spatial"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.PB_FRESH = 50  # Number of steps before a new progress bar is printed. Set 0 to disable the progress bar.
_C.OUTPUT.OUT_DIR = "./outputs"  # Directory to save outputs to (logs, checkpoints, etc.)

# -----------------------------------------------------------------------------
# Comet Logger (optional) - https://www.comet.ml/site/
# -----------------------------------------------------------------------------
_C.COMET = CN()
_C.COMET.ENABLE = False  # Set True to enable Comet logging (requires an API key).
_C.COMET.API_KEY = ""  # Your Comet API key
_C.COMET.PROJECT_NAME = "CNN Transformer"
_C.COMET.EXPERIMENT_NAME = "CNNTransformer"


def get_cfg_defaults():
    return _C.clone()

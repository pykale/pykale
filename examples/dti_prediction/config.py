from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = "DAVIS"
_C.DATASET.Y_LOG = True

# -----------------------------------------------------------------------------
# Model component
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DRUG_ENCODER = "CNN"
_C.MODEL.DRUG_DIM = 128
_C.MODEL.TARGET_DIM = 128
_C.MODEL.TARGET_ENCODER = "CNN"
_C.MODEL.DECODER = "MLP"
_C.MODEL.DRUG_LENGTH = 85
_C.MODEL.TARGET_LENGTH = 1200
_C.MODEL.NUM_FILTERS = 32  # for cnn only
_C.MODEL.DRUG_FILTER_LENGTH = 8  # for drug cnn only
_C.MODEL.TARGET_FILTER_LENGTH = 8  # for target cnn only
_C.MODEL.DECODER_IN_DIM = 192  # for mlp only, the concat of drug decoder output and target decode output
_C.MODEL.MLP_HIDDEN_DIM = 1024
_C.MODEL.MLP_OUT_DIM = 512
_C.MODEL.MLP_DROPOUT = 0.1

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.LR = 0.001
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.TRAIN_BATCH_SIZE = 256
_C.SOLVER.TEST_BATCH_SIZE = 256


def get_cfg_defaults():
    return _C.clone()

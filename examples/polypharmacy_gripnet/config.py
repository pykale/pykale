"""
Default configurations for Polypharmacy Side Effect Prediction using GripNet
"""
from yacs.config import CfgNode

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------

_C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "./data"
_C.DATASET.NAME = "pose"
_C.DATASET.URL = "https://github.com/pykale/data/raw/main/graphs/pose_pyg_2.pt"

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 1111
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.EPSILON = 1e-10
_C.SOLVER.MAX_EPOCHS = 66
_C.SOLVER.LOG_EVERY_N_STEPS = 1

# ---------------------------------------------------------
# GripNet supervertex configs
# ---------------------------------------------------------
_C.GRIPN_SV1 = CfgNode()
_C.GRIPN_SV1.NAME = "protein"
_C.GRIPN_SV1.INTER_FEAT_CHANNELS = 16
_C.GRIPN_SV1.INTER_AGG_CHANNELS_LIST = [16, 16]
_C.GRIPN_SV1.EXTER_AGG_CHANNELS_LIST = []
# Elements in `EXTER_AGG_CHANNELS_LIST` should be a list of combinations of supervertex name (str) and embedding
#   dimension (int). If the supervertex is a root supervertex, it should be an empty list.
_C.GRIPN_SV1.MODE = ""  # `MODE` is either "cat" or "add"

_C.GRIPN_SV2 = CfgNode()
_C.GRIPN_SV2.NAME = "drug"
_C.GRIPN_SV2.INTER_FEAT_CHANNELS = 32
_C.GRIPN_SV2.INTER_AGG_CHANNELS_LIST = [16, 16]
_C.GRIPN_SV2.EXTER_AGG_CHANNELS_LIST = [["protein", 16]]
_C.GRIPN_SV2.MODE = "cat"

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "./outputs"


def get_cfg_defaults():
    return _C.clone()

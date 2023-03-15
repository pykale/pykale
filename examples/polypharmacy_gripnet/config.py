from typing import List, Tuple

from yacs.config import CfgNode

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------

C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = "./data"
C.DATASET.NAME = "pose"
C.DATASET.URL = "https://github.com/pykale/data/raw/main/graphs/pose_pyg_2.pt"

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
C.SOLVER = CfgNode()
C.SOLVER.SEED = 1111
C.SOLVER.BASE_LR = 0.01
C.SOLVER.EPSILON = 1e-10
C.SOLVER.MAX_EPOCHS = 66
C.SOLVER.LOG_EVERY_N_STEPS = 1

# ---------------------------------------------------------
# GripNet supervertex configs
# ---------------------------------------------------------
C.GRIPN_SV1 = CfgNode()
C.GRIPN_SV1.NAME = "protein"
C.GRIPN_SV1.INTER_FEAT_CHANNELS = 16
C.GRIPN_SV1.INTER_AGG_CHANNELS_LIST = [16, 16]
C.GRIPN_SV1.EXTER_AGG_CHANNELS_LIST = []
# Elements in `EXTER_AGG_CHANNELS_LIST` should be a list of combinations of supervertex name (str) and embedding
#   dimension (int). If the supervertex is a root supervertex, it should be an empty list.
C.GRIPN_SV1.MODE = ""  # `MODE` is either "cat" or "add"

C.GRIPN_SV2 = CfgNode()
C.GRIPN_SV2.NAME = "drug"
C.GRIPN_SV2.INTER_FEAT_CHANNELS = 32
C.GRIPN_SV2.INTER_AGG_CHANNELS_LIST = [16, 16]
C.GRIPN_SV2.EXTER_AGG_CHANNELS_LIST = [["protein", 16]]
C.GRIPN_SV2.MODE = "cat"

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return C.clone()

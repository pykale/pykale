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
C.SOLVER.MAX_EPOCHS = 66
C.SOLVER.LOG_EVERY_N_STEPS = 1

# ---------------------------------------------------------
# GripNet supervertex configs
# ---------------------------------------------------------
C.GRIPN_SV1 = CfgNode()
C.GRIPN_SV1.NAME = "gene"
C.GRIPN_SV1.INTER_FEAT_CHANNELS = 5
C.GRIPN_SV1.INTER_AGG_CHANNELS_LIST = [4, 4]
C.GRIPN_SV1.EXTER_AGG_CHANNELS_LIST = []
C.GRIPN_SV1.MODE = ""

C.GRIPN_SV2 = CfgNode()
C.GRIPN_SV2.NAME = "drug"
C.GRIPN_SV2.INTER_FEAT_CHANNELS = 7
C.GRIPN_SV2.INTER_AGG_CHANNELS_LIST = [6, 6]
C.GRIPN_SV2.EXTER_AGG_CHANNELS_LIST = [["gene", 7]]
C.GRIPN_SV2.MODE = "cat"

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return C.clone()

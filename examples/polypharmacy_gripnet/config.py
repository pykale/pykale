from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = "./data"
C.DATASET.NAME = "PoSE"
C.DATASET.URL = "https://github.com/pykale/data/raw/main/graphs/pose.pt"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2020
C.SOLVER.BASE_LR = 0.01
C.SOLVER.LR_MILESTONES = [30, 60, 90]
C.SOLVER.LR_GAMMA = 0.1
C.SOLVER.MAX_EPOCHS = 5
C.SOLVER.WARMUP = False
C.SOLVER.WARMUP_EPOCHS = 100

# ---------------------------------------------------------------------------- #
# GripNet configs
# ---------------------------------------------------------------------------- #
C.GRIPN = CfgNode()
C.GRIPN.GG_LAYERS = [32, 16, 16]
C.GRIPN.GD_LAYERS = [16, 32]
C.GRIPN.DD_LAYERS = [sum(C.GRIPN.GD_LAYERS), 16]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return C.clone()

from yacs.config import CfgNode

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------

C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = "dataset/"
C.DATASET.NAME = "TCGA_BRCA"
C.DATASET.URL = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"
C.DATASET.RANDOM_SPLIT = False
C.DATASET.NUM_VIEW = 3
C.DATASET.NUM_CLASS = 5

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2023
C.SOLVER.MAX_EPOCHS_PRETRAIN = 500
C.SOLVER.MAX_EPOCHS = 2500

# -----------------------------------------------------------------------------
# Model (MOGONET) configs
# -----------------------------------------------------------------------------
C.MODEL = CfgNode()
C.MODEL.EDGE_PER_NODE = 10
C.MODEL.EQUAL_WEIGHT = False
C.MODEL.GCN_LR_PRETRAIN = 1e-3
C.MODEL.GCN_LR = 5e-4
C.MODEL.GCN_DROPOUT_RATE = 0.5
C.MODEL.GCN_HIDDEN_DIM = [400, 400, 200]
C.MODEL.VCDN_LR = 1e-3

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return C.clone()

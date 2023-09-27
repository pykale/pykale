"""
Default configurations according to the MOGONET method described in 'MOGONET integrates multi-omics data using
graph convolutional networks allowing patient classification and biomarker identification'
- Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021).

https://github.com/txWang/MOGONET/blob/main/main_mogonet.py
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
_C.DATASET.ROOT = "dataset/"
_C.DATASET.NAME = "TCGA_BRCA"
_C.DATASET.URL = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"
_C.DATASET.RANDOM_SPLIT = False
_C.DATASET.NUM_MODALITIES = 3  # Number of omics modalities in the dataset
_C.DATASET.NUM_CLASSES = 5

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.SEED = 2023
_C.SOLVER.MAX_EPOCHS_PRETRAIN = 500
_C.SOLVER.MAX_EPOCHS = 2500

# -----------------------------------------------------------------------------
# Model (MOGONET) configs
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()
_C.MODEL.EDGE_PER_NODE = 10  # Predefined number of edges per nodes in computing adjacency matrix
_C.MODEL.EQUAL_WEIGHT = False
_C.MODEL.GCN_LR_PRETRAIN = 1e-3
_C.MODEL.GCN_LR = 5e-4
_C.MODEL.GCN_DROPOUT_RATE = 0.5
_C.MODEL.GCN_HIDDEN_DIM = [400, 400, 200]

# The View Correlation Discovery Network (VCDN) to learn the higher-level intra-view and cross-view correlations
# in the label space. See the MOGONET paper for more information.
_C.MODEL.VCDN_LR = 1e-3

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
_C.OUTPUT = CfgNode()
_C.OUTPUT.OUT_DIR = "./outputs"


def get_cfg_defaults():
    return _C.clone()

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

C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = "dataset/"
C.DATASET.NAME = "TCGA_BRCA"
C.DATASET.URL = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"
C.DATASET.RANDOM_SPLIT = False
C.DATASET.NUM_MODALITIES = 3  # Number of omics modalities in the dataset
C.DATASET.NUM_CLASSES = 5

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
C.MODEL.EDGE_PER_NODE = 10  # Predefined number of edges per nodes in computing adjacency matrix
C.MODEL.EQUAL_WEIGHT = False
C.MODEL.GCN_LR_PRETRAIN = 1e-3
C.MODEL.GCN_LR = 5e-4
C.MODEL.GCN_DROPOUT_RATE = 0.5
C.MODEL.GCN_HIDDEN_DIM = [400, 400, 200]

# The View Correlation Discovery Network (VCDN) to learn the higher-level intra-view and cross-view correlations
# in the label space. See the MOGONET paper for more information.
C.MODEL.VCDN_LR = 1e-3

# ---------------------------------------------------------
# Misc options
# ---------------------------------------------------------
C.OUTPUT_DIR = "./outputs"


def get_cfg_defaults():
    return C.clone()

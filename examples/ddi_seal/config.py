from yacs.config import CfgNode as CN
import os

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "dataset/ogbl_ddi"
_C.DATASET.NAME = 'ogbl-ddi'
_C.DATASET.NUM_HOPS = 1
_C.DATASET.TRAIN_PERCENT = 1.0
_C.DATASET.VAL_PERCENT = 100
_C.DATASET.TEST_PERCENT = 100
_C.DATASET.COALESCE = False
_C.DATASET.NODE_LABEL = 'drnl'
_C.DATASET.RATIO_PER_HOP = 0.2
_C.DATASET.MAX_NODES_PER_HOP = False
_C.DATASET.BATCH_SIZE = 32
_C.DATASET.NUM_WORKERS = 8
_C.DATASET.MAX_Z = 1000 # set a large max_z so that every z has embeddings to look up

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020

# -----------------------------------------------------------------------------
# SEAL
# -----------------------------------------------------------------------------
_C.SEAL.MODEL = 'DGCNN' # for downstream gnn model selection, support DGCNN/SAGE


def get_cfg_defaults():
    """Clone config"""
    return _C.clone()


def dump_cfg(cfg):
    """Dump config to the output directory"""
    cfg_file = os.path.join(cfg.output_dir, cfg.cfg_dest)
    with open(cfg_file, 'w') as wf:
        cfg.dump(stream=wf)

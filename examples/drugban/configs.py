from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Drug feature extractor
# ---------------------------------------------------------------------------- #
_C.DRUG = CfgNode()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.MAX_NODES = 290

# ---------------------------------------------------------------------------- #
# Protein feature extractor
# ---------------------------------------------------------------------------- #
_C.PROTEIN = CfgNode()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]
_C.PROTEIN.EMBEDDING_DIM = 128
_C.PROTEIN.PADDING = True

# ---------------------------------------------------------------------------- #
# BCN setting
# ---------------------------------------------------------------------------- #
_C.BCN = CfgNode()
_C.BCN.HEADS = 2

# ---------------------------------------------------------------------------- #
# MLP decoder
# ---------------------------------------------------------------------------- #
_C.DECODER = CfgNode()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 128
_C.DECODER.BINARY = 1

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.LR = 5e-5
_C.SOLVER.DA_LR = 1e-3
_C.SOLVER.SEED = 2048

# ---------------------------------------------------------------------------- #
# RESULT
# ---------------------------------------------------------------------------- #
_C.RESULT = CfgNode()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True

# ---------------------------------------------------------------------------- #
# Domain adaptation
# ---------------------------------------------------------------------------- #
_C.DA = CfgNode()
_C.DA.TASK = False          # False: 'in-domain' splitting strategy, True: 'cross-domain' splitting strategy
_C.DA.METHOD = "CDAN"
_C.DA.USE = False           # False: no domain adaptation, True: domain adaptation
_C.DA.INIT_EPOCH = 10
_C.DA.LAMB_DA = 1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.RANDOM_DIM = None
_C.DA.USE_ENTROPY = True

# ---------------------------------------------------------------------------- #
# Comet config, ignore it If not installed.
# ---------------------------------------------------------------------------- #
_C.COMET = CfgNode()
# Please change to your own workspace name on comet.
_C.COMET.WORKSPACE = "pz-white"
_C.COMET.PROJECT_NAME = "DrugBAN"
_C.COMET.USE = False
_C.COMET.TAG = None


def get_cfg_defaults():
    return _C.clone()

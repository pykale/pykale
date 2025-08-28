from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Configuration definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = "data/fine_tune/train_data.json"
_C.DATASET.VAL_FRAC = 0.1  # fraction of training data used for validation
_C.DATASET.TEST = "data/fine_tune/test_data.json"


# -----------------------------------------------------------------------------
# Training parameters
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.SEED = 42
_C.SOLVER.EPOCHS = 1
_C.SOLVER.LR = 0.01
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.RANDOMIZE = False
_C.SOLVER.DISABLE_CUDA = False
_C.SOLVER.WORKERS = 0
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.NUM_FOLDS = 10  # for k-fold cross-validation
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.PRINT_FREQ = 10
_C.SOLVER.RESUME = ""

_C.SOLVER.OPTIM = "SGD"  # Choices: ['SGD', 'Adam']

# -----------------------------------------------------------------------------
# Model paths and parameters
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "cgcnn"  # Choices: ['cgcnn', 'leftnet', 'random_forest', 'linear_regression', 'svr']
_C.MODEL.PRETRAINED_MODEL_PATH = ""
_C.MODEL.CIF_FOLDER = "cif_file"
_C.MODEL.INIT_FILE = "cif_file/atom_init.json"
_C.MODEL.MAX_NBRS = 12
_C.MODEL.RADIUS = 8.0


# -----------------------------------------------------------------------------
# CGCNN parameters
# -----------------------------------------------------------------------------
_C.CGCNN = CN()
_C.CGCNN.ATOM_FEA_LEN = 64
_C.CGCNN.H_FEA_LEN = 128
_C.CGCNN.N_CONV = 3
_C.CGCNN.N_H = 1
_C.CGCNN.NUM_REPEAT = 1

# -----------------------------------------------------------------------------
# LeftNet configs
# -----------------------------------------------------------------------------
_C.LEFTNET = CN()
_C.LEFTNET.CUTOFF = 6.0
_C.LEFTNET.HIDDEN_CHANNELS = 128
_C.LEFTNET.NUM_LAYERS = 4
_C.LEFTNET.NUM_RADIAL = 32
_C.LEFTNET.REGRESS_FORCES = False
_C.LEFTNET.USE_PBC = True
_C.LEFTNET.OTF_GRAPH = False
_C.LEFTNET.OUTPUT_DIM = 1
_C.LEFTNET.ENCODING = "z"  # ['z', 'prop']
_C.LEFTNET.TARGET_DIM = 1  # Number of targets for regression or classification

# -----------------------------------------------------------------------------
# CartNet parameters
# -----------------------------------------------------------------------------
_C.CARTNET = CN()
_C.CARTNET.DIM_IN = 256
_C.CARTNET.DIM_RBF = 64
_C.CARTNET.NUM_LAYERS = 4
_C.CARTNET.INVARIANT = True
_C.CARTNET.TEMPERATURE = False
_C.CARTNET.USE_ENVELOPE = True
_C.CARTNET.ATOM_TYPES = True


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
_C.LOGGING.LOG_DIR = "./logs"
_C.LOGGING.LOG_DIR_NAME = None
_C.LOGGING.PROJECT_NAME = "bandgap-project"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the configuration."""
    return _C.clone()

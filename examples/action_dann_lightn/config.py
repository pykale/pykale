"""
Default configurations for action recognition domain adaptation
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "I:/Datasets/EgoAction/"  # "/shared/tale2/Shared"
_C.DATASET.SOURCE = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.SRC_TRAINLIST = "epic_D1_train.pkl"
_C.DATASET.SRC_TESTLIST = "epic_D1_test.pkl"
_C.DATASET.TARGET = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.TGT_TRAINLIST = "epic_D2_train.pkl"
_C.DATASET.TGT_TESTLIST = "epic_D2_test.pkl"
_C.DATASET.IMAGE_MODALITY = "rgb"  # mode options=["rgb", "flow", "audio", "joint", "all"]
_C.DATASET.INPUT_TYPE = "image"  # type options=["image", "feature"]
_C.DATASET.FRAMES_PER_SEGMENT = 16
_C.DATASET.NUM_REPEAT = 5  # 10
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "adaptive"  # options=["source", "max", "adaptive"]
_C.DATASET.CLASS_TYPE = "verb"  # options=["verb", "verb+noun"]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.DIR = os.path.join(_C.OUTPUT.ROOT, _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)
_C.OUTPUT.TB_DIR = os.path.join("lightning_logs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.01  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 30  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 5  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 16  # 150
# _C.SOLVER.TEST_BATCH_SIZE = 16  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1.0

# ---------------------------------------------------------------------------- #
# Feature Extraction configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.METHOD = "i3d"  # options=["r3d_18", "r2plus1d_18", "mc3_18", "i3d"]
_C.MODEL.ATTENTION = "None"  # options=["None", "SELayer"]

# ---------------------------------------------------------------------------- #
# Basic Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = "DANN"  # options=["CDAN", "CDAN-E", "DANN", "DAN", "TA3N"]
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024


# ---------------------------------------------------------------------------- #
# TA3N configs
# ---------------------------------------------------------------------------- #
_C.TA3N = CN()

# -----------------------------------------------------------------------------
# TA3N Dataset
# -----------------------------------------------------------------------------
_C.TA3N.DATASET = CN()
_C.TA3N.DATASET.NUM_SOURCE = 5002  # number of training data (source)
_C.TA3N.DATASET.NUM_TARGET = 7906  # number of training data (target)

_C.TA3N.DATASET.NUM_SEGMENTS = 5  # sample frame # of each video for training
# _C.TA3N.DATASET.VAL_SEGMENTS = 5  # sample frame # of each video for training
_C.TA3N.DATASET.BASELINE_TYPE = "video"  # choices = ['frame', 'tsn']
_C.TA3N.DATASET.FRAME_AGGREGATION = "trn-m"  # method to integrate the frame-level features. choices = [avgpool, trn, trn-m, rnn, temconv]

# ---------------------------------------------------------------------------- #
# TA3N Model
# ---------------------------------------------------------------------------- #

_C.TA3N.MODEL = CN()
_C.TA3N.MODEL.ADD_FC = 1  # number of shared features
_C.TA3N.MODEL.FC_DIM = 512  # dimension of shared features
_C.TA3N.MODEL.ARCH = "TBN"  # choices  = [resnet50]
_C.TA3N.MODEL.USE_TARGET = "uSv"  # choices  = [uSv, Sv, none]
_C.TA3N.MODEL.SHARE_PARAMS = "Y"  # choices  = [Y, N]
_C.TA3N.MODEL.PRED_NORMALIZE = "N"  # choices  = [Y, N]
_C.TA3N.MODEL.WEIGHTED_CLASS_LOSS_DA = "N"  # choices  = [Y, N]
_C.TA3N.MODEL.WEIGHTED_CLASS_LOSS = "N"  # choices  = [Y, N]

_C.TA3N.MODEL.DROPOUT_I = 0.5
_C.TA3N.MODEL.DROPOUT_V = 0.5
_C.TA3N.MODEL.NO_PARTIALBN = True

# DA configs
if _C.TA3N.MODEL.USE_TARGET == "none":
    _C.TA3N.MODEL.EXP_DA_NAME = "baseline"
else:
    _C.TA3N.MODEL.EXP_DA_NAME = "DA"
_C.TA3N.MODEL.DIS_DA = None  # choices  = [DAN, CORAL, JAN]
_C.TA3N.MODEL.ADV_POS_0 = "Y"  # discriminator for relation features. choices  = [Y, N]
_C.TA3N.MODEL.ADV_DA = "RevGrad"  # choices  = [None]
_C.TA3N.MODEL.ADD_LOSS_DA = "attentive_entropy"  # choices  = [None, target_entropy, attentive_entropy]
_C.TA3N.MODEL.ENS_DA = None  # choices  = [None, MCD]

# Attention configs
_C.TA3N.MODEL.USE_ATTN = "TransAttn"  # choices  = [None, TransAttn, general]
_C.TA3N.MODEL.USE_ATTN_FRAME = None  # choices  = [None, TransAttn, general]
_C.TA3N.MODEL.USE_BN = None  # choices  = [None, AdaBN, AutoDIAL]
_C.TA3N.MODEL.N_ATTN = 1
_C.TA3N.MODEL.PLACE_DIS = ["Y", "Y", "N"]
_C.TA3N.MODEL.PLACE_ADV = ["Y", "Y", "Y"]

_C.TA3N.MODEL.N_RNN = 1
_C.TA3N.MODEL.RNN_CELL = "LSTM"
_C.TA3N.MODEL.N_DIRECTIONS = 1
_C.TA3N.MODEL.N_TS = 5
# _C.TA3N.MODEL.TENSORBOARD = True
_C.TA3N.MODEL.FLOW_PREFIX = ""

# ---------------------------------------------------------------------------- #
# TA3N Hyperparameters
# ---------------------------------------------------------------------------- #
_C.TA3N.HYPERPARAMETERS = CN()
_C.TA3N.HYPERPARAMETERS.ALPHA = 0
_C.TA3N.HYPERPARAMETERS.BETA = [0.75, 0.75, 0.5]
_C.TA3N.HYPERPARAMETERS.GAMMA = 0.003  # U->H: 0.003 | H->U: 0.3
_C.TA3N.HYPERPARAMETERS.MU = 0

# ---------------------------------------------------------------------------- #
# TA3N Trainer
# ---------------------------------------------------------------------------- #

_C.TA3N.TRAINER = CN()
# _C.TA3N.TRAINER.TRAIN_METRIC = "all"  # choices  = [noun, verb]
# _C.TA3N.TRAINER.FC_DIM = 512  # dimension of shared features
# _C.TA3N.TRAINER.ARCH = "TBN"  # choices  = [resnet50]
# _C.TA3N.TRAINER.USE_TARGET = "uSv"  # choices  = [uSv, Sv, none]
# _C.TA3N.TRAINER.SHARE_PARAMS = "Y"  # choices  = [Y, N]
_C.TA3N.TRAINER.PRETRAIN_SOURCE = False
_C.TA3N.TRAINER.VERBOSE = True
_C.TA3N.TRAINER.DANN_WARMUP = True

# Learning configs
_C.TA3N.TRAINER.LOSS_TYPE = 'nll'
# _C.TA3N.TRAINER.LR = 0.003
# _C.TA3N.TRAINER.LR_DECAY = 10
_C.TA3N.TRAINER.LR_ADAPTIVE = None  # choices = [None, loss, dann]
_C.TA3N.TRAINER.LR_STEPS = [10, 20]
# _C.TA3N.TRAINER.MOMENTUM = 0.9
_C.TA3N.TRAINER.WEIGHT_DECAY = 0.0001
_C.TA3N.TRAINER.BATCH_SIZE = [_C.SOLVER.TRAIN_BATCH_SIZE, int(_C.SOLVER.TRAIN_BATCH_SIZE * _C.DATASET.NUM_TARGET / _C.DATASET.NUM_SOURCE), _C.SOLVER.TRAIN_BATCH_SIZE]
# _C.TA3N.TRAINER.OPTIMIZER_NAME = "SGD"  # choices = [SGD, Adam]
_C.TA3N.TRAINER.CLIP_GRADIENT = 20

_C.TA3N.TRAINER.PRETRAINED = None
_C.TA3N.TRAINER.RESUME = ""
_C.TA3N.TRAINER.RESUME_HP = ""

# _C.TA3N.TRAINER.MIN_EPOCHS = 20
# _C.TA3N.TRAINER.MAX_EPOCHS = 30

_C.TA3N.TRAINER.ACCELERATOR = "dp"

_C.TA3N.TRAINER.WORKERS = 0
_C.TA3N.TRAINER.EF = 1
_C.TA3N.TRAINER.PF = 50
_C.TA3N.TRAINER.SF = 50
_C.TA3N.TRAINER.COPY_LIST = ["N", "N"]
_C.TA3N.TRAINER.SAVE_MODEL = True

# _C.PATHS.EXP_PATH = os.path.join(
#     _C.PATHS.PATH_EXP + '_' + _C.TA3N.TRAINER.OPTIMIZER_NAME + '-share_params_' + _C.MODEL.SHARE_PARAMS + '-lr_' + str(
#         _C.TA3N.TRAINER.LR) + '-bS_' + str(_C.TA3N.TRAINER.BATCH_SIZE[0]),
#     _C.DATASET.DATASET + '-' + str(_C.DATASET.NUM_SEGMENTS) + '-alpha_' + str(
#         _C.TA3N.HYPERPARAMETERS.ALPHA) + '-beta_' + str(_C.TA3N.HYPERPARAMETERS.BETA[0]) + '_' + str(
#         _C.TA3N.HYPERPARAMETERS.BETA[1]) + '_' + str(_C.TA3N.HYPERPARAMETERS.BETA[2]) + "_gamma_" + str(
#         _C.TA3N.HYPERPARAMETERS.GAMMA) + "_mu_" + str(_C.TA3N.HYPERPARAMETERS.MU))

# ---------------------------------------------------------------------------- #
# TA3N Tester
# ---------------------------------------------------------------------------- #
_C.TA3N.TESTER = CN()

# _C.TA3N.TESTER.TEST_TARGET_DATA = os.path.join(_C.PATHS.PATH_DATA_ROOT, "target_val")

# _C.TA3N.TESTER.WEIGHTS = os.path.join(_C.PATHS.EXP_PATH, "checkpoint.pth.tar")
_C.TA3N.TESTER.NOUN_WEIGHTS = None
_C.TA3N.TESTER.BATCH_SIZE = 512
_C.TA3N.TESTER.DROPOUT_I = 0
_C.TA3N.TESTER.DROPOUT_V = 0
_C.TA3N.TESTER.NOUN_TARGET_DATA = None
_C.TA3N.TESTER.RESULT_JSON = "test.json"
# _C.TA3N.TESTER.TEST_SEGMENTS = 5  # sample frame # of each video for testing
# _C.TA3N.TESTER.SAVE_SCORES = os.path.join(_C.PATHS.EXP_PATH, "scores")
# _C.TA3N.TESTER.SAVE_CONFUSION = os.path.join(_C.PATHS.EXP_PATH, "confusion_matrix")

_C.TA3N.TESTER.VERBOSE = True

def get_cfg_defaults():
    return _C.clone()

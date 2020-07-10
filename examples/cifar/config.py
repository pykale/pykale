# Created by Haiping Lu directly from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/config.py
# Under the MIT License
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = './data'
C.DATASET.NAME = 'CIFAR10'
C.DATASET.NUM_CLASSES = 10
C.DATASET.NUM_WORKERS = 0

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2020
C.SOLVER.BASE_LR = 0.05
C.SOLVER.LR_MILESTONES = [30, 60, 90]
C.SOLVER.LR_GAMMA = 0.1
C.SOLVER.WEIGHT_DECAY = 1e-4
C.SOLVER.MOMENTUM = 0.9
C.SOLVER.DAMPENING = False
C.SOLVER.NESTEROV = False

C.SOLVER.TRAIN_BATCH_SIZE = 128
C.SOLVER.TEST_BATCH_SIZE = 200

C.SOLVER.MAX_EPOCHS = 100

C.SOLVER.WARMUP = False
C.SOLVER.WARMUP_EPOCH = 5
C.SOLVER.WARMUP_FACTOR = 0.2
# ---------------------------------------------------------------------------- #
# ISONet configs
# ---------------------------------------------------------------------------- #
C.ISON = CfgNode()
C.ISON.DEPTH = 34
C.ISON.ORTHO_COEFF = 1e-4
C.ISON.HAS_BN = False
C.ISON.HAS_ST = False
C.ISON.SReLU = True
C.ISON.DIRAC_INIT = True
C.ISON.HAS_RES_MULTIPLIER = False
C.ISON.RES_MULTIPLIER = 1.0
C.ISON.DROPOUT = False
C.ISON.DROPOUT_RATE = 0.0

C.ISON.TRANS_FUN = 'basic_transform'


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.OUTPUT_DIR = './outputs'

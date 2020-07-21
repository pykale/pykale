# Created by Haiping Lu directly from https://github.com/HaozhiQi/ISONet/blob/master/isonet/utils/config.py
# Under the MIT License
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

C = CfgNode()


# {
#     "dataset_group": "digits",
#     "dataset_name": "MNIST to USPS",
#     "source": "mnist",
#     "target": "usps",
#     "size_type": "source",
#     "weight_type": "natural"
# }
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = '../data' # '/shared/tale2/Shared'
C.DATASET.NAME = 'digits' #dset choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset used"
C.DATASET.SOURCE = 'mnist' #s_dset_path  , help="The source dataset path list"
C.DATASET.TARGET = 'usps' #s_dset_path  , help="The target dataset path list"
C.DATASET.NUM_CLASSES = 10
C.DATASET.DIMENSION = 784
C.DATASET.WEIGHT_TYPE = 'natural'
C.DATASET.SIZE_TYPE = 'source'
# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
C.DAN = CfgNode()
C.DAN.METHOD = 'CDAN' # choices=['CDAN', 'CDAN-E', 'DANN']
C.DAN.USERANDOM = False
C.DAN.RANDOM_DIM = 1024
# C.DAN.NET = 'ResNet50' # choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
# C.DAN.BOTTLENECK = True # use_bottleneck
# C.DAN.BNECK_DIM = 256 # bottleneck_dim":256,
# C.DAN.NEW_CLS = True
# C.DAN.RANDOM = False # "whether use random projection"
# C.DAN.RAND_DIM = 1024 # config["loss"]["random_dim"] = 1024
# C.DAN.TRADEOFF = 1.0 #config["loss"] = {"trade_off":1.0}
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2020

C.SOLVER.AD_LAMBDA = True
C.SOLVER.AD_LR = True
C.SOLVER.INIT_LAMBDA = 1
C.SOLVER.MAX_EPOCHS = 100    #"nb_adapt_epochs": 100,
C.SOLVER.INIT_EPOCHS = 20    # "nb_init_epochs": 20,
C.SOLVER.BASE_LR = 0.001 # Initial learning rate
C.SOLVER.BATCH_SIZE = 100 # 150

C.SOLVER.TYPE = "SGD"
C.SOLVER.MOMENTUM = 0.9
C.SOLVER.WEIGHT_DECAY = 0.0005 # 1e-4
C.SOLVER.NESTEROV = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.OUTPUT = CfgNode()
C.OUTPUT.DIR = './outputs' #output_dir
C.OUTPUT.VERBOSE = False
# parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
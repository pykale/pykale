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
C.DATASET.ROOT = '../data' # '/shared/tale2/Shared'
C.DATASET.NAME = 'OFFICE31' #dset choices=['office', 'image-clef', 'visda', 'office-home'], help="The dataset used"
C.DATASET.SOURCE = 'office/amazon_31_list.txt' #s_dset_path  , help="The source dataset path list"
C.DATASET.TARGET = 'office/webcam_10_list.txt' #s_dset_path  , help="The target dataset path list"
C.DATASET.TRAIN_BATCH = 36
C.DATASET.TEST_BATCH = 4
C.DATASET.TEST_AUG = 10 # test_10crop
C.DATASET.NUM_CLASSES = 31
C.DATASET.RESIZE = 256 #"resize_size":256
C.DATASET.CROPSIZE = 224 # "crop_size":224 %config["prep"] = {"test_10crop":True, 'params':{, , 'alexnet':False}}
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2020
C.SOLVER.BASE_LR = 0.001 # learning rate
C.SOLVER.TEST_INT = 500 # "interval of two continuous test phase"
# C.SOLVER.LR_MILESTONES = [30, 60, 90]
C.SOLVER.LR_GAMMA = 0.1
C.SOLVER.LR_POWER = 0.75
C.SOLVER.WEIGHT_DECAY = 0.0005 % 1e-4
C.SOLVER.MOMENTUM = 0.9
C.SOLVER.DAMPENING = False
C.SOLVER.NESTEROV = True
C.SOLVER.LR_TYPE = "inv"

C.SOLVER.MAX_ITERS = 20000 # 100004  num_iterations
# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
C.DAN = CfgNode()
C.DAN.METHOD = 'CDAN+E' # choices=['CDAN', 'CDAN+E', 'DANN']
C.DAN.NET = 'ResNet50' # choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
C.DAN.BOTTLENECK = True # use_bottleneck
C.DAN.BNECK_DIM = 256 # bottleneck_dim":256,
C.DNA.NEW_CLS = True
C.DAN.RANDOM = False # "whether use random projection"
C.DAN.RAND_DIM = 1024 # config["loss"]["random_dim"] = 1024
C.DAN.TRADEOFF = 1.0 #config["loss"] = {"trade_off":1.0}
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
C.OUTPUTOUTPUT = CfgNode()
C.OUTPUT.DIR = './outputs' #output_dir
C.OUTPUT.VERBOSE = False
# parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------- #
# DATA setting
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()
_C.DATA.DATASET = None  # Name of the dataset to use
_C.DATA.SPLIT = None  # Data splitting strategy

# ---------------------------------------------------------------------------- #
# Drug feature extractor
# ---------------------------------------------------------------------------- #
_C.DRUG = CfgNode()
_C.DRUG.NODE_IN_FEATS = 7  # Number of input node features
_C.DRUG.NODE_IN_EMBEDDING = 128  # Dimensionality of input node features after linear transformation
_C.DRUG.PADDING = True  # Whether to apply padding
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]  # Sizes of hidden layers in the GCN feature extractor
_C.DRUG.MAX_NODES = 290  # Max number of nodes to pad to (used when PADDING=True)

# ---------------------------------------------------------------------------- #
# Protein feature extractor
# ---------------------------------------------------------------------------- #
_C.PROTEIN = CfgNode()
_C.PROTEIN.NUM_FILTERS = [128, 128, 128]  # Number of filters in each convolutional layer
_C.PROTEIN.KERNEL_SIZE = [3, 6, 9]  # Kernel size for each convolutional layer
_C.PROTEIN.EMBEDDING_DIM = 128  # Dimension of character embedding for amino acids
_C.PROTEIN.PADDING = True  # Whether to apply zero-padding to the embedding

# ---------------------------------------------------------------------------- #
# BCN setting
# ---------------------------------------------------------------------------- #
_C.BCN = CfgNode()
_C.BCN.HEADS = 2  # Number of attention heads in the Bilinear Attention Network

# ---------------------------------------------------------------------------- #
# MLP decoder
# ---------------------------------------------------------------------------- #
_C.DECODER = CfgNode()
_C.DECODER.NAME = "MLP"  # Decoder type
_C.DECODER.IN_DIM = 256  # Input dimension to the MLP (typically fused BAN feature size)
_C.DECODER.HIDDEN_DIM = 512  # Hidden layer size in the MLP
_C.DECODER.OUT_DIM = 128  # Output dimension before the final classification layer
_C.DECODER.BINARY = 1  # Number of output classes

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.MAX_EPOCH = 100  # Total number of training epochs
_C.SOLVER.BATCH_SIZE = 64  # Batch size for training and evaluation
_C.SOLVER.NUM_WORKERS = 0  # Number of subprocesses for data loading
_C.SOLVER.LEARNING_RATE = 5e-5  # Learning rate for the main model
_C.SOLVER.DA_LEARNING_RATE = 1e-3  # Learning rate for the domain adaptation (if DA is enabled)
_C.SOLVER.SEED = 2048  # Random seed for reproducibility

# ---------------------------------------------------------------------------- #
# RESULT
# ---------------------------------------------------------------------------- #
_C.RESULT = CfgNode()
_C.RESULT.SAVE_MODEL = True  # Whether to save model checkpoints during training

# ---------------------------------------------------------------------------- #
# Domain adaptation
# ---------------------------------------------------------------------------- #
_C.DA = CfgNode()
_C.DA.TASK = False  # False = in-domain splitting task, True = cross-domain splitting task
_C.DA.METHOD = "CDAN"  # Domain adaptation method to use
_C.DA.USE = False  # Whether to enable domain adaptation
_C.DA.INIT_EPOCH = 10  # Number of epochs to wait before applying domain adaptation
_C.DA.LAMB_DA = 1  # Initial value of λ (lambda) used to weight the domain adaptation loss in the total loss   # Total loss = model loss + λ * domain loss
_C.DA.RANDOM_LAYER = False  # Whether to use a random projection layer in CDAN
_C.DA.ORIGINAL_RANDOM = False  # If True, uses the original RandomLayer from the CDAN paper (multi-input form)  # If False, uses a simplified linear layer implementation.
_C.DA.RANDOM_DIM = None  # Output dimensionality of the random layer (only used if RANDOM_LAYER is True)
_C.DA.USE_ENTROPY = True  # Whether to use entropy-based weighting when computing domain adversarial loss

# ---------------------------------------------------------------------------- #
# Comet config, ignore it If not installed.
# ---------------------------------------------------------------------------- #
_C.COMET = CfgNode()
_C.COMET.USE = True  # Enable Comet logging (set True if Comet is installed and configured)
_C.COMET.PROJECT_NAME = "drugban-23-May"  # Comet project name (if applicable)
_C.COMET.EXPERIMENT_NAME = None  # Optional experiment name (e.g., 'drugban-run-1')
_C.COMET.TAG = None  # Comet tags (optional)
_C.COMET.API_KEY = "InDQ1UsqJt7QMiANWg55Ulebe"  # Comet API key (leave blank if unused)


# ---------------------------------------------------------------------------- #
# Function to return a clone of the default config
# ---------------------------------------------------------------------------- #
def get_cfg_defaults():
    return _C.clone()

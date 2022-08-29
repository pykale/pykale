# from config import get_cfg_defaults
import os

import numpy as np
import torch

# ========== config ==========
from yacs.config import CfgNode

import kale.utils.seed as seed
from kale.embed.gripnet import GripNet
from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting
from kale.utils.download import download_file_by_url

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

C = CfgNode()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
C.DATASET = CfgNode()
C.DATASET.ROOT = "./data"
C.DATASET.NAME = "pose"
C.DATASET.URL = "https://github.com/pykale/data/raw/main/graphs/pose_pyg_2.pt"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
C.SOLVER = CfgNode()
C.SOLVER.SEED = 2020
C.SOLVER.BASE_LR = 0.01
C.SOLVER.LR_MILESTONES = [30, 60, 90]
C.SOLVER.LR_GAMMA = 0.1
C.SOLVER.MAX_EPOCHS = 5
C.SOLVER.WARMUP = False
C.SOLVER.WARMUP_EPOCHS = 100


def get_cfg_defaults():
    return C.clone()


# ---- setup device ----
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# ---- setup configs ----
cfg = get_cfg_defaults()
cfg.freeze()
seed.set_seed(cfg.SOLVER.SEED)

# ---- setup dataset ----


# ---- setup dataset ----
def load_data(cfg_dataset: CfgNode):
    """Setup dataset: download and load it."""
    # download data if not exist
    download_file_by_url(cfg.DATASET.URL, cfg.DATASET.ROOT, f"{cfg.DATASET.NAME}.pt")
    data_path = os.path.join(cfg.DATASET.ROOT, f"{cfg.DATASET.NAME}.pt")

    # load data
    return torch.load(data_path)


data = load_data(cfg.DATASET)

# ---- setup supergraph ----
# create gene and drug supervertex
supervertex_gene = SuperVertex("gene", data.g_feat, data.gg_edge_index)
supervertex_drug = SuperVertex("drug", data.d_feat, data.train_idx)

# create superedge form gene to drug supervertex
superedge = SuperEdge("gene", "drug", data.gd_edge_index)

setting_gene = SuperVertexParaSetting("gene", 5, [4, 4])
setting_drug = SuperVertexParaSetting("drug", 7, [6, 6], exter_agg_channels_dict={"gene": 7}, mode="cat")

supergraph = SuperGraph([supervertex_gene, supervertex_drug], [superedge])
supergraph.set_supergraph_para_setting([setting_gene, setting_drug])

gripnet = GripNet(supergraph)
print(gripnet)


class MultiRelaInnerProductDecoder(torch.nn.Module):
    """
    Build `DistMult
    <https://arxiv.org/abs/1412.6575>`_ factorization as GripNet decoder in PoSE dataset.
    """

    def __init__(self, in_dim, num_et):
        super(MultiRelaInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = torch.nn.Parameter(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        """
        Args:
            z: input node feature embeddings.
            edge_index: edge index in COO format with shape [2, num_edges].
            edge_type: The one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: use sigmoid function or not.
        """
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


y = gripnet()

import warnings

import pytorch_lightning as pl
import torch
from config import get_cfg_defaults
from model import GripNetLinkPrediction
from utils import get_all_dataloader, load_data

import kale.utils.seed as seed
from kale.embed.gripnet import GripNet
from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting

warnings.filterwarnings(action="ignore")


# ---- setup device ----
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# ---- setup configs ----
cfg = get_cfg_defaults()
cfg.freeze()
seed.set_seed(cfg.SOLVER.SEED)

# ---- setup dataset ----
data = load_data(cfg.DATASET)

# ---- setup supergraph ----
# create gene and drug supervertex
supervertex_gene = SuperVertex("gene", data.g_feat, data.gg_edge_index)
supervertex_drug = SuperVertex("drug", data.d_feat, data.train_idx, data.train_et)

# create superedge form gene to drug supervertex
superedge = SuperEdge("gene", "drug", data.gd_edge_index)

setting_gene = SuperVertexParaSetting("gene", 5, [4, 4])
setting_drug = SuperVertexParaSetting("drug", 7, [6, 6], exter_agg_channels_dict={"gene": 7}, mode="cat")

supergraph = SuperGraph([supervertex_gene, supervertex_drug], [superedge])
supergraph.set_supergraph_para_setting([setting_gene, setting_drug])

gripnet = GripNet(supergraph)
print(gripnet)


y = gripnet()


dataloader_train, dataloader_test = get_all_dataloader(data)

a = list(dataloader_train)
aa = a[0][0][0]


model = GripNetLinkPrediction(supergraph, cfg.SOLVER)
# model.forward(data.train_idx, data.train_et, data.train_range)

trainer = pl.Trainer(
    default_root_dir=cfg.OUTPUT_DIR, max_epochs=cfg.SOLVER.MAX_EPOCHS, log_every_n_steps=cfg.SOLVER.LOG_EVERY_N_STEPS
)

trainer.fit(model, dataloader_train)

test_result = trainer.test(model, dataloaders=dataloader_train)

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from utils import EPS
from yacs.config import CfgNode

from kale.embed.gripnet import GripNet
from kale.evaluate.metrics import auprc_auroc_ap
from kale.predict.decode import MultiRelaInnerProductDecoder
from kale.prepdata.graph_negative_sampling import typed_negative_sampling
from kale.prepdata.supergraph_construct import SuperGraph


class GripNetLinkPrediction(pl.LightningModule):
    """Build GripNet-DistMult (encoder-decoder) model for link prediction"""

    def __init__(self, supergraph: SuperGraph, conf_solver: CfgNode):
        super().__init__()

        self.conf_solver = conf_solver

        self.encoder = GripNet(supergraph)
        self.decoder = self.__init_decoder__()

    def __init_decoder__(self) -> MultiRelaInnerProductDecoder:
        in_channels = self.encoder.out_channels
        supergraph = self.encoder.supergraph
        task_supervertex_name = supergraph.topological_order[-1]
        num_edge_type = supergraph.supervertex_dict[task_supervertex_name].num_edge_type

        # get the number of nodes on the task-associated supervertex
        self.num_task_nodes = supergraph.supervertex_dict[task_supervertex_name].num_node

        return MultiRelaInnerProductDecoder(in_channels, num_edge_type)

    def forward(self, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_type_range: torch.Tensor) -> Tuple:
        x = self.encoder()

        pos_score = self.decoder(x, edge_index, edge_type)
        pos_loss = -torch.log(pos_score + EPS).mean()

        edge_index = typed_negative_sampling(edge_index, self.num_task_nodes, edge_type_range)

        neg_score = self.decoder(x, edge_index, edge_type)
        neg_loss = -torch.log(1 - neg_score + EPS).mean()

        loss = pos_loss + neg_loss

        # compute averaged metric scores over edge types
        num_edge_type = edge_type_range.shape[0]
        record = []
        for i in range(num_edge_type):
            start, end = edge_type_range[i]
            ps, ns = pos_score[start:end], neg_score[start:end]

            score = torch.cat([ps, ns])
            target = torch.cat([torch.ones(ps.shape[0]), torch.zeros(ns.shape[0])])

            record.append(list(auprc_auroc_ap(target, score)))
        auprc, auroc, ap = np.array(record).mean(axis=0)

        return loss, auprc, auroc, ap

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf_solver.BASE_LR)

        return optimizer

    def __step__(self, batch, mode="train"):
        edge_index, edge_type, edge_type_range = batch
        loss, auprc, auroc, ap = self.forward(
            edge_index.reshape((2, -1)), edge_type.flatten(), edge_type_range.reshape((-1, 2))
        )

        if mode == "train":
            self.log(f"{mode}_loss", loss)
        else:
            self.log(f"{mode}_auprc", auprc)
            self.log(f"{mode}_auroc", auroc)
            self.log(f"{mode}_ap@50", ap)

        return loss

    def training_step(self, batch, batch_idx):
        return self.__step__(batch)

    def test_step(self, batch, batch_idx):
        return self.__step__(batch, mode="test")

    # uncomment this function if a valid set is used
    # def validation_step(self, batch, batch_idx):
    # return self.__step__(batch, mode="val")

    def __repr__(self) -> str:
        return "{}: \nEncoder: {} ModuleDict(\n{})\n Decoder: {}".format(
            self.__class__.__name__, self.encoder.__class__.__name__, self.encoder.supervertex_module_dict, self.decoder
        )

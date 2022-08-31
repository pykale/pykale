import numpy as np
import pytorch_lightning as pl
import torch
from utils import auprc_auroc_ap, EPS, typed_negative_sampling
from yacs.config import CfgNode

from kale.embed.gripnet import GripNet
from kale.prepdata.supergraph_construct import SuperGraph


class MultiRelaInnerProductDecoder(torch.nn.Module):
    """
    Build `DistMult
    <https://arxiv.org/abs/1412.6575>`_ factorization as GripNet decoder in PoSE dataset.
    """

    def __init__(self, in_channels, num_edge_type):
        super(MultiRelaInnerProductDecoder, self).__init__()
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(num_edge_type, in_channels))

        self.reset_parameters()

    def forward(self, x, edge_index, edge_type, sigmoid=True):
        """
        Args:
            z: input node feature embeddings.
            edge_index: edge index in COO format with shape [2, num_edges].
            edge_type: The one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: use sigmoid function or not.
        """
        value = (x[edge_index[0]] * x[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_channels))

    def __repr__(self) -> str:
        return "{}: DistMultLayer(in_channels={}, num_relations={})".format(
            self.__class__.__name__, self.in_channels, self.num_edge_type
        )


class GripNetLinkPrediction(pl.LightningModule):
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

        self.num_task_nodes = supergraph.supervertex_dict[task_supervertex_name].num_node

        return MultiRelaInnerProductDecoder(in_channels, num_edge_type)

    def forward(self, edge_index, edge_type, edge_type_range):
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

        record = np.zeros((3, num_edge_type))
        for i in range(num_edge_type):
            start, end = edge_type_range[i]
            ps, ns = pos_score[start:end], neg_score[start:end]

            score = torch.cat([ps, ns])
            target = torch.cat([torch.ones(ps.shape[0]), torch.zeros(ns.shape[0])])

            record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target, score)
        auprc, auroc, ap = record.mean(axis=1)

        return loss, auprc, auroc, ap

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.conf_solver.BASE_LR)

        return optimizer

    def __step__(self, batch, mode="train"):
        edge_index, edge_type, edge_type_range = batch
        loss, auprc, auroc, ap = self.forward(
            edge_index.reshape((2, -1)), edge_type.flatten(), edge_type_range.reshape((-1, 2))
        )

        if mode != "test":
            self.log(f"{mode}_loss", loss)

        self.log(f"{mode}_auprc", auprc)
        self.log(f"{mode}_auroc", auroc)
        self.log(f"{mode}_ap@50", ap)

        return loss

    def training_step(self, batch, batch_idx):
        return self.__step__(batch)

    def test_step(self, batch, batch_idx):
        return self.__step__(batch, mode="test")

        # def validation_step(self, batch, batch_idx):
        return self.__step__(batch, mode="val")

    def __repr__(self) -> str:
        return "{}: \nEncoder: {} ModuleDict(\n{})\n Decoder: {}".format(
            self.__class__.__name__, self.encoder.__class__.__name__, self.encoder.supervertex_module_dict, self.decoder
        )

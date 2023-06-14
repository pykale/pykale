# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Construct a pipeline to run MOGONET architecture based on PyTorch Lightning.

This code is written by refactoring MOGONET code (https://github.com/txWang/MOGONET/blob/main/train_test.py)
within the PyTorch Lightning.

Reference:
Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021). MOGONET integrates multi-omics data
using graph convolutional networks allowing patient classification and biomarker identification. Nature communications.
https://www.nature.com/articles/s41467-021-23774-w
"""

from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor
from torch.nn import CrossEntropyLoss, ModuleList
from torch.optim.optimizer import Optimizer
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

from kale.embed.mogonet import MogonetGCN
from kale.loaddata.multiomics_datasets import SparseMultiOmicsDataset
from kale.predict.decode import LinearClassifier, VCDN


class ModalityTrainer(pl.LightningModule):
    r"""The implementation of the MOGONET method, which is based on PyTorch Lightning.

    Args:
        dataset (SparseMultiOmicsDataset): The input dataset created in form of :class:`~torch_geometric.data.Dataset`.
        num_modalities (int): The total number of modalities in the dataset.
        num_classes (int): The total number of classes in the dataset.
        modality_encoder (List[MogonetGCN]): The list of GCN encoders for each modality.
        modality_decoder (List[LinearClassifier]): The list of linear classifier decoders for each modality.
        loss_fn (CrossEntropyLoss): The loss function used to gauge the error between the prediction outputs and the
            provided target values.
        multi_modality_decoder (VCDN, optional): The VCDN decoder used in the multi modality dataset.
            (default: ``None``)
        train_multi_modality_decoder (bool, optional): Whether to train VCDN module. (default: ``True``)
        gcn_lr (float, optional): The learning rate used in GCN module. (default: 5e-4)
        vcdn_lr (float, optional): The learning rate used in VCDN module. (default: 1e-3)
    """

    def __init__(
        self,
        dataset: SparseMultiOmicsDataset,
        num_modalities: int,
        num_classes: int,
        modality_encoder: List[MogonetGCN],
        modality_decoder: List[LinearClassifier],
        loss_fn: CrossEntropyLoss,
        multi_modality_decoder: Optional[VCDN] = None,
        train_multi_modality_decoder: bool = True,
        gcn_lr: float = 5e-4,
        vcdn_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.modality_encoder = ModuleList(modality_encoder)
        self.modality_decoder = ModuleList(modality_decoder)
        self.multi_modality_decoder = multi_modality_decoder
        self.train_multi_modality_decoder = train_multi_modality_decoder
        self.loss_fn = loss_fn
        self.gcn_lr = gcn_lr
        self.vcdn_lr = vcdn_lr

        # activate manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self) -> Union[Optimizer, List[Optimizer]]:
        """Return the optimizers that are being used during training."""
        optimizers = []

        for modality in range(self.num_modalities):
            optimizers.append(
                torch.optim.Adam(
                    list(self.modality_encoder[modality].parameters()) + list(self.modality_decoder[modality].parameters()),
                    lr=self.gcn_lr,
                )
            )

        if self.multi_modality_decoder is not None:
            optimizers.append(torch.optim.Adam(self.multi_modality_decoder.parameters(), lr=self.vcdn_lr))

        return optimizers

    def forward(
        self, x: List[Tensor], adj_t: List[SparseTensor], multi_modality: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        """Same as :meth:`torch.nn.Module.forward()`.

        Raises:
            TypeError: If `multi_modality_decoder` is `None` for multi-modality datasets.
        """
        output = []

        for modality in range(self.num_modalities):
            output.append(self.modality_decoder[modality](self.modality_encoder[modality](x[modality], adj_t[modality])))

        if not multi_modality:
            return output

        if self.multi_modality_decoder is not None:
            return self.multi_modality_decoder(output)

        raise TypeError("multi_modality_decoder must be defined for multi-modality datasets.")

    def training_step(self, train_batch, batch_idx: int):
        """Compute and return the training loss.

        Args:
            train_batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (``int``): Integer displaying index of this batch.
        """
        optimizer = self.optimizers()

        x = []
        adj_t = []
        y = []
        sample_weight = []
        for modality in range(self.num_modalities):
            data = train_batch[modality]
            x.append(data.x[data.train_idx])
            adj_t.append(data.adj_t_train)
            y.append(data.y[data.train_idx])
            sample_weight.append(data.train_sample_weight)

        outputs = self.forward(x, adj_t, multi_modality=False)

        for modality in range(self.num_modalities):
            loss = self.loss_fn(outputs[modality], y[modality])
            loss = torch.mean(torch.mul(loss, sample_weight[modality]))
            self.logger.log_metrics({f"train_modality_step_loss ({modality + 1})": loss.detach()}, self.global_step)

            optimizer[modality].zero_grad()
            self.manual_backward(loss)
            optimizer[modality].step()

        if self.train_multi_modality_decoder and self.multi_modality_decoder is not None:
            output = self.forward(x, adj_t, multi_modality=True)
            multi_loss = self.loss_fn(output, y[0])
            multi_loss = torch.mean(torch.mul(multi_loss, sample_weight[0]))
            self.logger.log_metrics({"train_multi_modality_step_loss": multi_loss.detach()}, self.global_step)

            optimizer[-1].zero_grad()
            self.manual_backward(multi_loss)
            optimizer[-1].step()

    def test_step(self, test_batch, batch_idx: int):
        """Compute and return the test loss.

        Args:
            test_batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch.
        """
        x = []
        adj_t = []
        y = []
        for modality in range(self.num_modalities):
            data = test_batch[modality]
            x.append(data.x)
            adj_t.append(data.adj_t)
            y.append(torch.argmax(data.y[data.test_idx], dim=1))

        if self.multi_modality_decoder is not None:
            output = self.forward(x, adj_t, multi_modality=True)
        else:
            output = self.forward(x, adj_t, multi_modality=False)[0]

        pred_test_data = torch.index_select(output, dim=0, index=test_batch[0].test_idx)
        final_output = F.softmax(pred_test_data, dim=1).detach().cpu().numpy()
        actual_output = y[0].detach().cpu()

        if self.num_classes == 2:
            self.log("Accuracy", round(accuracy_score(actual_output, final_output.argmax(1)), 3))
            self.log("F1", round(f1_score(actual_output, final_output.argmax(1)), 3))
            self.log("AUC", round(roc_auc_score(actual_output, final_output[:, 1]), 3))
        else:
            self.log("Accuracy", round(accuracy_score(actual_output, final_output.argmax(1)), 3))
            self.log("F1 weighted", round(f1_score(actual_output, final_output.argmax(1), average="weighted"), 3))
            self.log("F1 macro", round(f1_score(actual_output, final_output.argmax(1), average="macro"), 3))

        return final_output

    def _custom_data_loader(self) -> DataLoader:
        """Return an iterable or collection of iterables specifying samples."""
        dataloaders = DataLoader(self.dataset, batch_size=1)
        return dataloaders

    def train_dataloader(self) -> DataLoader:
        """Return an iterable or collection of iterables specifying training samples."""
        return self._custom_data_loader()

    def test_dataloader(self) -> DataLoader:
        """Return an iterable or collection of iterables specifying test samples."""
        return self._custom_data_loader()

    def __repr__(self) -> str:
        model_str = ["\nModel info:\n", "   Modality encoder:\n"]

        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.modality_encoder[modality]}")

        model_str.append("\n\n  Modality decoder:\n")
        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.modality_decoder[modality]}")

        if self.multi_modality_decoder is not None:
            model_str.append("\n\n  Multi-modality decoder:\n")
            model_str.append(f"    {self.multi_modality_decoder}")

        return "".join(model_str)

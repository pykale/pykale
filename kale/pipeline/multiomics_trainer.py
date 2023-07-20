# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Construct a pipeline to run the MOGONET method based on PyTorch Lightning. MOGONET is a multiomics fusion framework for
cancer classification and biomarker identification that utilizes supervised graph convolutional networks for omics
datasets.

This code is written by refactoring the MOGONET code (https://github.com/txWang/MOGONET/blob/main/train_test.py)
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
from kale.loaddata.multiomics_datasets import SparseMultiomicsDataset
from kale.predict.decode import LinearClassifier, VCDN


class MultiomicsTrainer(pl.LightningModule):
    r"""The PyTorch Lightning implementation of the MOGONET method, a multiomics fusion method designed for
    classification tasks.

    Args:
        dataset (SparseMultiomicsDataset): The input dataset created in form of :class:`~torch_geometric.data.Dataset`.
        num_modalities (int): The total number of modalities in the dataset.
        num_classes (int): The total number of classes in the dataset.
        unimodal_encoder (List[MogonetGCN]): The list of GCN encoders for each modality.
        unimodal_decoder (List[LinearClassifier]): The list of linear classifier decoders for each modality.
        loss_fn (CrossEntropyLoss): The loss function used to gauge the error between the prediction outputs and the
            provided target values.
        multimodal_decoder (VCDN, optional): The VCDN decoder used in the multiomics dataset.
            (default: ``None``)
        train_multimodal_decoder (bool, optional): Whether to train VCDN module. (default: ``True``)
        gcn_lr (float, optional): The learning rate used in the GCN module. (default: 5e-4)
        vcdn_lr (float, optional): The learning rate used in the VCDN module. (default: 1e-3)
    """

    def __init__(
        self,
        dataset: SparseMultiomicsDataset,
        num_modalities: int,
        num_classes: int,
        unimodal_encoder: List[MogonetGCN],
        unimodal_decoder: List[LinearClassifier],
        loss_fn: CrossEntropyLoss,
        multimodal_decoder: Optional[VCDN] = None,
        train_multimodal_decoder: bool = True,
        gcn_lr: float = 5e-4,
        vcdn_lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.unimodal_encoder = ModuleList(unimodal_encoder)
        self.unimodal_decoder = ModuleList(unimodal_decoder)
        self.multimodal_decoder = multimodal_decoder
        self.train_multimodal_decoder = train_multimodal_decoder
        self.loss_fn = loss_fn
        self.gcn_lr = gcn_lr
        self.vcdn_lr = vcdn_lr

        # activate manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self) -> Union[Optimizer, List[Optimizer]]:
        """Return the optimizers used during training."""
        optimizers = []

        for modality in range(self.num_modalities):
            optimizers.append(
                torch.optim.Adam(
                    list(self.unimodal_encoder[modality].parameters())
                    + list(self.unimodal_decoder[modality].parameters()),
                    lr=self.gcn_lr,
                )
            )

        if self.multimodal_decoder is not None:
            optimizers.append(torch.optim.Adam(self.multimodal_decoder.parameters(), lr=self.vcdn_lr))

        return optimizers

    def forward(
        self, x: List[Tensor], adj_t: List[SparseTensor], multimodal: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        """Same as :meth:`torch.nn.Module.forward()`.

        Raises:
            TypeError: If `multimodal_decoder` is `None` for multiomics datasets.
        """
        output = []

        for modality in range(self.num_modalities):
            output.append(
                self.unimodal_decoder[modality](self.unimodal_encoder[modality](x[modality], adj_t[modality]))
            )

        if not multimodal:
            return output

        if self.multimodal_decoder is not None:
            return self.multimodal_decoder(output)

        raise TypeError("multimodal_decoder must be defined for multiomics datasets.")

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

        outputs = self.forward(x, adj_t, multimodal=False)

        for modality in range(self.num_modalities):
            loss = self.loss_fn(outputs[modality], y[modality])
            loss = torch.mean(torch.mul(loss, sample_weight[modality]))
            self.logger.log_metrics({f"train_unimodal_step_loss ({modality + 1})": loss.detach()}, self.global_step)

            optimizer[modality].zero_grad()
            self.manual_backward(loss)
            optimizer[modality].step()

        if self.train_multimodal_decoder and self.multimodal_decoder is not None:
            output = self.forward(x, adj_t, multimodal=True)
            multi_loss = self.loss_fn(output, y[0])
            multi_loss = torch.mean(torch.mul(multi_loss, sample_weight[0]))
            self.logger.log_metrics({"train_multimodal_step_loss": multi_loss.detach()}, self.global_step)

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

        if self.multimodal_decoder is not None:
            output = self.forward(x, adj_t, multimodal=True)
        else:
            output = self.forward(x, adj_t, multimodal=False)[0]

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
        """Return an iterable or a collection of iterables that specifies all the samples in the dataset."""
        dataloaders = DataLoader(self.dataset, batch_size=1)
        return dataloaders

    def train_dataloader(self) -> DataLoader:
        """Return an iterable or a collection of iterables that specifies training samples in the dataset."""
        return self._custom_data_loader()

    def test_dataloader(self) -> DataLoader:
        """Return an iterable or a collection of iterables that specifies test samples in the dataset."""
        return self._custom_data_loader()

    def __str__(self) -> str:
        r"""Returns a string representation of the multiomics trainer object.

        Returns:
            str: The string representation of the multiomics trainer object.
        """
        model_str = ["\nModel info:\n", "   Unimodal encoder:\n"]

        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.unimodal_encoder[modality]}")

        model_str.append("\n\n  Unimodal decoder:\n")
        for modality in range(self.num_modalities):
            model_str.append(f"    ({modality + 1}) {self.unimodal_decoder[modality]}")

        if self.multimodal_decoder is not None:
            model_str.append("\n\n  Multimodal decoder:\n")
            model_str.append(f"    {self.multimodal_decoder}")

        return "".join(model_str)

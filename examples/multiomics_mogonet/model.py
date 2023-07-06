from typing import List, Optional

from torch.nn import CrossEntropyLoss
from yacs.config import CfgNode

from kale.embed.mogonet import MogonetGCN
from kale.loaddata.multiomics_datasets import SparseMultiOmicsDataset
from kale.pipeline.multiomics_trainer import ModalityTrainer
from kale.predict.decode import LinearClassifier, VCDN


class MogonetModel:
    r"""Setup the MOGONET model via the config file.

    Args:
        cfg (CfgNode): A YACS config object.
        dataset (SparseMultiOmicsDataset): The input dataset created in form of :class:`~torch_geometric.data.Dataset`.
    """

    def __init__(self, cfg: CfgNode, dataset: SparseMultiOmicsDataset) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.modality_encoder: List[MogonetGCN] = []
        self.modality_decoder: List[LinearClassifier] = []
        self.multi_modality_decoder: Optional[VCDN] = None
        self.loss_function = CrossEntropyLoss(reduction="none")
        self._create_model()

    def _create_model(self) -> None:
        """Create the MOGONET model via the config file."""
        num_modalities = self.cfg.DATASET.NUM_MODALITIES
        num_classes = self.cfg.DATASET.NUM_CLASSES
        gcn_dropout_rate = self.cfg.MODEL.GCN_DROPOUT_RATE
        gcn_hidden_dim = self.cfg.MODEL.GCN_HIDDEN_DIM
        vcdn_hidden_dim = pow(num_classes, num_modalities)

        for modality in range(num_modalities):
            self.modality_encoder.append(
                MogonetGCN(
                    in_channels=self.dataset.get(modality).num_features,
                    hidden_channels=gcn_hidden_dim,
                    dropout=gcn_dropout_rate,
                )
            )

            self.modality_decoder.append(LinearClassifier(in_dim=gcn_hidden_dim[-1], out_dim=num_classes))

        if num_modalities >= 2:
            self.multi_modality_decoder = VCDN(
                num_modalities=num_modalities, num_classes=num_classes, hidden_dim=vcdn_hidden_dim
            )

    def get_model(self, pretrain: bool = False) -> ModalityTrainer:
        """Return the prepared MOGONET model based on user preference.

        Args:
            pretrain (bool, optional): Whether to return the pretrain model. (default: ``False``)

        Returns:
            ModalityTrainer: The prepared MOGONET model.
        """
        num_modalities = self.cfg.DATASET.NUM_MODALITIES
        num_classes = self.cfg.DATASET.NUM_CLASSES
        gcn_lr_pretrain = self.cfg.MODEL.GCN_LR_PRETRAIN
        gcn_lr = self.cfg.MODEL.GCN_LR
        vcdn_lr = self.cfg.MODEL.VCDN_LR

        if pretrain:
            multi_modality_model = None
            train_multi_modality_decoder = False
            gcn_lr = gcn_lr_pretrain
        else:
            multi_modality_model = self.multi_modality_decoder
            train_multi_modality_decoder = True
            gcn_lr = gcn_lr

        model = ModalityTrainer(
            dataset=self.dataset,
            num_modalities=num_modalities,
            num_classes=num_classes,
            modality_encoder=self.modality_encoder,
            modality_decoder=self.modality_decoder,
            loss_fn=self.loss_function,
            multi_modality_decoder=multi_modality_model,
            train_multi_modality_decoder=train_multi_modality_decoder,
            gcn_lr=gcn_lr,
            vcdn_lr=vcdn_lr,
        )

        return model

    def __str__(self) -> str:
        r"""Returns a string representation of the model object.

        Returns:
            str: The string representation of the model object.
        """
        return self.get_model().__str__()

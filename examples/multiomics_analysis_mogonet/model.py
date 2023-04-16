from yacs.config import CfgNode

from torch.nn import CrossEntropyLoss

from kale.loaddata.multiomics_gnn_dataset import MogonetDataset
from kale.pipeline.mogonet_multiomics_trainer import ModalityTrainer
from kale.embed.mogonet import MogonetGCN
from kale.predict.decode import LinearClassifier, VCDN


class MogonetModel:
    r"""Setup MOGONET model via the config file.

    Args:
        cfg (CfgNode): A YACS config object.
        dataset (MogonetDataset): The input dataset created in form of :class:`~torch_geometric.data.Dataset`.
    """

    def __init__(self,
                 cfg: CfgNode,
                 dataset: MogonetDataset
                 ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.modality_encoder = []
        self.modality_decoder = []
        self.multi_modality_decoder = None
        self.loss_function = CrossEntropyLoss(reduction='none')
        self._create_model()

    def _create_model(self) -> None:
        """Create MOGONET model via the config file."""
        num_view = self.cfg.DATASET.NUM_VIEW
        num_class = self.cfg.DATASET.NUM_CLASS
        gcn_dropout_rate = self.cfg.MODEL.GCN_DROPOUT_RATE
        gcn_hidden_dim = self.cfg.MODEL.GCN_HIDDEN_DIM
        vcdn_hidden_dim = pow(num_class, num_view)

        for view in range(num_view):
            self.modality_encoder.append(MogonetGCN(in_channels=self.dataset.get(view).num_features,
                                                    hidden_channels=gcn_hidden_dim,
                                                    dropout=gcn_dropout_rate
                                                    ))

            self.modality_decoder.append(LinearClassifier(in_dim=gcn_hidden_dim[-1],
                                                          out_dim=num_class
                                                          ))

        if num_view >= 2:
            self.multi_modality_decoder = VCDN(num_view=num_view, num_class=num_class, hidden_dim=vcdn_hidden_dim)

    def get_model(self, pretrain: bool = False) -> ModalityTrainer:
        """Return the prepared MOGONET model based on user preference.

        Args:
            pretrain (bool, optional): Whether to return the pretrain model. (default: :obj:`False`)

        Returns:
            ModalityTrainer: The prepared MOGONET model.
        """
        num_view = self.cfg.DATASET.NUM_VIEW
        num_class = self.cfg.DATASET.NUM_CLASS
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
            num_view=num_view,
            num_class=num_class,
            modality_encoder=self.modality_encoder,
            modality_decoder=self.modality_decoder,
            loss_fn=self.loss_function,
            multi_modality_decoder=multi_modality_model,
            train_multi_modality_decoder=train_multi_modality_decoder,
            gcn_lr=gcn_lr,
            vcdn_lr=vcdn_lr
        )

        return model

    def __repr__(self) -> str:
        return self.get_model().__repr__()

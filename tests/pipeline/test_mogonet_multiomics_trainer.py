import pytest
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss

import kale.prepdata.tabular_transform as T
from kale.embed.mogonet import MogonetGCN
from kale.loaddata.multiomics_datasets import SparseMultiOmicsDataset
from kale.pipeline.mogonet_multiomics_trainer import ModalityTrainer
from kale.predict.decode import LinearClassifier, VCDN
from kale.utils.seed import set_seed


@pytest.fixture
def test_model():
    num_view = 3
    num_class = 2
    gcn_hidden_dim = [200, 200, 100]
    vcdn_hidden_dim = pow(num_class, num_view)
    gcn_dropout_rate = 0.5
    gcn_lr = 5e-4
    vcdn_lr = 1e-3
    loss_function = CrossEntropyLoss(reduction="none")

    modality_encoder = []
    modality_decoder = []
    multi_modality_decoder = None

    url = "https://github.com/SinaTabakhi/pykale-data/raw/main/multiomics/ROSMAP.zip"
    root = "tests/test_data/multiomics/trainer/"
    file_names = []
    for view in range(1, num_view + 1):
        file_names.append(f"{view}_tr.csv")
        file_names.append(f"{view}_lbl_tr.csv")
        file_names.append(f"{view}_te.csv")
        file_names.append(f"{view}_lbl_te.csv")

    dataset = SparseMultiOmicsDataset(
        root=root,
        raw_file_names=file_names,
        num_view=num_view,
        num_class=num_class,
        edge_per_node=10,
        url=url,
        random_split=False,
        train_size=0.7,
        equal_weight=False,
        pre_transform=T.ToTensor(dtype=torch.float),
        target_pre_transform=T.ToOneHotEncoding(dtype=torch.float),
    )

    for view in range(num_view):
        modality_encoder.append(
            MogonetGCN(
                in_channels=dataset.get(view).num_features, hidden_channels=gcn_hidden_dim, dropout=gcn_dropout_rate,
            )
        )

        modality_decoder.append(LinearClassifier(in_dim=gcn_hidden_dim[-1], out_dim=num_class))

    if num_view >= 2:
        multi_modality_decoder = VCDN(num_view=num_view, num_class=num_class, hidden_dim=vcdn_hidden_dim)

    trainer = ModalityTrainer(
        dataset=dataset,
        num_view=num_view,
        num_class=num_class,
        modality_encoder=modality_encoder,
        modality_decoder=modality_decoder,
        loss_fn=loss_function,
        multi_modality_decoder=multi_modality_decoder,
        train_multi_modality_decoder=True,
        gcn_lr=gcn_lr,
        vcdn_lr=vcdn_lr,
    )

    return trainer


@pytest.fixture
def test_model_multi_class():
    num_view = 3
    num_class = 5
    gcn_hidden_dim = [400, 400, 200]
    vcdn_hidden_dim = pow(num_class, num_view)
    gcn_dropout_rate = 0.5
    gcn_lr = 5e-4
    vcdn_lr = 1e-3
    loss_function = CrossEntropyLoss(reduction="none")

    modality_encoder = []
    modality_decoder = []
    multi_modality_decoder = None

    url = "https://github.com/SinaTabakhi/pykale-data/raw/main/multiomics/TCGA_BRCA.zip"
    root = "tests/test_data/multiomics/trainer/multi_class/"
    file_names = []
    for view in range(1, num_view + 1):
        file_names.append(f"{view}_tr.csv")
        file_names.append(f"{view}_lbl_tr.csv")
        file_names.append(f"{view}_te.csv")
        file_names.append(f"{view}_lbl_te.csv")

    dataset = SparseMultiOmicsDataset(
        root=root,
        raw_file_names=file_names,
        num_view=num_view,
        num_class=num_class,
        edge_per_node=10,
        url=url,
        random_split=False,
        train_size=0.7,
        equal_weight=False,
        pre_transform=T.ToTensor(dtype=torch.float),
        target_pre_transform=T.ToOneHotEncoding(dtype=torch.float),
    )

    for view in range(num_view):
        modality_encoder.append(
            MogonetGCN(
                in_channels=dataset.get(view).num_features, hidden_channels=gcn_hidden_dim, dropout=gcn_dropout_rate,
            )
        )

        modality_decoder.append(LinearClassifier(in_dim=gcn_hidden_dim[-1], out_dim=num_class))

    if num_view >= 2:
        multi_modality_decoder = VCDN(num_view=num_view, num_class=num_class, hidden_dim=vcdn_hidden_dim)

    trainer = ModalityTrainer(
        dataset=dataset,
        num_view=num_view,
        num_class=num_class,
        modality_encoder=modality_encoder,
        modality_decoder=modality_decoder,
        loss_fn=loss_function,
        multi_modality_decoder=multi_modality_decoder,
        train_multi_modality_decoder=True,
        gcn_lr=gcn_lr,
        vcdn_lr=vcdn_lr,
    )

    return trainer


def test_init(test_model):
    assert isinstance(test_model.dataset, SparseMultiOmicsDataset)
    assert test_model.num_view == 3
    assert test_model.num_class == 2
    assert all(isinstance(encoder, MogonetGCN) for encoder in test_model.modality_encoder)
    assert all(isinstance(decoder, LinearClassifier) for decoder in test_model.modality_decoder)
    assert isinstance(test_model.loss_fn, CrossEntropyLoss)
    assert isinstance(test_model.multi_modality_decoder, VCDN)
    assert test_model.train_multi_modality_decoder
    assert test_model.gcn_lr == 5e-4
    assert test_model.vcdn_lr == 1e-3


def test_configure_optimizers(test_model):
    optimizers = test_model.configure_optimizers()
    assert len(optimizers) == 4
    assert all(isinstance(optimizer, torch.optim.Adam) for optimizer in optimizers)
    for view in range(test_model.num_view):
        assert optimizers[view].param_groups[0]["lr"] == 5e-4
    assert optimizers[test_model.num_view].param_groups[0]["lr"] == 1e-3


@pytest.mark.parametrize("multi_modality", [False, True])
def test_forward(test_model, multi_modality):
    x = []
    adj_t = []
    for view in range(test_model.num_view):
        data = test_model.dataset.get(view)
        x.append(data.x[data.train_idx])
        adj_t.append(data.adj_t_train)

    outputs = test_model.forward(x, adj_t, multi_modality)

    assert isinstance(outputs, list) != multi_modality
    if not multi_modality:
        assert len(outputs) == test_model.num_view
        for view in range(test_model.num_view):
            assert outputs[view].shape == (test_model.dataset.get(view).num_train, test_model.num_class)
    else:
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (test_model.dataset.get(0).num_train, test_model.num_class)
        test_model.multi_modality_decoder = None
        with pytest.raises(TypeError):
            _ = test_model.forward(x, adj_t, multi_modality)


def test_pipeline(test_model):
    set_seed(2023)
    trainer_pretrain = pl.Trainer(max_epochs=2, gpus=0, enable_model_summary=False,)

    trainer_pretrain.fit(test_model)
    result = trainer_pretrain.test(test_model)

    assert 0 <= result[0]["Accuracy"] <= 1
    assert 0 <= result[0]["F1"] <= 1
    assert 0 <= result[0]["AUC"] <= 1

    assert repr(test_model) is not None
    assert len(repr(test_model)) > 0


def test_multi_class_example(test_model_multi_class):
    set_seed(2023)
    trainer_pretrain = pl.Trainer(max_epochs=2, gpus=0, enable_model_summary=False,)

    trainer_pretrain.fit(test_model_multi_class)
    result = trainer_pretrain.test(test_model_multi_class)

    assert 0 <= result[0]["Accuracy"] <= 1
    assert 0 <= result[0]["F1 weighted"] <= 1
    assert 0 <= result[0]["F1 macro"] <= 1

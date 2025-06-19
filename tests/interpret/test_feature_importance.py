import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss

from kale.embed.mogonet import MogonetGCN
from kale.interpret.feature_importance import select_top_features
from kale.loaddata.multiomics_datasets import SparseMultiomicsDataset
from kale.pipeline.multiomics_trainer import MultiomicsTrainer
from kale.predict.decode import LinearClassifier, VCDN
from kale.prepdata.tabular_transform import ToOneHotEncoding, ToTensor
from kale.utils.seed import set_seed

binary_class_data_url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"
multi_class_data_url = "https://github.com/pykale/data/raw/main/multiomics/TCGA_BRCA.zip"


@pytest.fixture
def test_model(num_classes, url):
    num_modalities = 3
    gcn_hidden_dim = [200, 200, 100]
    vcdn_hidden_dim = pow(num_classes, num_modalities)
    gcn_dropout_rate = 0.5
    gcn_lr = 5e-4
    vcdn_lr = 1e-3
    loss_function = CrossEntropyLoss(reduction="none")

    unimodal_encoder = []
    unimodal_decoder = []
    multimodal_decoder = None

    if num_classes > 2:
        root = "tests/test_data/multiomics/trainer/multi_class"
    else:
        root = "tests/test_data/multiomics/trainer/binary_class"

    file_names = []
    for modality in range(1, num_modalities + 1):
        file_names.append(f"{modality}_tr.csv")
        file_names.append(f"{modality}_lbl_tr.csv")
        file_names.append(f"{modality}_te.csv")
        file_names.append(f"{modality}_lbl_te.csv")
        file_names.append(f"{modality}_feat_name.csv")

    dataset = SparseMultiomicsDataset(
        root=root,
        raw_file_names=file_names,
        num_modalities=num_modalities,
        num_classes=num_classes,
        edge_per_node=10,
        url=url,
        random_split=False,
        train_size=0.7,
        equal_weight=False,
        pre_transform=ToTensor(dtype=torch.float),
        target_pre_transform=ToOneHotEncoding(dtype=torch.float),
    )

    for modality in range(num_modalities):
        unimodal_encoder.append(
            MogonetGCN(
                in_channels=dataset.get(modality).num_features,
                hidden_channels=gcn_hidden_dim,
                dropout=gcn_dropout_rate,
            )
        )

        unimodal_decoder.append(LinearClassifier(in_dim=gcn_hidden_dim[-1], out_dim=num_classes))

    if num_modalities >= 2:
        multimodal_decoder = VCDN(num_modalities=num_modalities, num_classes=num_classes, hidden_dim=vcdn_hidden_dim)

    trainer = MultiomicsTrainer(
        dataset=dataset,
        num_modalities=num_modalities,
        num_classes=num_classes,
        unimodal_encoder=unimodal_encoder,
        unimodal_decoder=unimodal_decoder,
        loss_fn=loss_function,
        multimodal_decoder=multimodal_decoder,
        train_multimodal_decoder=True,
        gcn_lr=gcn_lr,
        vcdn_lr=vcdn_lr,
    )

    return trainer


@pytest.mark.parametrize("num_classes, url", [(2, binary_class_data_url), (5, multi_class_data_url)])
def test_select_top_features(test_model, num_classes, url):
    set_seed(2023)
    trainer = pl.Trainer(
        default_root_dir="./tests/outputs",
        max_epochs=2,
        accelerator="cpu",
        enable_model_summary=False,
    )
    trainer.fit(test_model)

    f1_key = "F1" if test_model.dataset.num_classes == 2 else "F1 macro"
    df_featimp_top = select_top_features(
        trainer=trainer,
        model=test_model,
        dataset=test_model.dataset,
        metric=f1_key,
        num_top_feats=30,
        verbose=False,
    )

    # Check type
    assert isinstance(df_featimp_top, pd.DataFrame)

    # Check expected columns
    expected_columns = {"feat_name", "imp", "omics"}
    assert expected_columns.issubset(df_featimp_top.columns)

    # Check the number of returned features
    assert len(df_featimp_top) <= 30

    # Check importance score is sorted in descending order
    assert df_featimp_top["imp"].is_monotonic_decreasing


def test_select_top_features_raises_for_invalid_metric(test_model):
    trainer = pl.Trainer(
        default_root_dir="./tests/outputs",
        max_epochs=1,
        accelerator="cpu",
        enable_model_summary=False,
    )
    trainer.fit(test_model)

    with pytest.raises(ValueError):
        select_top_features(
            trainer=trainer,
            model=test_model,
            dataset=test_model.dataset,
            metric="nonexistent_metric",
            num_top_feats=10,
            verbose=False,
        )

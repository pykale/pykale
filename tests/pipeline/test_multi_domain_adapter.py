import pytest
import torch
from sklearn.metrics import accuracy_score
from torch.nn.functional import one_hot

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.image_access import ImageAccess
from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.pipeline.multi_domain_adapter import _CoDeRLS, create_ms_adapt_trainer
from kale.predict.class_domain_nets import ClassNetSmallImage
from tests.helpers.pipe_test_helper import ModelTestHelper


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    config_params = {
        "train_params": {
            "adapt_lambda": True,
            "adapt_lr": True,
            "lambda_init": 1,
            "nb_adapt_epochs": 2,
            "nb_init_epochs": 1,
            "init_lr": 0.001,
            "batch_size": 60,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


@pytest.fixture(scope="module")
def office_caltech_access(office_path):
    return ImageAccess.get_multi_domain_images("OFFICE_CALTECH", office_path, download=True, return_domain_label=True)


MSDA_METHODS = ["MFSAN", "M3SDA", "DIN"]


@pytest.mark.parametrize("method", MSDA_METHODS)
def test_multi_source(method, office_caltech_access, testing_cfg):
    dataset = MultiDomainAdapDataset(office_caltech_access)
    feature_network = ResNet18Feature()
    if method == "MFSAN":
        feature_network = torch.nn.Sequential(*(list(feature_network.children())[:-1]))
    # setup classifier
    classifier_network = ClassNetSmallImage
    train_params = testing_cfg["train_params"]
    model = create_ms_adapt_trainer(
        method=method,
        dataset=dataset,
        feature_extractor=feature_network,
        task_classifier=classifier_network,
        n_classes=10,
        target_domain="amazon",
        **train_params,
    )
    kwargs = {"limit_train_batches": 0.1, "limit_val_batches": 0.3, "limit_test_batches": 0.2}
    ModelTestHelper.test_model(model, train_params, **kwargs)


# @pytest.mark.parametrize("loss", ["logits", "mse", "hinge"])
@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_coder(kernel, office_caltech_access):
    dataset = MultiDomainAdapDataset(office_caltech_access)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders(split="train", batch_size=100)
    feature_network = ResNet18Feature()
    x, y, z = next(iter(dataloader))
    tgt_idx = torch.where(z == 0)
    src_idx = torch.where(z != 0)

    x_feat = feature_network(x)
    z_ont_hot = one_hot(z)
    clf = _CoDeRLS(kernel=kernel, alpha=0.01)

    x_train = torch.cat((x_feat[src_idx], x_feat[tgt_idx]))
    y_train = y[src_idx]
    z_train = torch.cat((z_ont_hot[src_idx], z_ont_hot[tgt_idx]))
    clf.fit(x_train, y_train, z_train)
    y_pred = clf.predict(x_feat[tgt_idx])

    acc = accuracy_score(y[tgt_idx], y_pred)

    assert 0 <= acc <= 1

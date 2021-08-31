import pytest
import torch

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.loaddata.office_access import OfficeCaltech
from kale.pipeline.multi_domain_adapter import create_ms_adapt_trainer
from kale.predict.class_domain_nets import ClassNetSmallImage
from tests.pipeline.pipe_utils import test_model


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
    return OfficeCaltech(root=office_path, download=True, return_domain_label=True)


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
    test_model(model, train_params, **kwargs)

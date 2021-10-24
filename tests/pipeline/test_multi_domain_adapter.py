import pytest
import torch

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.image_access import ImageAccess
from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.pipeline.multi_domain_adapter import create_ms_adapt_trainer
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
@pytest.mark.parametrize("input_dimension", [1, 2])
def test_multi_source(method, input_dimension, office_caltech_access, testing_cfg):
    if method != "MFSAN" and input_dimension == 2:
        pytest.skip()
    dataset = MultiDomainAdapDataset(office_caltech_access)
    feature_network = ResNet18Feature()
    # setup classifier
    classifier_network = ClassNetSmallImage
    train_params = testing_cfg["train_params"].copy()
    if method == "MFSAN":
        train_params["input_dimension"] = input_dimension
        if input_dimension == 2:
            feature_network = torch.nn.Sequential(*(list(feature_network.children())[:-1]))

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

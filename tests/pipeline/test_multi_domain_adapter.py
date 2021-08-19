import pytest

from kale.embed.image_cnn import ResNet18Feature
from kale.loaddata.multi_domain import MultiDomainAdapDataset
from kale.loaddata.office_access import OfficeCaltech
from kale.pipeline.multi_domain_adapter import M3SDATrainer
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
            "batch_size": 100,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


@pytest.fixture(scope="module")
def office_caltech_access(office_path):
    return OfficeCaltech(root=office_path, download=True, return_domain_label=True)


def test_multi_source(office_caltech_access, testing_cfg):
    dataset = MultiDomainAdapDataset(office_caltech_access)
    # num_channels = 3
    feature_network = ResNet18Feature()
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = ClassNetSmallImage(feature_dim, 10)
    train_params = testing_cfg["train_params"]
    model = M3SDATrainer(
        dataset=dataset,
        feature_extractor=feature_network,
        task_classifier=classifier_network,
        target_label=1,
        k_moment=3,
        **train_params,
    )
    test_model(model, train_params)

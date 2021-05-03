import pytest
import pytorch_lightning as pl

import kale.pipeline.domain_adapter as domain_adapter
from kale.embed.image_cnn import SmallCNNFeature
from kale.loaddata.digits_access import DigitDataset
from kale.loaddata.multi_domain import MultiDomainDatasets
from kale.predict.class_domain_nets import ClassNetSmallImage, DomainNetSmallImage
from kale.utils.seed import set_seed

SOURCE = "USPS"
TARGET = "USPS"

DA_METHODS = ["DANN", "CDAN", "CDAN-E", "WDGRL", "WDGRLMod", "DAN", "JAN", "FSDANN", "MME", "Source"]

WEIGHT_TYPE = "natural"
DATASIZE_TYPE = "source"
VAL_RATIO = [0.1]
NUM_CLASSES = 10
FEW_SHOT = [None, 2]

# To move to conftest later, or no need
seed = 36
set_seed(seed)


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
            "batch_size": 10,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


@pytest.mark.parametrize("DA_method", DA_METHODS)
@pytest.mark.parametrize("n_fewshot", FEW_SHOT)
def test_domain_adaptor(DA_method, n_fewshot, download_path, testing_cfg):
    if n_fewshot is None:
        if DA_method in ["FSDANN", "MME", "Source"]:
            return
    else:
        if DA_method in ["DANN", "CDAN", "CDAN-E", "WDGRL", "WDGRLMod", "DAN", "JAN"]:
            return

    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(SOURCE), DigitDataset(TARGET), download_path
    )
    dataset = MultiDomainDatasets(
        source, target, config_weight_type=WEIGHT_TYPE, config_size_type=DATASIZE_TYPE, n_fewshot=n_fewshot
    )

    # setup feature extractor
    feature_network = SmallCNNFeature(num_channels)
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = ClassNetSmallImage(feature_dim, NUM_CLASSES)
    train_params = testing_cfg["train_params"]
    method_params = {}
    method = domain_adapter.Method(DA_method)

    if method.is_mmd_method():
        model = domain_adapter.create_mmd_based(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            **method_params,
            **train_params,
        )
    else:  # All other non-mmd DA methods are dann like with critic
        critic_input_size = feature_dim
        # setup critic network
        if method.is_cdan_method():
            critic_input_size = 1024
            method_params["use_random"] = True

        critic_network = DomainNetSmallImage(critic_input_size)

        # The following calls kale.loaddata.dataset_access for the first time
        model = domain_adapter.create_dann_like(
            method=method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            **method_params,
            **train_params,
        )

    assert isinstance(model, domain_adapter.BaseAdaptTrainer)

    trainer = pl.Trainer(min_epochs=train_params["nb_init_epochs"], max_epochs=train_params["nb_adapt_epochs"], gpus=0)
    trainer.fit(model)
    metric_values = trainer.callback_metrics
    assert isinstance(metric_values, dict)

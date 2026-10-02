from unittest.mock import patch

import pytest
import torch

import kale.pipeline.domain_adapter as domain_adapter
from kale.embed.image_cnn import SmallCNNFeature
from kale.loaddata.image_access import DigitDataset
from kale.loaddata.multi_domain import MultiDomainAccess, MultiDomainDataset
from kale.predict.class_domain_nets import ClassNetSmallImage, DomainNetSmallImage
from tests.helpers.pipe_test_helper import ModelTestHelper

# from kale.utils.seed import set_seed

SOURCE = "USPS"
TARGET = "USPS"

DA_METHODS = ["DANN", "CDAN", "CDAN-E", "WDGRL", "WDGRLMod", "DAN", "JAN", "FSDANN", "MME", "Source"]

WEIGHT_TYPE = "natural"
DATASIZE_TYPE = "source"
NUM_CLASSES = 10
FEW_SHOT = [0, 20]

# Not checking values so seed is not needed. If seed, move all seeds to conftest later
# seed = 36
# set_seed(seed)


class _DummyDataset:
    def prepare_data_loaders(self, target_domain=None):
        return None


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    config_params = {
        "train_params": {
            "adapt_lambda": True,
            "adapt_lr": True,
            "lambda_init": 1.0,
            "nb_adapt_epochs": 2,
            "nb_init_epochs": 1,
            "init_lr": 0.001,
            "batch_size": 20,
            "num_workers": 0,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


def test_base_adapt_trainer_configure_optimizers_with_adamw():
    model = domain_adapter.BaseAdaptTrainer(
        dataset=_DummyDataset(),
        feature_extractor=torch.nn.Linear(2, 3),
        task_classifier=torch.nn.Linear(3, 2),
        nb_init_epochs=1,
        nb_adapt_epochs=2,
        init_lr=0.004,
        optimizer={"type": "AdamW", "optim_params": {"eps": 0.2, "weight_decay": 0.3}},
    )

    # adapt_lr defaults to True on BaseAdaptTrainer, so a scheduler comes back alongside the
    # optimizer for AdamW as it does for SGD.
    optimizers, schedulers = model.configure_optimizers()

    assert len(optimizers) == 1
    assert isinstance(optimizers, list)
    assert isinstance(optimizers[0], torch.optim.AdamW)
    assert optimizers[0].defaults["lr"] == 0.004
    assert optimizers[0].defaults["eps"] == 0.2
    assert optimizers[0].defaults["weight_decay"] == 0.3
    assert len(schedulers) == 1


def _wdgrl_trainer(optimizer, adapt_lr):
    return domain_adapter.WDGRLTrainer(
        dataset=_DummyDataset(),
        feature_extractor=torch.nn.Linear(2, 3),
        task_classifier=torch.nn.Linear(3, 2),
        critic=torch.nn.Linear(3, 1),
        nb_init_epochs=1,
        nb_adapt_epochs=2,
        init_lr=0.004,
        adapt_lr=adapt_lr,
        optimizer=optimizer,
    )


OPTIMIZER_CASES = [
    (None, torch.optim.Adam),
    ({"type": "Adam", "optim_params": {}}, torch.optim.Adam),
    ({"type": "AdamW", "optim_params": {}}, torch.optim.AdamW),
    ({"type": "SGD", "optim_params": {"momentum": 0.9}}, torch.optim.SGD),
]
OPTIMIZER_IDS = ["default", "Adam", "AdamW", "SGD"]


@pytest.mark.parametrize("optimizer_params, expected_type", OPTIMIZER_CASES, ids=OPTIMIZER_IDS)
def test_wdgrl_configure_optimizers_with_adapt_lr(optimizer_params, expected_type):
    """Every optimizer type gets a scheduler under adapt_lr, including the default."""
    model = _wdgrl_trainer(optimizer_params, adapt_lr=True)

    optimizers, schedulers = model.configure_optimizers()

    assert isinstance(optimizers[0], expected_type)
    assert len(schedulers) == 1
    # The critic is stepped manually, so its optimizer and scheduler are held on the trainer.
    assert isinstance(model.critic_opt, expected_type)
    assert model.critic_sched is not None


@pytest.mark.parametrize("optimizer_params, expected_type", OPTIMIZER_CASES, ids=OPTIMIZER_IDS)
def test_wdgrl_configure_optimizers_without_adapt_lr(optimizer_params, expected_type):
    """With adapt_lr off, every optimizer type returns a bare list and no scheduler."""
    model = _wdgrl_trainer(optimizer_params, adapt_lr=False)

    optimizers = model.configure_optimizers()

    assert isinstance(optimizers, list)
    assert len(optimizers) == 1
    assert isinstance(optimizers[0], expected_type)
    assert isinstance(model.critic_opt, expected_type)
    assert model.critic_sched is None


def test_wdgrl_configure_optimizers_handles_a_scheduler_less_result():
    """configure_optimizers keys on the returned shape, not on adapt_lr (issue #548).

    _configure_optimizer now returns a scheduler for every optimizer under adapt_lr, so the
    original crash is no longer reachable through configuration. This pins the handling directly:
    a scheduler-less result while adapt_lr is set must still be accepted rather than unpacked as a
    pair, which is what raised `ValueError: not enough values to unpack (expected 2, got 1)`.
    """
    model = _wdgrl_trainer({"type": "Adam", "optim_params": {}}, adapt_lr=True)

    with patch.object(model, "_configure_optimizer", return_value=[torch.optim.Adam(model.parameters())]):
        optimizers = model.configure_optimizers()

    assert isinstance(optimizers, list)
    assert model.critic_sched is None


@pytest.mark.parametrize("da_method", DA_METHODS)
@pytest.mark.parametrize("n_fewshot", FEW_SHOT)
def test_domain_adaptor(da_method, n_fewshot, download_path, testing_cfg):
    if n_fewshot == 0:
        if da_method in ["FSDANN", "MME", "Source"]:
            return
    else:
        if da_method in ["DANN", "CDAN", "CDAN-E", "WDGRL", "WDGRLMod", "DAN", "JAN"]:
            return

    source = DigitDataset.get_access(DigitDataset(SOURCE), download_path)[0]
    target = DigitDataset.get_access(DigitDataset(TARGET), download_path)[0]
    data_access = MultiDomainAccess({"SOURCE": source, "TARGET": target}, 10, return_domain_label=True)
    num_channels = 1
    dataset = MultiDomainDataset(data_access, n_fewshot=n_fewshot)
    # dataset = BiDomainDatasets(
    #     source, target, config_weight_type=WEIGHT_TYPE, config_size_type=DATASIZE_TYPE, n_fewshot=n_fewshot
    # )

    # setup feature extractor
    feature_network = SmallCNNFeature(num_channels)
    # setup classifier
    feature_dim = feature_network.output_size()
    classifier_network = ClassNetSmallImage(feature_dim, NUM_CLASSES)
    train_params = testing_cfg["train_params"]
    method_params = {}
    da_method = domain_adapter.Method(da_method)

    if da_method.is_mmd_method():
        model = domain_adapter.create_mmd_based(
            method=da_method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            target_domain="TARGET",
            **method_params,
            **train_params,
        )
    else:  # All other non-mmd DA methods are dann like with critic
        critic_input_size = feature_dim
        # setup critic network
        if da_method.is_cdan_method():
            critic_input_size = 1024
            method_params["use_random"] = True

        critic_network = DomainNetSmallImage(critic_input_size)

        # The following calls kale.loaddata.dataset_access for the first time
        model = domain_adapter.create_dann_like(
            method=da_method,
            dataset=dataset,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            target_domain="TARGET",
            **method_params,
            **train_params,
        )

    ModelTestHelper.test_model(model, train_params)

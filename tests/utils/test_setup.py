from unittest.mock import MagicMock, patch

import torch

from kale.utils.setup import setup_comet, setup_device


def test_setup_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = setup_device()
    assert device.type == "cpu"


def test_setup_device_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    device = setup_device()
    assert device.type == "cuda"


@patch("kale.utils.setup.Experiment")
def test_setup_comet_minimal(mock_experiment_cls):
    mock_experiment = MagicMock()
    mock_experiment_cls.return_value = mock_experiment

    result = setup_comet(api_key="dummy", project_name="test", workspace="test_user")

    mock_experiment_cls.assert_called_once_with(api_key="dummy", project_name="test", workspace="test_user")
    assert result == mock_experiment


@patch("kale.utils.setup.Experiment")
def test_setup_comet_with_params(mock_experiment_cls):
    mock_experiment = MagicMock()
    mock_experiment_cls.return_value = mock_experiment

    result = setup_comet(
        api_key="dummy",
        project_name="test",
        workspace="test_user",
        log_params={"lr": 0.01},
        experiment_tag="unit-test",
        experiment_name="Test Experiment",
    )

    mock_experiment_cls.assert_called_once_with(api_key="dummy", project_name="test", workspace="test_user")
    mock_experiment.log_parameters.assert_called_once_with({"lr": 0.01})
    mock_experiment.add_tag.assert_called_once_with("unit-test")
    mock_experiment.set_name.assert_called_once_with("Test Experiment")
    assert result == mock_experiment

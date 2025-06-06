import torch

from kale.utils.setup import setup_device


def test_setup_device_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = setup_device()
    assert device.type == "cpu"


def test_setup_device_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    device = setup_device()
    assert device.type == "cuda"

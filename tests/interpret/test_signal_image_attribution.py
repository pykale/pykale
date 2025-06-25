# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

from unittest.mock import patch

import numpy as np
import pytest
import torch

# Import your function here (adjust the import as needed)
from kale.interpret.signal_image_attribution import multimodal_signal_image_attribution


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, image, signal):
        # Returns logits: shape (batch_size, num_classes)
        return torch.tensor([[1.0, 2.0]], device=image.device)

    def parameters(self, recurse=True):
        # Fake parameter to get device
        param = torch.nn.Parameter(torch.zeros(1))
        yield param


@pytest.fixture
def dummy_data():
    # Batch size 2, image and signal are small arrays
    images = torch.ones((2, 1, 8, 8))
    signals = torch.ones((2, 1, 12 * 500))  # 12 leads, 1 second at 500 Hz
    labels = torch.tensor([0, 1])
    return [(images, signals, labels)]


def dummy_ecg_clean(sig, sampling_rate=500):
    # Pass-through for test
    return sig


class DummyIntegratedGradients:
    def __init__(self, model):
        self.model = model

    def attribute(self, inputs, target=None, return_convergence_delta=False):
        # Return attributions as ones, same shape as inputs
        img_attr = torch.ones_like(inputs[0])
        sig_attr = torch.ones_like(inputs[1])
        return (img_attr, sig_attr), None


@patch("kale.interpret.signal_image_attribution.IntegratedGradients", DummyIntegratedGradients)
@patch("kale.interpret.signal_image_attribution.nk.ecg_clean", dummy_ecg_clean)
def test_multimodal_signal_image_attribution_full_coverage(dummy_data):
    model = DummyModel()
    for p in model.parameters():
        p.data = p.data.cpu()

    result = multimodal_signal_image_attribution(
        last_fold_model=model,
        last_val_loader=dummy_data,
        sample_idx=0,
        signal_threshold=0.5,
        image_threshold=0.5,
        zoom_range=(0.1, 0.2),
        lead_number=12,
        sampling_rate=500,
    )

    # --- Checks for all outputs ---
    assert isinstance(result, dict)
    expected_keys = [
        "label",
        "predicted_label",
        "predicted_probability",
        "signal_waveform_np",
        "full_time",
        "full_length",
        "important_indices_full",
        "segment_signal_waveform",
        "zoom_time",
        "important_indices_zoom",
        "zoom_start_sec",
        "zoom_end_sec",
        "image_np",
        "x_pts",
        "y_pts",
        "importance_pts",
        "signal_threshold",
        "image_threshold",
    ]
    for key in expected_keys:
        assert key in result

    assert isinstance(result["label"], int)
    assert isinstance(result["predicted_label"], int)
    assert isinstance(result["predicted_probability"], float)
    assert isinstance(result["signal_waveform_np"], np.ndarray)
    assert isinstance(result["image_np"], np.ndarray)
    assert isinstance(result["x_pts"], np.ndarray)
    assert isinstance(result["y_pts"], np.ndarray)
    assert isinstance(result["importance_pts"], np.ndarray)
    assert isinstance(result["important_indices_full"], np.ndarray)
    assert isinstance(result["important_indices_zoom"], np.ndarray)
    assert isinstance(result["zoom_time"], np.ndarray)
    assert isinstance(result["segment_signal_waveform"], np.ndarray)
    assert result["signal_waveform_np"].ndim == 1
    assert result["image_np"].ndim == 2 or result["image_np"].ndim == 3
    assert result["signal_threshold"] == 0.5
    assert result["image_threshold"] == 0.5
    assert 0 <= result["predicted_probability"] <= 1


@patch("kale.interpret.signal_image_attribution.IntegratedGradients", DummyIntegratedGradients)
@patch("kale.interpret.signal_image_attribution.nk.ecg_clean", dummy_ecg_clean)
def test_multimodal_signal_image_attribution_index_bounds(dummy_data):
    model = DummyModel()
    multimodal_signal_image_attribution(
        last_fold_model=model,
        last_val_loader=dummy_data,
        sample_idx=1,  # Second sample
    )


@patch("kale.interpret.signal_image_attribution.IntegratedGradients", DummyIntegratedGradients)
@patch("kale.interpret.signal_image_attribution.nk.ecg_clean", dummy_ecg_clean)
def test_multimodal_signal_image_attribution_default_args(dummy_data):
    model = DummyModel()
    multimodal_signal_image_attribution(
        last_fold_model=model,
        last_val_loader=dummy_data,
    )


@patch("kale.interpret.signal_image_attribution.IntegratedGradients", DummyIntegratedGradients)
@patch("kale.interpret.signal_image_attribution.nk.ecg_clean", dummy_ecg_clean)
def test_multimodal_signal_image_attribution_empty_loader():
    model = DummyModel()
    with pytest.raises(ValueError, match="Validation loader is empty. No data to interpret."):
        multimodal_signal_image_attribution(
            last_fold_model=model,
            last_val_loader=[],
            sample_idx=0,
        )

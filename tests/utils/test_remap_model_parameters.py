# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

import pytest

from kale.utils.remap_model_parameters import remap_state_dict_keys  # Change this to your actual module path


@pytest.fixture
def original_state_dict():
    # Covers all remapping cases, plus some that should remain unchanged.
    return {
        "ecg_encoder.layer1.weight": 1,
        "ecg_decoder.layer2.bias": 2,
        "fc_logvar.weight": 3,
        "fc_logvar.bias": 4,
        "signal_encoder.fc_logvar.weight": 5,
        "signal_encoder.fc_logvar.bias": 6,
        "unaffected_param": 7,
    }


def test_remap_state_dict_keys_full_coverage(original_state_dict):
    new_state_dict = remap_state_dict_keys(original_state_dict)

    # Test main remappings
    assert "signal_encoder.layer1.weight" in new_state_dict
    assert "signal_decoder.layer2.bias" in new_state_dict
    assert "fc_log_var.weight" in new_state_dict
    assert "fc_log_var.bias" in new_state_dict
    assert "signal_encoder.fc_log_var.weight" in new_state_dict
    assert "signal_encoder.fc_log_var.bias" in new_state_dict

    # Check remapped values
    assert new_state_dict["signal_encoder.layer1.weight"] == 1
    assert new_state_dict["signal_decoder.layer2.bias"] == 2
    assert new_state_dict["fc_log_var.weight"] == 3
    assert new_state_dict["fc_log_var.bias"] == 4
    assert new_state_dict["signal_encoder.fc_log_var.weight"] == 5
    assert new_state_dict["signal_encoder.fc_log_var.bias"] == 6

    # Unaffected params remain unchanged
    assert "unaffected_param" in new_state_dict
    assert new_state_dict["unaffected_param"] == 7

    # Ensure no old keys remain
    for old_key in [
        "ecg_encoder.layer1.weight",
        "ecg_decoder.layer2.bias",
        "fc_logvar.weight",
        "fc_logvar.bias",
        "signal_encoder.fc_logvar.weight",
        "signal_encoder.fc_logvar.bias",
    ]:
        assert old_key not in new_state_dict


def test_remap_state_dict_keys_no_changes():
    # No keys to be remapped
    state_dict = {"something_else": 42}
    new_state_dict = remap_state_dict_keys(state_dict)
    assert new_state_dict == state_dict


def test_remap_state_dict_keys_empty():
    # Handles empty state dict
    state_dict = {}
    new_state_dict = remap_state_dict_keys(state_dict)
    assert new_state_dict == {}

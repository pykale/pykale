# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================


def remap_state_dict_keys(state_dict):
    """
    Remap parameter names in a state_dict to match updated module names or variable conventions.

    This utility function updates parameter keys from a loaded checkpoint to match a model’s new architecture,
    such as when module or variable names have changed (e.g., from "ecg_encoder" to "signal_encoder").
    This allows loading pretrained weights into a model with different internal naming conventions.

    Args:
        state_dict (dict):
            State dictionary from a pretrained PyTorch model, typically loaded using ``torch.load``.

    Returns:
        dict:
            A new state dictionary with parameter names remapped according to the defined mapping.
    """
    mapping = [
        ("ecg_encoder.", "signal_encoder."),
        ("ecg_decoder.", "signal_decoder."),
        ("fc_logvar", "fc_log_var"),
    ]
    new_state_dict = {}
    for k, v in state_dict.items():
        for old, new in mapping:
            if old in k:
                k = k.replace(old, new)
        new_state_dict[k] = v
    return new_state_dict

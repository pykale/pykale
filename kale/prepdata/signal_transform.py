import numpy as np
import pandas as pd
import torch


def normalize_signal(signal):
    """
    Normalizes a multi-channel ECG signal by removing mean and scaling to unit variance per channel.

    Args:
        signal (ndarray): Array of shape (samples, channels)

    Returns:
        ndarray: Normalized signal, same shape as input.
    """
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    std_safe = np.where(std == 0, 1e-10, std)
    normalized_signal = (signal - mean) / std_safe
    return normalized_signal


def interpolate_signal(signal):
    """
    Linearly interpolates missing or NaN values in the ECG signal.

    Args:
        signal (ndarray): Array of shape (samples, channels)

    Returns:
        ndarray: Interpolated signal, same shape as input.
    """
    signal_df = pd.DataFrame(signal)
    signal_df.interpolate(method="linear", axis=0, inplace=True, limit_direction="both")
    return signal_df.to_numpy()


def prepare_ecg_tensor(signal):
    """
    Converts preprocessed ECG numpy array to a torch tensor of shape (1, -1).

    Args:
        signal (ndarray): Preprocessed and normalized ECG array (samples, channels).

    Returns:
        Tensor: Flattened ECG tensor, shape (1, total_samples).
    """
    return torch.tensor(signal.reshape(1, -1), dtype=torch.float32)

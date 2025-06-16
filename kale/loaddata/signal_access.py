import logging
import os

import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from kale.prepdata.signal_transform import interpolate_signal, normalize_signal, prepare_ecg_tensor


def load_ecg_from_folder(base_path, csv_file):
    """
    Loads and preprocesses a batch of ECG signals from a CSV file listing file paths.

    Args:
        base_path (str): Root directory containing ECG files.
        csv_file (str): CSV file listing files in column 'path'.

    Returns:
        Tensor: Batch of preprocessed ECG signals, shape (N, 1, total_samples).
    Example:
        ecg_tensor = load_ecg_from_csv("/data/ecg/", "ecg_files.csv")
    """
    cases = pd.read_csv(os.path.join(base_path, csv_file))
    full_paths = cases["path"].apply(lambda x: os.path.join(base_path, x))
    all_ecg = []

    for f in tqdm(full_paths, desc="Loading ECG data"):
        wave_array, meta = wfdb.rdsamp(f)
        wave_array = interpolate_signal(wave_array)
        num_channels = meta["n_sig"]
        num_samples_per_channel = wave_array.size // num_channels

        if wave_array.size % num_channels == 0:
            wave_array = wave_array.reshape(num_samples_per_channel, num_channels)
            wave_array = normalize_signal(wave_array)
            ecg_tensor = prepare_ecg_tensor(wave_array)
            all_ecg.append(ecg_tensor)
        else:
            logging.warning(f"Unexpected data size in {f}. Skipping file.")
    if all_ecg:
        return torch.cat(all_ecg, dim=0)
    else:
        return torch.empty(0)

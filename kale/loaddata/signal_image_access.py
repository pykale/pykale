# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

import numpy as np
from torch.utils.data import Dataset


class SignalImageDataset(Dataset):
    """
    SignalImageDataset prepares paired signal (e.g., ECG) and image (e.g., CXR) features for multimodal deep learning tasks.

    This class simplifies data preparation by accepting two tensors: one for signal features and one for image features.
    Each sample returned by the dataset consists of a pair of (signal_features, image_features) at the same index,
    making it suitable for tasks where both modalities are required as input (such as multimodal classification,
    reconstruction, or representation learning).

    Args:
        signal_features (Tensor or ndarray): Tensor containing the signal features for all samples.
        image_features (Tensor or ndarray): Tensor containing the image features for all samples.

    Usage:
        dataset = SignalImageDataset(signal_features, image_features)
        signal, image = dataset[0]
        # Can be used with DataLoader for batching in model training.

    Returns:
        Tuple: (signal_features, image_features) for the requested sample index.
    """

    def __init__(self, signal_features, image_features):
        self.signal_features = signal_features
        self.image_features = image_features

    def __len__(self):
        return len(self.signal_features)

    def __getitem__(self, idx):
        return self.signal_features[idx], self.image_features[idx]

    @classmethod
    def prepare_data_loaders(cls, signal_features, image_features, train_ratio=0.8, random_seed=None):
        """
        Splits the dataset into training and validation subsets.

        Args:
            signal_features (Tensor or ndarray): Tensor containing the signal features.
            image_features (Tensor or ndarray): Tensor containing the image features.
            train_ratio (float, optional): Ratio of the training set (e.g., 0.8 for 80% train, 20% val). Default is 0.8.
            random_seed (int, optional): Seed for reproducibility.

        Returns:
            train_dataset (SignalImageDataset): Training subset.
            val_dataset (SignalImageDataset): Validation subset.
        """
        assert len(signal_features) == len(image_features), "Mismatch in number of samples."

        num_samples = len(signal_features)
        indices = np.arange(num_samples)
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_size = int(train_ratio * num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_signal = signal_features[train_indices]
        train_image = image_features[train_indices]
        val_signal = signal_features[val_indices]
        val_image = image_features[val_indices]

        return (cls(train_signal, train_image), cls(val_signal, val_image))

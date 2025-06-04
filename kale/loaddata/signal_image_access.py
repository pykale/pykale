# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

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


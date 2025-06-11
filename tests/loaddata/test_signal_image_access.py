import torch

from kale.loaddata.signal_image_access import SignalImageDataset


def test_signal_image_dataset():
    # Create dummy data
    num_samples = 4
    signal_features = torch.randn(num_samples, 20)  # 4 samples, 20-dim signal features
    image_features = torch.randn(num_samples, 1, 8, 8)  # 4 samples, 1x8x8 images

    # Test __init__ and __len__
    dataset = SignalImageDataset(signal_features, image_features)
    assert len(dataset) == num_samples

    # Test __getitem__ for each index
    for i in range(num_samples):
        signal, image = dataset[i]
        assert torch.equal(signal, signal_features[i])
        assert torch.equal(image, image_features[i])
        assert signal.shape == (20,)
        assert image.shape == (1, 8, 8)

    # Test prepare_data_loaders (train/val split)
    train_dataset, val_dataset = SignalImageDataset.prepare_data_loaders(
        signal_features, image_features, train_ratio=0.75, random_seed=42
    )
    # The split should be deterministic due to random_seed=42
    assert isinstance(train_dataset, SignalImageDataset)
    assert isinstance(val_dataset, SignalImageDataset)

    total = len(train_dataset) + len(val_dataset)
    assert total == num_samples
    # Shapes should match original
    assert train_dataset.signal_features.shape[1:] == signal_features.shape[1:]
    assert train_dataset.image_features.shape[1:] == image_features.shape[1:]
    assert val_dataset.signal_features.shape[1:] == signal_features.shape[1:]
    assert val_dataset.image_features.shape[1:] == image_features.shape[1:]

    # Check non-overlapping indices between train and val (due to shuffling)
    all_train_indices = set(tuple(sf.tolist()) for sf in train_dataset.signal_features)
    all_val_indices = set(tuple(sf.tolist()) for sf in val_dataset.signal_features)
    assert all_train_indices.isdisjoint(all_val_indices) or len(val_dataset) == 0

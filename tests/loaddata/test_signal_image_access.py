import torch

from kale.loaddata.signal_image_access import SignalImageDataset


def test_signal_image_dataset_full_coverage():
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

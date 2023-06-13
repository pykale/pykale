import os
import tempfile

import numpy as np
import pytest

from kale.loaddata.avmnist_datasets import AVMNISTDataset


@pytest.fixture(scope="module")
def dummy_data():
    # Create temporary directory for data
    temp_dir = tempfile.TemporaryDirectory()
    data_dir = temp_dir.name

    # Create dummy AVMNIST data
    create_dummy_data(data_dir)

    yield data_dir

    # Clean up temporary directory
    temp_dir.cleanup()


def create_dummy_data(data_dir):
    os.makedirs(os.path.join(data_dir, "image"))
    os.makedirs(os.path.join(data_dir, "audio"))

    train_image_data = np.random.rand(100, 28, 28)
    train_audio_data = np.random.rand(100, 112, 112)
    train_labels = np.random.randint(0, 10, size=(100,))

    test_image_data = np.random.rand(20, 28, 28)
    test_audio_data = np.random.rand(20, 112, 112)
    test_labels = np.random.randint(0, 10, size=(20,))

    np.save(os.path.join(data_dir, "image/train_data.npy"), train_image_data)
    np.save(os.path.join(data_dir, "audio/train_data.npy"), train_audio_data)
    np.save(os.path.join(data_dir, "train_labels.npy"), train_labels)

    np.save(os.path.join(data_dir, "image/test_data.npy"), test_image_data)
    np.save(os.path.join(data_dir, "audio/test_data.npy"), test_audio_data)
    np.save(os.path.join(data_dir, "test_labels.npy"), test_labels)


def test_avmnist_dataset(dummy_data):
    # Initialize AVMNISTDataset with temporary data directory
    dataset = AVMNISTDataset(dummy_data)

    # Test train, validation, and test data loaders
    train_loader = dataset.get_train_loader()
    valid_loader = dataset.get_valid_loader()
    test_loader = dataset.get_test_loader()

    # Check if the loaders are working and have the correct dimensions
    for loader in [train_loader, valid_loader, test_loader]:
        for images, audios, labels in loader:
            assert images.shape[1:] == (1, 28, 28)
            assert audios.shape[1:] == (1, 112, 112)
            assert labels.min() >= 0
            assert labels.max() < 10
            break

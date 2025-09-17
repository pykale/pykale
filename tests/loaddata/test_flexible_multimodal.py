"""Tests for flexible multimodal dataset loader."""

import os
import tempfile
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

from kale.loaddata.flexible_multimodal import (
    ModalityConfig,
    FlexibleMultimodalDataset,
    FlexibleMultimodalDataLoader,
    create_custom_multimodal_loader,
)


@pytest.fixture(scope="function")
def temp_data_dir():
    """Create temporary directory with dummy multimodal data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy data for different modalities
        n_samples = 100
        
        # Image-like data (28x28)
        image_data = np.random.rand(n_samples, 28, 28).astype(np.float32)
        np.save(os.path.join(temp_dir, "images.npy"), image_data)
        
        # Audio-like data (1D signal)
        audio_data = np.random.rand(n_samples, 1000).astype(np.float32)
        np.save(os.path.join(temp_dir, "audio.npy"), audio_data)
        
        # Text-like data (embeddings)
        text_data = np.random.rand(n_samples, 300).astype(np.float32)
        np.save(os.path.join(temp_dir, "text.npy"), text_data)
        
        # Labels
        labels = np.random.randint(0, 5, n_samples)
        np.save(os.path.join(temp_dir, "labels.npy"), labels)
        
        yield temp_dir


class TestModalityConfig:
    """Test ModalityConfig class."""
    
    def test_basic_config(self):
        """Test basic modality configuration."""
        config = ModalityConfig(
            name="test_modality",
            data_path="/path/to/data.npy",
            flatten=True,
            normalize=True
        )
        
        assert config.name == "test_modality"
        assert config.data_path == "/path/to/data.npy"
        assert config.flatten is True
        assert config.normalize is True
        assert config.unsqueeze_channel is False
        assert config.dtype == torch.float32
    
    def test_config_with_transform(self):
        """Test configuration with custom transform."""
        def custom_transform(x):
            return x * 2
        
        config = ModalityConfig(
            name="transformed_modality",
            transform=custom_transform,
            extra_param="test_value"
        )
        
        assert config.transform == custom_transform
        assert config.extra_params["extra_param"] == "test_value"


class TestFlexibleMultimodalDataset:
    """Test FlexibleMultimodalDataset class."""
    
    def test_basic_dataset_creation(self, temp_data_dir):
        """Test basic dataset creation with multiple modalities."""
        configs = [
            ModalityConfig("image", os.path.join(temp_data_dir, "images.npy")),
            ModalityConfig("audio", os.path.join(temp_data_dir, "audio.npy"), flatten=True),
            ModalityConfig("text", os.path.join(temp_data_dir, "text.npy"), normalize=True)
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        
        assert len(dataset) == 100
        assert dataset.get_modality_names() == ["image", "audio", "text"]
        
        # Test single sample
        sample = dataset[0]
        assert len(sample) == 3  # Three modalities
        assert isinstance(sample[0], torch.Tensor)  # Image
        assert isinstance(sample[1], torch.Tensor)  # Audio (flattened)
        assert isinstance(sample[2], torch.Tensor)  # Text (normalized)
        
        # Check shapes
        assert sample[0].shape == (28, 28)  # Image not flattened
        assert sample[1].shape == (1000,)   # Audio flattened
        assert sample[2].shape == (300,)    # Text
    
    def test_dataset_with_labels(self, temp_data_dir):
        """Test dataset with labels."""
        configs = [
            ModalityConfig("image", os.path.join(temp_data_dir, "images.npy")),
            ModalityConfig("audio", os.path.join(temp_data_dir, "audio.npy"))
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        
        # Load and set labels
        labels = np.load(os.path.join(temp_data_dir, "labels.npy"))
        dataset.set_labels(labels)
        
        # Test sample with labels
        sample_data, label = dataset[0]
        assert len(sample_data) == 2
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.int64  # Default for integer conversion
    
    def test_preprocessing_options(self, temp_data_dir):
        """Test various preprocessing options."""
        configs = [
            ModalityConfig(
                "test",
                os.path.join(temp_data_dir, "images.npy"),
                flatten=True,
                normalize=True,
                unsqueeze_channel=True
            )
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        sample = dataset[0]
        
        # Should be flattened (28*28=784), normalized, with channel dim added
        data = sample[0]
        assert data.shape == (784, 1)  # Flattened + channel dimension
        assert 0.0 <= data.min() <= data.max() <= 1.0  # Normalized range
    
    def test_custom_data_loader(self, temp_data_dir):
        """Test with custom data loader function."""
        def custom_loader(config):
            # Return mock data
            return np.random.rand(50, 10, 10)
        
        configs = [ModalityConfig("custom", data_path=None)]
        dataset = FlexibleMultimodalDataset(configs, data_loader=custom_loader)
        
        assert len(dataset) == 50
        sample = dataset[0]
        assert sample[0].shape == (10, 10)
    
    def test_shape_validation(self, temp_data_dir):
        """Test shape validation."""
        configs = [
            ModalityConfig(
                "image",
                os.path.join(temp_data_dir, "images.npy"),
                shape=(28, 28)  # Correct shape
            )
        ]
        
        # Should not raise warning for correct shape
        dataset = FlexibleMultimodalDataset(configs)
        assert len(dataset) == 100
        
        # Test with incorrect shape
        configs_wrong = [
            ModalityConfig(
                "image",
                os.path.join(temp_data_dir, "images.npy"),
                shape=(32, 32)  # Wrong shape
            )
        ]
        
        with pytest.warns(UserWarning, match="Shape mismatch"):
            dataset_wrong = FlexibleMultimodalDataset(configs_wrong)
    
    def test_mismatched_sample_counts(self, temp_data_dir):
        """Test error handling for mismatched sample counts."""
        # Create data with different sample counts
        small_data = np.random.rand(50, 10)
        small_path = os.path.join(temp_data_dir, "small.npy")
        np.save(small_path, small_data)
        
        configs = [
            ModalityConfig("large", os.path.join(temp_data_dir, "images.npy")),  # 100 samples
            ModalityConfig("small", small_path)  # 50 samples
        ]
        
        with pytest.raises(ValueError, match="same number of samples"):
            FlexibleMultimodalDataset(configs)
    
    def test_caching(self, temp_data_dir):
        """Test data caching functionality."""
        configs = [ModalityConfig("image", os.path.join(temp_data_dir, "images.npy"))]
        
        # Test with caching enabled
        dataset_cached = FlexibleMultimodalDataset(configs, cache_data=True)
        assert dataset_cached._data_cache is not None
        assert "image" in dataset_cached._data_cache
        
        # Test without caching
        dataset_no_cache = FlexibleMultimodalDataset(configs, cache_data=False)
        assert dataset_no_cache._data_cache is None


class TestFlexibleMultimodalDataLoader:
    """Test FlexibleMultimodalDataLoader class."""
    
    def test_data_loader_creation(self, temp_data_dir):
        """Test data loader creation and splits."""
        configs = [
            ModalityConfig("image", os.path.join(temp_data_dir, "images.npy")),
            ModalityConfig("audio", os.path.join(temp_data_dir, "audio.npy"))
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        
        # Load labels
        labels = np.load(os.path.join(temp_data_dir, "labels.npy"))
        dataset.set_labels(labels)
        
        data_loader = FlexibleMultimodalDataLoader(
            dataset=dataset,
            batch_size=16,
            train_split=0.7,
            valid_split=0.2,
            test_split=0.1
        )
        
        # Test loader creation
        train_loader = data_loader.get_train_loader()
        valid_loader = data_loader.get_valid_loader()
        test_loader = data_loader.get_test_loader()
        
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(valid_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)
        
        # Test sizes approximately match splits
        total_size = len(dataset)
        train_size = len(data_loader.train_dataset)
        valid_size = len(data_loader.valid_dataset)
        test_size = len(data_loader.test_dataset)
        
        assert train_size + valid_size + test_size == total_size
        assert train_size == int(0.7 * total_size)
        assert valid_size == int(0.2 * total_size)
    
    def test_invalid_splits(self, temp_data_dir):
        """Test error handling for invalid splits."""
        configs = [ModalityConfig("test", os.path.join(temp_data_dir, "images.npy"))]
        dataset = FlexibleMultimodalDataset(configs)
        
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            FlexibleMultimodalDataLoader(
                dataset=dataset,
                train_split=0.6,
                valid_split=0.3,
                test_split=0.2  # Sum > 1.0
            )
    
    def test_data_iteration(self, temp_data_dir):
        """Test iterating through data loaders."""
        configs = [
            ModalityConfig("image", os.path.join(temp_data_dir, "images.npy")),
            ModalityConfig("audio", os.path.join(temp_data_dir, "audio.npy"))
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        labels = np.load(os.path.join(temp_data_dir, "labels.npy"))
        dataset.set_labels(labels)
        
        data_loader = FlexibleMultimodalDataLoader(dataset=dataset, batch_size=8)
        train_loader = data_loader.get_train_loader()
        
        # Test one batch
        for batch_data, batch_labels in train_loader:
            assert len(batch_data) == 2  # Two modalities
            assert batch_data[0].shape[0] <= 8  # Batch size
            assert batch_data[1].shape[0] <= 8  # Batch size
            assert batch_labels.shape[0] <= 8  # Batch size
            break


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_custom_multimodal_loader(self, temp_data_dir):
        """Test create_custom_multimodal_loader function."""
        modality_paths = {
            'image': os.path.join(temp_data_dir, 'images.npy'),
            'audio': os.path.join(temp_data_dir, 'audio.npy'),
            'text': os.path.join(temp_data_dir, 'text.npy')
        }
        
        modality_configs = {
            'image': {'flatten': True, 'normalize': True},
            'audio': {'flatten': False, 'normalize': False},
            'text': {'unsqueeze_channel': True}
        }
        
        loader = create_custom_multimodal_loader(
            modality_paths=modality_paths,
            modality_configs=modality_configs,
            batch_size=16
        )
        
        assert isinstance(loader, FlexibleMultimodalDataLoader)
        assert len(loader.dataset.get_modality_names()) == 3
        
        # Test that configurations were applied
        image_config = loader.dataset.get_modality_config('image')
        assert image_config.flatten is True
        assert image_config.normalize is True
        
        text_config = loader.dataset.get_modality_config('text')
        assert text_config.unsqueeze_channel is True
    
    def test_create_custom_loader_without_configs(self, temp_data_dir):
        """Test create_custom_multimodal_loader without additional configs."""
        modality_paths = {
            'image': os.path.join(temp_data_dir, 'images.npy'),
            'audio': os.path.join(temp_data_dir, 'audio.npy')
        }
        
        loader = create_custom_multimodal_loader(
            modality_paths=modality_paths,
            batch_size=32
        )
        
        assert isinstance(loader, FlexibleMultimodalDataLoader)
        assert len(loader.dataset.get_modality_names()) == 2
        assert loader.batch_size == 32


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_missing_data_path(self):
        """Test error when data path is missing."""
        config = ModalityConfig("test")  # No data_path provided
        
        with pytest.raises(ValueError, match="No data_path provided"):
            FlexibleMultimodalDataset([config])
    
    def test_invalid_data_path(self):
        """Test error when data path is invalid."""
        config = ModalityConfig("test", data_path="/nonexistent/path.npy")
        
        with pytest.raises(RuntimeError, match="Failed to load data"):
            FlexibleMultimodalDataset([config])
    
    def test_label_length_mismatch(self, temp_data_dir):
        """Test error when label length doesn't match dataset."""
        configs = [ModalityConfig("test", os.path.join(temp_data_dir, "images.npy"))]
        dataset = FlexibleMultimodalDataset(configs)
        
        wrong_labels = np.random.randint(0, 5, 50)  # Wrong length
        
        with pytest.raises(ValueError, match="Label length .* doesn't match dataset length"):
            dataset.set_labels(wrong_labels)
    
    def test_unknown_modality_config(self, temp_data_dir):
        """Test error when requesting unknown modality config."""
        configs = [ModalityConfig("test", os.path.join(temp_data_dir, "images.npy"))]
        dataset = FlexibleMultimodalDataset(configs)
        
        with pytest.raises(KeyError, match="Modality 'unknown' not found"):
            dataset.get_modality_config("unknown")


class TestTransforms:
    """Test transform functionality."""
    
    def test_modality_specific_transform(self, temp_data_dir):
        """Test modality-specific transformations."""
        def double_transform(x):
            return x * 2
        
        configs = [
            ModalityConfig(
                "test",
                os.path.join(temp_data_dir, "images.npy"),
                transform=double_transform
            )
        ]
        
        dataset = FlexibleMultimodalDataset(configs)
        sample = dataset[0]
        
        # Load original data to compare
        original_data = np.load(os.path.join(temp_data_dir, "images.npy"))
        original_sample = torch.tensor(original_data[0], dtype=torch.float32)
        
        # Check that transform was applied
        expected = double_transform(original_sample)
        torch.testing.assert_close(sample[0], expected)
    
    def test_common_transform(self, temp_data_dir):
        """Test common transform applied to all modalities."""
        def common_transform(modalities):
            # Add 1 to all modalities
            return [mod + 1 for mod in modalities]
        
        configs = [
            ModalityConfig("image", os.path.join(temp_data_dir, "images.npy")),
            ModalityConfig("audio", os.path.join(temp_data_dir, "audio.npy"))
        ]
        
        dataset = FlexibleMultimodalDataset(configs, common_transform=common_transform)
        sample = dataset[0]
        
        # Load original data
        original_image = np.load(os.path.join(temp_data_dir, "images.npy"))
        original_audio = np.load(os.path.join(temp_data_dir, "audio.npy"))
        
        # Check that common transform was applied
        expected_image = torch.tensor(original_image[0], dtype=torch.float32) + 1
        expected_audio = torch.tensor(original_audio[0], dtype=torch.float32) + 1
        
        torch.testing.assert_close(sample[0], expected_image)
        torch.testing.assert_close(sample[1], expected_audio)
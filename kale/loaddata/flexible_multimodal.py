"""Flexible multimodal dataset loader with configurable modalities and preprocessing."""

from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
import warnings


class ModalityConfig:
    """Configuration for a single modality.
    
    Args:
        name (str): Name identifier for the modality (e.g., 'image', 'audio', 'text')
        data_path (str, optional): Path to the modality data
        transform (callable, optional): Transformation function to apply to the data
        flatten (bool): Whether to flatten the data. Defaults to False.
        normalize (bool): Whether to normalize the data. Defaults to False.
        unsqueeze_channel (bool): Whether to add a channel dimension. Defaults to False.
        dtype (torch.dtype): Target data type. Defaults to torch.float32.
        shape (tuple, optional): Expected/target shape for validation
    """
    
    def __init__(
        self,
        name: str,
        data_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        flatten: bool = False,
        normalize: bool = False,
        unsqueeze_channel: bool = False,
        dtype: torch.dtype = torch.float32,
        shape: Optional[tuple] = None,
        **kwargs
    ):
        self.name = name
        self.data_path = data_path
        self.transform = transform
        self.flatten = flatten
        self.normalize = normalize
        self.unsqueeze_channel = unsqueeze_channel
        self.dtype = dtype
        self.shape = shape
        self.extra_params = kwargs


class FlexibleMultimodalDataset(Dataset):
    """Flexible multimodal dataset that can handle variable number of modalities.
    
    This dataset class provides a unified interface for loading and preprocessing
    multimodal data with configurable modalities, transformations, and data loading
    strategies.
    
    Args:
        modality_configs (List[ModalityConfig]): List of modality configurations
        data_loader (callable, optional): Custom data loading function
        common_transform (callable, optional): Transform applied to all modalities
        validate_shapes (bool): Whether to validate data shapes. Defaults to True.
        cache_data (bool): Whether to cache data in memory. Defaults to False.
    """
    
    def __init__(
        self,
        modality_configs: List[ModalityConfig],
        data_loader: Optional[Callable] = None,
        common_transform: Optional[Callable] = None,
        validate_shapes: bool = True,
        cache_data: bool = False
    ):
        self.modality_configs = modality_configs
        self.data_loader_fn = data_loader or self._default_data_loader
        self.common_transform = common_transform
        self.validate_shapes = validate_shapes
        self.cache_data = cache_data
        
        # Internal storage
        self._data_cache = {} if cache_data else None
        self._labels = None
        self._length = None
        
        # Load and validate data
        self._load_data()
        
    def _default_data_loader(self, config: ModalityConfig) -> np.ndarray:
        """Default data loader that loads from .npy files."""
        if config.data_path is None:
            raise ValueError(f"No data_path provided for modality '{config.name}'")
        
        try:
            data = np.load(config.data_path)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load data for modality '{config.name}': {e}")
    
    def _load_data(self):
        """Load data for all modalities."""
        modality_data = {}
        
        for config in self.modality_configs:
            data = self.data_loader_fn(config)
            
            if self.validate_shapes and config.shape is not None:
                if data.shape[1:] != config.shape:
                    warnings.warn(
                        f"Shape mismatch for modality '{config.name}': "
                        f"expected {config.shape}, got {data.shape[1:]}"
                    )
            
            modality_data[config.name] = data
        
        # Validate that all modalities have the same number of samples
        lengths = [len(data) for data in modality_data.values()]
        if len(set(lengths)) > 1:
            raise ValueError(f"All modalities must have the same number of samples. Got: {lengths}")
        
        self._length = lengths[0]
        
        if self.cache_data:
            self._data_cache = modality_data
        else:
            # Store references for lazy loading
            self._modality_paths = {config.name: config for config in self.modality_configs}
    
    def _preprocess_modality(self, data: np.ndarray, config: ModalityConfig) -> torch.Tensor:
        """Apply preprocessing to a single modality."""
        # Convert to tensor
        data = torch.tensor(data, dtype=config.dtype)
        
        # Apply normalization
        if config.normalize:
            if data.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
                data = data.float()
                data = data / 255.0
            else:
                # For float data, normalize to [0, 1] range
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)
        
        # Apply flattening
        if config.flatten:
            original_shape = data.shape
            data = data.view(original_shape[0], -1) if len(original_shape) > 1 else data
        
        # Add channel dimension
        if config.unsqueeze_channel:
            data = data.unsqueeze(-1) if not config.flatten else data.unsqueeze(1)
        
        # Apply modality-specific transform
        if config.transform is not None:
            data = config.transform(data)
        
        return data
    
    def get_modality_names(self) -> List[str]:
        """Get list of modality names."""
        return [config.name for config in self.modality_configs]
    
    def get_modality_config(self, name: str) -> ModalityConfig:
        """Get configuration for a specific modality."""
        for config in self.modality_configs:
            if config.name == name:
                return config
        raise KeyError(f"Modality '{name}' not found")
    
    def set_labels(self, labels: Union[np.ndarray, torch.Tensor, List]):
        """Set labels for the dataset."""
        if isinstance(labels, (list, np.ndarray)):
            labels = torch.tensor(labels)
        
        if len(labels) != self._length:
            raise ValueError(f"Label length {len(labels)} doesn't match dataset length {self._length}")
        
        self._labels = labels
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int) -> Union[Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        """Get a sample with all modalities."""
        modality_samples = []
        
        for config in self.modality_configs:
            if self.cache_data:
                data = self._data_cache[config.name][idx]
            else:
                # Lazy loading - load the full modality data and extract sample
                full_data = self.data_loader_fn(config)
                data = full_data[idx]
            
            # Preprocess the sample
            processed_data = self._preprocess_modality(
                np.expand_dims(data, 0), config  # Add batch dimension for processing
            )[0]  # Remove batch dimension
            
            modality_samples.append(processed_data)
        
        # Apply common transform if provided
        if self.common_transform is not None:
            modality_samples = self.common_transform(modality_samples)
        
        # Return with or without labels
        if self._labels is not None:
            return modality_samples, self._labels[idx]
        else:
            return modality_samples


class FlexibleMultimodalDataLoader:
    """Convenient wrapper for creating DataLoaders from FlexibleMultimodalDataset."""
    
    def __init__(
        self,
        dataset: FlexibleMultimodalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        train_split: float = 0.8,
        valid_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Validate splits
        if abs(train_split + valid_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_split = train_split
        self.valid_split = valid_split
        self.test_split = test_split
        
        self._create_splits()
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        dataset_size = len(self.dataset)
        
        # Calculate split sizes
        train_size = int(self.train_split * dataset_size)
        valid_size = int(self.valid_split * dataset_size)
        test_size = dataset_size - train_size - valid_size
        
        # Create splits
        torch.manual_seed(self.random_seed)
        self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, valid_size, test_size]
        )
    
    def get_train_loader(self, **kwargs) -> DataLoader:
        """Get training data loader."""
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            **kwargs
        }
        return DataLoader(self.train_dataset, **loader_kwargs)
    
    def get_valid_loader(self, **kwargs) -> DataLoader:
        """Get validation data loader."""
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            **kwargs
        }
        return DataLoader(self.valid_dataset, **loader_kwargs)
    
    def get_test_loader(self, **kwargs) -> DataLoader:
        """Get test data loader."""
        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            **kwargs
        }
        return DataLoader(self.test_dataset, **loader_kwargs)


# Utility functions for common use cases
def create_avmnist_compatible_loader(
    data_dir: str,
    batch_size: int = 40,
    flatten_audio: bool = False,
    flatten_image: bool = False,
    unsqueeze_channel: bool = True,
    normalize_image: bool = True,
    normalize_audio: bool = True
) -> FlexibleMultimodalDataLoader:
    """Create a flexible loader compatible with AVMNIST format."""
    
    # Define modality configurations
    image_config = ModalityConfig(
        name='image',
        data_path=f"{data_dir}/image/train_data.npy",  # Will need to be handled differently for splits
        flatten=flatten_image,
        normalize=normalize_image,
        unsqueeze_channel=unsqueeze_channel,
    )
    
    audio_config = ModalityConfig(
        name='audio',
        data_path=f"{data_dir}/audio/train_data.npy",  # Will need to be handled differently for splits
        flatten=flatten_audio,
        normalize=normalize_audio,
        unsqueeze_channel=unsqueeze_channel,
    )
    
    # Note: This is a simplified version. For full AVMNIST compatibility,
    # we'd need to handle train/test splits properly
    configs = [image_config, audio_config]
    
    dataset = FlexibleMultimodalDataset(configs)
    
    return FlexibleMultimodalDataLoader(
        dataset=dataset,
        batch_size=batch_size
    )


def create_custom_multimodal_loader(
    modality_paths: Dict[str, str],
    modality_configs: Optional[Dict[str, dict]] = None,
    batch_size: int = 32,
    **loader_kwargs
) -> FlexibleMultimodalDataLoader:
    """Create a multimodal loader from paths and optional configurations.
    
    Args:
        modality_paths (Dict[str, str]): Mapping of modality names to data paths
        modality_configs (Dict[str, dict], optional): Additional configuration for each modality
        batch_size (int): Batch size for data loaders
        **loader_kwargs: Additional arguments for FlexibleMultimodalDataLoader
    
    Returns:
        FlexibleMultimodalDataLoader: Configured multimodal data loader
    """
    configs = []
    
    for modality_name, path in modality_paths.items():
        config_params = {'name': modality_name, 'data_path': path}
        
        # Add any additional configuration
        if modality_configs and modality_name in modality_configs:
            config_params.update(modality_configs[modality_name])
        
        configs.append(ModalityConfig(**config_params))
    
    dataset = FlexibleMultimodalDataset(configs)
    
    return FlexibleMultimodalDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        **loader_kwargs
    )
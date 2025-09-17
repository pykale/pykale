# Flexible Multimodal Dataloader

This document describes the improved multimodal dataloader infrastructure in PyKale that provides flexible handling of variable multimodal data.

## Overview

The flexible multimodal dataloader addresses key limitations in existing multimodal dataset handling:

- **Fixed modality count**: Previous datasets like `AVMNISTDataset` only handled specific modality combinations
- **Rigid preprocessing**: Preprocessing options were hardcoded per dataset class
- **Limited extensibility**: Adding new modalities required creating new dataset classes
- **Inconsistent APIs**: Different dataset classes had different interfaces

## Key Components

### 1. ModalityConfig

A configuration class that defines how each modality should be handled:

```python
from kale.loaddata.flexible_multimodal import ModalityConfig

config = ModalityConfig(
    name='image',
    data_path='/path/to/images.npy',
    flatten=False,
    normalize=True,
    unsqueeze_channel=True,
    dtype=torch.float32,
    transform=custom_transform_function
)
```

**Parameters:**
- `name`: Identifier for the modality
- `data_path`: Path to the data file
- `transform`: Custom transformation function
- `flatten`: Whether to flatten the data
- `normalize`: Whether to normalize the data
- `unsqueeze_channel`: Whether to add a channel dimension
- `dtype`: Target PyTorch data type
- `shape`: Expected shape for validation

### 2. FlexibleMultimodalDataset

The main dataset class that handles multiple modalities:

```python
from kale.loaddata.flexible_multimodal import FlexibleMultimodalDataset, ModalityConfig

# Define configurations for each modality
configs = [
    ModalityConfig('image', '/path/to/images.npy', normalize=True),
    ModalityConfig('audio', '/path/to/audio.npy', flatten=True),
    ModalityConfig('text', '/path/to/text.npy', transform=text_embedding_fn)
]

# Create dataset
dataset = FlexibleMultimodalDataset(
    modality_configs=configs,
    validate_shapes=True,
    cache_data=False  # Set True for better performance with smaller datasets
)

# Add labels if available
labels = np.load('/path/to/labels.npy')
dataset.set_labels(labels)

# Access samples
sample = dataset[0]  # Returns [modality1_data, modality2_data, modality3_data]
# OR with labels:
modalities, label = dataset[0]  # Returns ([modality_data...], label)
```

**Key Features:**
- Variable number of modalities
- Per-modality preprocessing configuration
- Optional data caching for performance
- Shape validation
- Custom data loading functions
- Common transforms applied across modalities

### 3. FlexibleMultimodalDataLoader

Convenient wrapper for creating train/validation/test splits:

```python
from kale.loaddata.flexible_multimodal import FlexibleMultimodalDataLoader

loader = FlexibleMultimodalDataLoader(
    dataset=dataset,
    batch_size=32,
    train_split=0.8,
    valid_split=0.1,
    test_split=0.1,
    shuffle=True,
    num_workers=4
)

# Get data loaders
train_loader = loader.get_train_loader()
valid_loader = loader.get_valid_loader()
test_loader = loader.get_test_loader()

# Iterate through batches
for batch_modalities, batch_labels in train_loader:
    # batch_modalities is a list of tensors, one per modality
    images, audio, text = batch_modalities
    # Process multimodal data...
```

## Utility Functions

### create_custom_multimodal_loader()

Quick setup for common use cases:

```python
from kale.loaddata.flexible_multimodal import create_custom_multimodal_loader

# Simple setup
loader = create_custom_multimodal_loader(
    modality_paths={
        'image': '/path/to/images.npy',
        'audio': '/path/to/audio.npy',
        'text': '/path/to/text.npy'
    },
    batch_size=32
)

# With custom configurations
loader = create_custom_multimodal_loader(
    modality_paths={
        'image': '/path/to/images.npy',
        'audio': '/path/to/audio.npy'
    },
    modality_configs={
        'image': {'normalize': True, 'unsqueeze_channel': True},
        'audio': {'flatten': True, 'normalize': False}
    },
    batch_size=64,
    train_split=0.7,
    valid_split=0.2,
    test_split=0.1
)
```

### create_avmnist_compatible_loader()

Backward compatibility with existing AVMNIST usage patterns:

```python
from kale.loaddata.flexible_multimodal import create_avmnist_compatible_loader

# Drop-in replacement for AVMNISTDataset
loader = create_avmnist_compatible_loader(
    data_dir='/path/to/avmnist',
    batch_size=40,
    flatten_audio=True,
    flatten_image=False,
    normalize_image=True,
    normalize_audio=True
)
```

## Advanced Usage

### Custom Data Loading

For non-standard data formats:

```python
def custom_hdf5_loader(config):
    import h5py
    with h5py.File(config.data_path, 'r') as f:
        return f[config.name][:]

dataset = FlexibleMultimodalDataset(
    modality_configs=configs,
    data_loader=custom_hdf5_loader
)
```

### Custom Transforms

Per-modality and common transforms:

```python
def image_augment(images):
    # Apply random rotations, flips, etc.
    return augmented_images

def text_tokenize(text_data):
    # Convert text to token IDs
    return tokenized_data

def common_normalize(modalities):
    # Apply consistent normalization across all modalities
    return [torch.nn.functional.normalize(mod) for mod in modalities]

configs = [
    ModalityConfig('image', path, transform=image_augment),
    ModalityConfig('text', path, transform=text_tokenize)
]

dataset = FlexibleMultimodalDataset(
    modality_configs=configs,
    common_transform=common_normalize
)
```

### Performance Optimization

```python
# Enable caching for small datasets that fit in memory
dataset = FlexibleMultimodalDataset(
    modality_configs=configs,
    cache_data=True  # Loads all data into memory at initialization
)

# Use multiple workers for data loading
loader = FlexibleMultimodalDataLoader(
    dataset=dataset,
    num_workers=8,  # Use multiple processes for data loading
    batch_size=64
)
```

## Migration Guide

### From AVMNISTDataset

**Old code:**
```python
from kale.loaddata.avmnist_datasets import AVMNISTDataset

dataset = AVMNISTDataset(
    data_dir='/path/to/avmnist',
    batch_size=40,
    flatten_audio=True,
    normalize_image=True
)
train_loader = dataset.get_train_loader()
```

**New code (option 1 - direct migration):**
```python
from kale.loaddata.flexible_multimodal import create_avmnist_compatible_loader

loader = create_avmnist_compatible_loader(
    data_dir='/path/to/avmnist',
    batch_size=40,
    flatten_audio=True,
    normalize_image=True
)
train_loader = loader.get_train_loader()
```

**New code (option 2 - full flexibility):**
```python
from kale.loaddata.flexible_multimodal import ModalityConfig, FlexibleMultimodalDataset, FlexibleMultimodalDataLoader

configs = [
    ModalityConfig('image', f'{data_dir}/image/train_data.npy', normalize=True),
    ModalityConfig('audio', f'{data_dir}/audio/train_data.npy', flatten=True)
]

dataset = FlexibleMultimodalDataset(configs)
labels = np.load(f'{data_dir}/train_labels.npy')
dataset.set_labels(labels)

loader = FlexibleMultimodalDataLoader(dataset, batch_size=40)
train_loader = loader.get_train_loader()
```

### Adding New Modalities

**Easy extension:**
```python
# Add text modality to existing image+audio setup
configs = [
    ModalityConfig('image', f'{data_dir}/image/train_data.npy', normalize=True),
    ModalityConfig('audio', f'{data_dir}/audio/train_data.npy', flatten=True),
    ModalityConfig('text', f'{data_dir}/text/embeddings.npy', transform=text_transform)
]

dataset = FlexibleMultimodalDataset(configs)
# Now handles 3 modalities instead of just 2!
```

## Error Handling

The flexible dataloader provides comprehensive error checking:

```python
# Shape validation
config = ModalityConfig('image', path, shape=(28, 28))
dataset = FlexibleMultimodalDataset([config])  # Warns if shapes don't match

# Sample count validation
# Automatically checks that all modalities have the same number of samples

# Label validation
dataset.set_labels(wrong_length_labels)  # Raises ValueError

# Missing data handling
config = ModalityConfig('missing')  # Raises error if no data_path or custom loader
```

## Benefits

1. **Flexibility**: Handle any number of modalities with different preprocessing
2. **Consistency**: Unified API across different multimodal scenarios
3. **Extensibility**: Easy to add new modalities without code changes
4. **Performance**: Optional caching and multi-worker support
5. **Reliability**: Comprehensive error checking and validation
6. **Backward Compatibility**: Works with existing code patterns
7. **Maintainability**: Cleaner, more modular code structure

## Examples

See the `examples/` directory for complete usage examples:
- `multimodal_basic_usage.py` - Basic three-modality example
- `multimodal_custom_transforms.py` - Custom preprocessing examples
- `multimodal_performance_optimization.py` - Performance tuning examples
- `avmnist_migration.py` - Migration from AVMNISTDataset
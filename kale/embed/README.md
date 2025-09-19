# Embedding modules

Machine learning modules for representation learning, embedding, feature extraction, and feature selection.

## Structure of `kale.embed` APIs

```text
embed/
├── __init__.py
├── attention.py
├── cnn.py
├── factorization.py
├── gcn.py
├── model_lib/
│   ├── __init__.py
│   ├── drugban.py
│   ├── gripnet.py
│   └── mogonet.py
├── multimodal_encoder.py
├── multimodal_fusion.py
├── nn.py
├── uncertainty_fitting.py
└── video/
    ├── __init__.py
    ├── feature_extractor.py
    ├── i3d.py
    ├── res3d.py
    ├── se_i3d.py
    ├── se_res3d.py
    └── selayer.py
```

where:
- `embed.model_lib/`: Predefined models for specific applications (e.g., for reproducing the models designed in published papers).
- `embed.video/`: Modules for video feature extraction and processing.
- `attention.py`: General attention modules.
- `cnn.py`: General convolutional neural network modules.
- `factorization.py`: Matrix factorization/decomposition modules.
- `gcn.py`: General graph convolutional network modules.
- `multimodal_encoder.py`: Modules for encoding multimodal data.
- `multimodal_fusion.py`: Modules for fusing multimodal data.
- `nn.py`: General neural network modules.
- `uncertainty_fitting.py`: Modules for uncertainty estimation and fitting.

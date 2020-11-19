# Examples in computer vision - Domain adaptation for action recognition using PyTorch Lightning

### 1. Description
This example is constructed by refactoring the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada), with many domain adaptation algorithms included and is modified from `digits_dann_lightn`, by replacing the feature extractor and the data loaders for video data.

### 2. Usage

* Dataset: GTEA, KITCHEN, ADL(P04, P06, P11), EPIC(D1, D2, D3)
* Algorithms: DANN, CDAN, CDAN+E, ...
* Example:

`python main.py --cfg configs/EPIC-D12D2-DANN.yaml --gpus 1`

`python main.py --cfg configs/EPIC-D12D2-CDAN.yaml --gpus 1`

### 3. Related Kale core

`kale.embed.video_cnn`: Video feature extractor networks (Res3D, I3D)

`kale.loaddata.video_access`: Data loaders for video datasets

`kale.loaddata.video_data`: `basic_video_dataset` for loading data from GTEA, KITCHEN and ADL Datasets. `epickitchen` inherited from `basic_video_dataset` for EPIC-Kitchen dataset

`kale.prepdata.video_transform`: Transforms for video data

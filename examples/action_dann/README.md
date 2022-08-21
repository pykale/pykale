# Video Classification: Domain Adaptation for Action Recognition with Lightning

### 1. Description

This example is constructed by refactoring the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada), with many domain adaptation algorithms included and is modified from `digits_dann`, by replacing the feature extractor and the data loaders for video data.

### 2. Usage

* Datasets: GTEA, KITCHEN, ADL(P4, P6, P11), EPIC(D1, D2, D3)
* Algorithms: DANN, CDAN, DAN, ...
* Example:

For training (using 1 GPU):

`python main.py --cfg configs/EPIC-D12D2-DANN.yaml --gpus 1`

`python main.py --cfg configs/EPIC-D12D2-CDAN.yaml --gpus 1`

`python main.py --cfg configs/EPIC-D12D2-DAN.yaml --gpus 1`

For test:

`python test.py --cfg configs/EPIC-D12D2-DANN.yaml --gpus 1 --ckpt your_pretrained_model.ckpt `

### 3. Related `kale` API

`kale.embed.video_feature_extractor`: Get video feature extractor networks (Res3D, I3D, etc.).

`kale.embed.video_i3d`: Inflated 3D ConvNets (I3D) model for action recognition.

`kale.embed.video_res3d`: MC3_18, R3D_18, and R2plus1D_18 models for action recognition.

`kale.embed.video_se_i3d`: I3D model with SELayers.

`kale.embed.video_se_res3d`: MC3_18, R3D_18, and R2plus1D_18 models with SELayers.

`kale.embed.video_selayer`: Basic SELayers.

`kale.loaddata.action_multi_domain`: Construct the dataset for action videos with (multiple) source and target domains.

`kale.loaddata.video_access`: Data loaders for video datasets.

`kale.loaddata.video_datasets`: Based on `kale.loaddata.videos`. `BasicVideoDataset` for loading data from GTEA, KITCHEN and ADL Datasets. `EPIC` inherited from `BasicVideoDataset` for EPIC-Kitchen dataset.

`kale.loaddata.videos`: `VideoRecord` represents a video sample's metadata. `VideoFrameDataset` For loading video data effieciently.

`kale.pipeline.action_domain_adapter`: Domain adaptation pipelines for action video on action recognition.

`kale.predict.class_domain_nets`: Classifiers for data or domain.

`kale.prepdata.video_transform`: Transforms for video data.

# Multimodal Digit Classification: Multimodal Deep Learning model (MMDL) on AVMNIST

### 1. Description

This example is constructed by refactoring the [MultiBenchy repository](https://github.com/pliang279/MultiBench) for [Pykale library](https://github.com/pykale/pykale).

### 2. Usage
* Dataset: [AVMNIST](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view)
* Algorithm: MMDL
* Example: AVMNIST with MMDL

`python main.py --cfg configs/late_fusion.yaml --output AVMNIST-LATE`

### 3. Related `kale` API

`kale.pipeline.mmdl`: Multimodal Deep Learning model

`kale.loaddata.avmnist_datasets`: Data loaders for AVMNIST datasets.

`kale.embed.multimodal_common_fusions`: Some common fuison two fuse extracted features from 2 modalities.

`kale.embed.lenet`: Extract features from images and audios.

`kale.predict.two_layered_mlp`: Predict final out from the fused features. 

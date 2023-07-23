# Multimodal Digit Classification: Multimodal Neural Network (MMNN) Method on AVMNIST

### 1. Description

This example is constructed by refactoring the code of the [MultiBench repository](https://github.com/pliang279/MultiBench) for the [PyKale library](https://github.com/pykale/pykale).

### 2. Usage
* Dataset: [AVMNIST](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view)
<br>The Audio-Visual MNIST (AVMNIST) dataset is an innovative and unique extension on the classic MNIST dataset, intended to facilitate multimodal learning, where models are trained to interpret information from both visual and auditory inputs simultaneously. This is an interesting area of research, as many real-world applications involve multiple modalities of input, and models that can integrate and learn from these different modalities of data can potentially achieve more robust and accurate performance. AVMNIST was created by pairing audio clips from the Free Spoken Digit Dataset (FSDD) with the corresponding written digit images from the MNIST dataset. The FSDD is a collection of 3,000 recordings of spoken digits in English, where each clip consists of a person speaking a digit between 0 and 9. The MNIST dataset, on the other hand, consists of 70,000 grayscale images of hand-written digits between 0 and 9. Each sample in the AVMNIST dataset thus consists of an image-audio pair where the image is a handwritten digit from the MNIST dataset, and the corresponding audio is a clip of a person saying the same digit from the FSDD dataset. The task, therefore, is to predict the digit that will be between 0 and 9.
* Algorithm: Multimodal Neural Network (MMNN)
* Example: AVMNIST with MMNN

Note: In this example, we used a [small subset of the AVMNIST dataset](https://drive.google.com/file/d/1N5k-LvLwLbPBgn3GdVg6fXMBIR6pYrKb/view), specifically including only the samples with labels 0 and 1, to illustrate the usage of MMNN in predicting digit.

We provided some `yaml` config file for a quick testing in the `configs` folder. To use it, run:
```python
python main.py --cfg configs/late_fusion.yaml --output AVMNIST-LATE
```
or
```python
python main.py --cfg configs/low_rank_tensor_fusion.yaml --output AVMNIST_LOW_RANK_TENSOR
```
or
```python
python main.py --cfg configs/bimodal_interaction_fusion.yaml --output AVMNIST-BIMODAL_INTERACTION
```

### 3. Related `kale` API

`kale.loaddata.avmnist_datasets`: This is a data loading module specifically designed for handling the AVMNIST dataset. The AVMNIST dataset is a multimodal version of the traditional MNIST, containing both audio and visual data corresponding to the original MNIST data. This module provides efficient ways to load, preprocess, and format this data in a manner that makes it ready for training a multimodal neural network model.

`kale.embed.feature_fusion`: This module provides a comprehensive set of feature fusion methods, designed for integrating distinct modalities like image and audio. It utilizes a wide range of methods, from simple concatenation, addition, and multiplication fusion, to more sophisticated approaches such as low-rank tensor fusion. The choice of fusion method can greatly influence the performance of the resulting multimodal model.

`kale.embed.image_cnn.LeNet`: This is a base neural network trainer that allows the extraction of features from different modalities like images and audios.

`kale.pipeline.base_nn_trainer.MultimodalNNTrainer`: This module includes a model trainer that consists of a training loop, optimizer configuration, and evaluation metrics.

`kale.predict.decode`: This module is responsible for generating final predictions from the fused features. A two-layered MLPDecoder has been used to generate the final output.

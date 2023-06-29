# Multimodal Digit Classification: Multimodal Deep Learning (MMDL) Method on AVMNIST

### 1. Description

This example is constructed by refactoring the code of the [MultiBench repository](https://github.com/pliang279/MultiBench) for [Pykale library](https://github.com/pykale/pykale).

### 2. Usage
* Dataset: [AVMNIST](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view)
<br>The Audio-Visual MNIST (AVMNIST) dataset is an innovative and unique extension on the classic MNIST dataset, intended to facilitate multimodal learning, where models are trained to interpret information from both visual and auditory inputs simultaneously. This is an interesting area of research, as many real-world applications involve multiple modalities of input, and models that can integrate and learn from these different modalities of data can potentially achieve more robust and accurate performance. AV-MNIST was created by pairing audio clips from the Free Spoken Digit Dataset (FSDD) with the corresponding written digit images from the MNIST dataset. The FSDD is a collection of 3,000 recordings of spoken digits in English, where each clip consists of a person speaking a digit between 0 and 9. The MNIST dataset, on the other hand, consists of 70,000 grayscale images of hand-written digits between 0 and 9. Each sample in the AV-MNIST dataset thus consists of an image-audio pair where the image is a handwritten digit from the MNIST dataset, and the corresponding audio is a clip of a person saying the same digit from the FSDD dataset. The task, therefore, is to predict the digit in question, which is one of 10 classes (0 through 9).
* Algorithm: Multimodal Deep Learning (MMDL)
* Example: AVMNIST with MMDL

Note: In this example, we used a [small subset of the AV-MNIST dataset](https://drive.google.com/file/d/1N5k-LvLwLbPBgn3GdVg6fXMBIR6pYrKb/view), specifically including only the samples with labels 0 and 1, to illustrate the usage of MMDL in predicting digit.

We provided some `yaml` config file for a quick testing in the `configs` folder. To use it, run:
```python
python main.py --cfg configs/late_fusion.yaml --output AVMNIST-LATE
```
or
```python
python main.py --cfg configs/low_rank_tensor_fusion.yaml --output AVMNIST-LOW_RANK_TENSOR
```
or
```python
python main.py --cfg configs/bimodal_matrix_fusion_interactor.yaml --output AVMNIST-BIMODA_MATRIX
```

### 3. Related `kale` API

`kale.pipeline.multimodal_deep_learning`: This module provides a pipeline to perform multimodal deep learning. It's used for handling and processing data from multiple 'modalities' such as images, text, audio, etc.

`kale.loaddata.avmnist_datasets`: This is a data loading module specifically designed for handling the AVMNIST dataset. The AVMNIST dataset is a multimodal version of the traditional MNIST, containing both audio and visual data corresponding to the original MNIST data. This module provides efficient ways to load, preprocess, and format this data in a manner that makes it ready for training a multimodal deep learning model.

`kale.embed.feature_fusion`: This module provides a set of feature fusion techniques used for integrating the features extracted from different modalities, like image and audio. Fusion methods could include techniques such as concatenation, addition, multiplication, or more complex approaches like low-rank tensor fusion. The choice of fusion method can greatly influence the performance of the resulting multimodal model.

`kale.embed.image_cnn.LeNet`: This is a base neural network trainer that allows the extraction of features from different modalities like images and audios.

`kale.pipeline.base_nn_trainer.MultimodalTrainer`: This module includes a model trainer that consists of a training loop, optimizer configuration, and evaluation metrics.

`kale.predict.decode`: This module is responsible for generating final predictions from the fused features. A two-layered MLPDecoder has been used to generate the final output.

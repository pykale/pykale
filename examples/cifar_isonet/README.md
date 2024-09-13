# Image Classification: ISONet on CIFAR10/100

### 1. Description

This example is constructed by refactoring the [Deep Isometric Learning for Visual Recognition repository](https://github.com/HaozhiQi/ISONet) for an [ICML 2020 paper](http://proceedings.mlr.press/v119/qi20a.html).

### 2. Usage

* Dataset: [CIFAR10, CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* Algorithm: ISONet
* Example: CIFAR10 with ISONet38

`python main.py --cfg configs/CF10-ISO38.yaml --output CF10-ISO38`

### 3. Related `kale` API

`kale.embed.image_cnn`: Extract features from small-size (32x32) images using CNN.

`kale.loaddata.image_access`: Data loaders for CIFAR datasets.

`kale.predict.isonet`: ISONet classifier.

`kale.prepdata.image_transform`: Transforms for image data.

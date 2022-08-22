# Image Classification: Multi-source Domain Adaptation on Images (e.g. Office, Digits) with Lightning

### 1. Description

This example demonstrates multi-source domain adaptation methods with application in image recognition
on [office-caltech dataset](https://github.com/pykale/data/tree/main/images/office).

### 2. Usage

* Datasets:
  * Digits (10 classes), 4 domains: MNIST, Modified MNIST (MNISTM), UPSP, SVHN
  * Office-31 (31 classes), 3 domains: Amazon (A), DSLR (D), Webcam (W)
  * Office-Caltech (10 classes), 4 domains: Amazon (A), DSLR (D), Webcam (W), Caltech (C)
* Algorithms:
  * [Moment matching for multi-source domain adaptation (M3SDA)](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html) [[1](#4-references)][[Code](https://github.com/pykale/pykale/blob/main/kale/pipeline/multi_domain_adapter.py#L105)]
  * [Multiple Feature Spaces Adaptation Network (MFSAN)](https://ojs.aaai.org/index.php/AAAI/article/view/4551) [[2](#4-references)][[Code](https://github.com/pykale/pykale/blob/main/kale/pipeline/multi_domain_adapter.py#L245)]
* Example: Caltech, DSLR, and Webcam (three sources) to Amazon (target) using M3SDA and MFSAN

`python main.py --cfg configs/Office2A-M3SDA.yaml --gpus 1`

`python main.py --cfg configs/Office2A-MFSAN.yaml --gpus 1`

### 3. Related `kale` API

`kale.embed.image_cnn`: Extract features from small-size (32x32) images using CNN.

`kale.loaddata.image_access`: General APIs for data loaders of image datasets.

`kale.loaddata.multi_domain`: Construct the dataset for (multiple) source and target domains.

`kale.pipeline.multi_domain_adapter`: Multi-source domain adaptation pipelines for image classification.

`kale.predict.class_domain_nets`: Classifiers for data or domain.

`kale.prepdata.image_transform`: Transforms for image data.

### 4. References

[1] Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., & Wang, B. (2019). [Moment matching for multi-source domain adaptation](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html). In *ICCV 2019* (pp. 1406-1415).

[2] Zhu, Y., Zhuang, F. and Wang, D. (2019). [Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources](https://ojs.aaai.org/index.php/AAAI/article/view/4551). In *AAAI 2019* (pp. 5989-5996).

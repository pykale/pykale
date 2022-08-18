# PAH Diagnosis from Cardiac MRI via Multilinear PCA

## 1. Description

This example demonstrates the multilinear PCA-based machine learning pipeline for cardiac MRI analysis [1], with application in pulmonary arterial hypertension (PAH) diagnosis.

**Reference:**

[1] Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2021). [A machine learning cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial hypertension diagnosis](https://academic.oup.com/ehjcimaging/article/22/2/236/5717931). European Heart Journal-Cardiovascular Imaging.

## 2. Usage

* Datasets: [ShefPAH-179 v2.0 (short-axis) cardiac MRI dataset](https://github.com/pykale/data/tree/main/images/ShefPAH-179)
* Algorithms: MPCA + Linear SVM / Kernel SVM / Logistic Regression,...
* Example: Classification using SVM

`python main.py --cfg configs/tutorial_svc.yaml`

## 3. Related `kale` API

`kale.interpret.model_weights`: Get model weights for interpretation.

`kale.interpret.visualize`: Plot model weights or images.

`kale.loaddata.image_access`: Load DICOM images as ndarray data.

`kale.pipeline.mpca_trainer`: Pipeline of MPCA + feature selection + classification.

`kale.prepdata.image_transform`: CMR images pre-processing.

`kale.utils.download`: Download data.

# Drug-Target Interaction Prediction using DeepDTA

### 1. Description
Drug-target interaction prediction is an important research area in the field of drug discovery. It refers to predicting the binding affinity between the given chemical compounds and protein targets. In this example we train a standard DeepDTA model as a baseline in BindingDB, a public, web-accessible dataset of measured binding affinities.

### 2. DeepDTA
[DeepDTA](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245) is the modeling of protein sequences and compound 1D
representations with convolutional neural networks (CNNs). The whole architecture of DeepDTA is shown below.

![DeepDTA](https://github.com/hkmztrk/DeepDTA/blob/master/docs/figures/deepdta.PNG)

### 3. Datasets
We construct **three datasets** from BindingDB distinguished by different affinity measurement metrics
(**Kd, IC50 and Ki**). They are acquired from [Therapeutics Data Commons](https://tdcommons.ai/) (TDC), which is a collection of machine learning
tasks spread across different domains of therapeutics. The data statistics is shown:

|  Metrics   | Drugs | Targets | Pairs |
|  :----:  | :----:  |   :----:  | :----:  |
| Kd  | 10,655 | 1,413 | 52,284 |
| IC50  | 549,205 | 5,078 | 991,486 |
| Ki | 174,662 | 3,070 | 375,032 |

This figure is the binding affinity distribution for the three datasets respectively, and the metrics values (x-axis) have been transformed into
log space.
![Binding affinity distribution](figures/bindingdb.jpg)

### 4. Requirements
You'll need to install the external [RDKit](https://github.com/rdkit/rdkit) package for running the example codes.

```
conda install -c conda-forge rdkit
```

### 5. Usage
Run model for BindingDB datasets with IC50, Kd and Ki metrics respectively.
```
python main.py --cfg configs/IC50-DeepDTA.yaml
python main.py --cfg configs/Kd-DeepDTA.yaml
python main.py --cfg configs/Ki-DeepDTA.yaml
```

### 6. Results
Here are the MSE loss results for the three BindingDB datasets, and the minimal validation loss's epoch is saved as the
best checkpoint, which is applied to calculate test dataset loss. All default maximum epochs are 100.

|  Datasets   | valid_loss | test_loss | best_epoch |
|  :----:  | :----:  |   :----:  | :----:  |
| Kd  | 0.7898 | 0.7453 | 47 |
| IC50  | 0.9264 | 0.9198 | 83 |
| Ki | 1.071 | 1.072 | 91 |

### 7. Architecture
Below is the architecture of DeepDTA with default hyperparameters settings.

<pre>
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─CNNEncoder: 1-1                        [256, 96]                 --
|    └─Embedding: 2-1                    [256, 85, 128]            8,320
|    └─Conv1d: 2-2                       [256, 32, 121]            21,792
|    └─Conv1d: 2-3                       [256, 64, 114]            16,448
|    └─Conv1d: 2-4                       [256, 96, 107]            49,248
|    └─AdaptiveMaxPool1d: 2-5            [256, 96, 1]              --
├─CNNEncoder: 1-2                        [256, 96]                 --
|    └─Embedding: 2-6                    [256, 1200, 128]          3,328
|    └─Conv1d: 2-7                       [256, 32, 121]            307,232
|    └─Conv1d: 2-8                       [256, 64, 114]            16,448
|    └─Conv1d: 2-9                       [256, 96, 107]            49,248
|    └─AdaptiveMaxPool1d: 2-10           [256, 96, 1]              --
├─MLPDecoder: 1-3                        [256, 1]                  --
|    └─Linear: 2-11                      [256, 1024]               197,632
|    └─Dropout: 2-12                     [256, 1024]               --
|    └─Linear: 2-13                      [256, 1024]               1,049,600
|    └─Dropout: 2-14                     [256, 1024]               --
|    └─Linear: 2-15                      [256, 512]                524,800
|    └─Linear: 2-16                      [256, 1]                  513
==========================================================================================
Total params: 2,244,609
Trainable params: 2,244,609
Non-trainable params: 0
Total mult-adds (M): 58.08
==========================================================================================
Input size (MB): 1.32
Forward/backward pass size (MB): 429.92
Params size (MB): 8.98
Estimated Total Size (MB): 440.21

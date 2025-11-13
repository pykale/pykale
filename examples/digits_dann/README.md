# Image Classification: Domain Adaptation on Digits with Lightning

## 1. Description

This example is constructed by refactoring the [ADA: (Yet) Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada), with many domain adaptation algorithms included.

## 2. Usage

* Datasets: MNIST, Modified MNIST (MNISTM), UPSP
* Algorithms: DAN, JAN, DANN, CDAN, WDGRL, FewShotDANN, MME, and more. See the [PyKale docs](https://pykale.readthedocs.io/en/latest/kale.pipeline.html#kale.pipeline.domain_adapter.Method) for more details.
* Example: MNIST (source) to UPSP (target) using CDAN and DANN

`python main.py --cfg configs/MN2UP-CDAN.yaml --devices 1`

`python main.py --cfg configs/MN2UP-DANN.yaml --devices 1`

* Please refer to [`configs.py`](./config.py) for more configurations.

## 3. Related `kale` API

`kale.embed.image_cnn`: Extract features from small-size (32x32) images using CNN.

`kale.loaddata.image_access`: Data loaders for digits datasets.

`kale.loaddata.mnistm`: Data loader for the [modified MNIST data](https://github.com/zumpchke/keras_mnistm).

`kale.loaddata.multi_domain`: Construct the dataset for (multiple) source and target domains.

`kale.loaddata.usps`: Data loader for the [USPS data](https://git-disl.github.io/GTDLBench/datasets/usps_dataset/).

`kale.pipeline.domain_adapter`: Domain adaptation pipelines for image classification.

`kale.predict.class_domain_nets`: Classifiers for data or domain.

`kale.prepdata.image_transform`: Transforms for image data.

## 4. *Sample* output CSV of 10 runs for reference

|    |                     |                     |                     |      |        |            |
|----|---------------------|---------------------|---------------------|------|--------|------------|
|    | source acc          | target acc          | domain acc          | seed | method | split      |
| 0  | 0.9879999894183128  | 0.9251666567579376  | 0.4914166614034912  | 2020 | CDAN   | Validation |
| 1  | 0.9886868991015944  | 0.9240404324664268  | 0.4774747621631832  | 2020 | CDAN   | Test       |
| 2  | 0.9888333227427212  | 0.915999990189448   | 0.4176666621933691  | 2021 | CDAN   | Validation |
| 3  | 0.988585888997477   | 0.9242424526746618  | 0.4765151661740674  | 2021 | CDAN   | Test       |
| 4  | 0.9898333227320107  | 0.9173333235085008  | 0.4382499953062505  | 2022 | CDAN   | Validation |
| 5  | 0.9884848788933596  | 0.9236363920499572  | 0.475858600497304   | 2022 | CDAN   | Test       |
| 6  | 0.9903333227266558  | 0.9256666567525826  | 0.4074166623031488  | 2023 | CDAN   | Validation |
| 7  | 0.988585888997477   | 0.9230303314252524  | 0.4785858733084751  | 2023 | CDAN   | Test       |
| 8  | 0.9899999893968924  | 0.9214999901305418  | 0.3995833290537121  | 2024 | CDAN   | Validation |
| 9  | 0.988585888997477   | 0.9237374021540744  | 0.4785858733084751  | 2024 | CDAN   | Test       |
| 10 | 0.9903333227266558  | 0.9161666568543296  | 0.3934999957855325  | 2025 | CDAN   | Validation |
| 11 | 0.988585888997477   | 0.9231313415293698  | 0.4771212267987721  | 2025 | CDAN   | Test       |
| 12 | 0.9903333227266558  | 0.9168333235138562  | 0.3583333294955082  | 2026 | CDAN   | Validation |
| 13 | 0.988585888997477   | 0.9228283112170176  | 0.4766161762781849  | 2026 | CDAN   | Test       |
| 14 | 0.9891666560724844  | 0.9136666568811052  | 0.37308332933753263 | 2027 | CDAN   | Validation |
| 15 | 0.9887879092057121  | 0.9235353819458396  | 0.4776262773193594  | 2027 | CDAN   | Test       |
| 16 | 0.9894999894022476  | 0.9154999901948032  | 0.4342499953490915  | 2028 | CDAN   | Validation |
| 17 | 0.9887879092057121  | 0.9234343718417222  | 0.4766666813302436  | 2028 | CDAN   | Test       |
| 18 | 0.9903333227266558  | 0.9254999900877008  | 0.3904166624852223  | 2029 | CDAN   | Validation |
| 19 | 0.9884848788933596  | 0.9228283112170176  | 0.4779798126837704  | 2029 | CDAN   | Test       |
| 20 | 0.11333333607763052 | 0.12000000290572645 | 0.5000000121071935  | 2020 | CDAN   | Validation |

## 2025 PyTorch 2.0+ new results

### MNIST → USPS, Source-only

| Seed | Test Source Acc | Test Target Acc | Domain Acc |
|------|-----------------|------------------|------------|
| 2025 | 0.9938 | 0.5183 | 1.0 |
| 2035 | 0.9934 | 0.5246 | 1.0 |
| 2045 | 0.9936 | 0.4000 | 1.0 |
| 2055 | 0.9941 | 0.5144 | 1.0 |
| 2065 | 0.9931 | 0.5175 | 1.0 |
| 2075 | 0.9929 | 0.5300 | 1.0 |
| 2085 | 0.9939 | 0.4498 | 1.0 |
| 2095 | 0.9924 | 0.5202 | 1.0 |
| 2105 | 0.9931 | 0.4268 | 1.0 |
| 2115 | 0.9931 | 0.5548 | 1.0 |

### MNIST → USPS, CDAN

| Seed | Test Source Acc | Test Target Acc | Domain Acc |
|------|-----------------|------------------|------------|
| 2025 | 0.9919 | 0.8338 | 0.4186 |
| 2035 | 0.9929 | 0.7676 | 0.4447 |
| 2045 | 0.9927 | 0.7804 | 0.5569 |
| 2055 | 0.9917 | 0.8039 | 0.5783 |
| 2065 | 0.9892 | 0.8035 | 0.5370 |
| 2075 | 0.9932 | 0.8371 | 0.5627 |
| 2085 | 0.9899 | 0.7978 | 0.4238 |
| 2095 | 0.9919 | 0.7727 | 0.3091 |
| 2105 | 0.9916 | 0.7966 | 0.3510 |
| 2115 | 0.9902 | 0.7431 | 0.4143 |

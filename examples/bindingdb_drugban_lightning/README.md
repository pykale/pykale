# Drug-target Interaction Prediction: Interpretable bilinear attention network with domain adaptation on BindingDB/BioSNAP/Human dataset

### 1. Description

This example is constructed by refactoring the [Interpretable bilinear attention network with domain adaptation improves drug-target prediction repository](https://github.com/peizhenbai/DrugBAN) for a [Nature Machine Intelligence 2023 paper](https://www.nature.com/articles/s42256-022-00605-1).

This is a ``PyTorch Lightning <https://github.com/Lightning-AI/lightning>`` version of the original implementation. 

### 2. Usage

### Datasets
For downloading the datasets, please refer to the following links: https://github.com/pykale/data/blob/add-drugban-data/molecular/README.md

We have three datasets in this collection:
- [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp)
- [BioSNAP](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets)
- [Human](https://github.com/lifanchen-simm/transformerCPI?tab=readme-ov-file#data-sets)

where BindingDB and BioSNAP are split into random and cluster splits, while Human is split into cold and random splits. The full dataset is also provided for each dataset.

All datasets should be organised as follows:

```sh
└───root
    ├───datasets
    │   ├───bindingdb
    │   │   ├───cluster
    │   │   │   ├───source_train.csv
    │   │   │   ├───target_train.csv
    │   │   │   ├───target_test.csv
    │   │   ├───random
    │   │   │   ├───test.csv
    │   │   │   ├───train.csv
    │   │   │   ├───val.csv
    │   │   ├───full.csv
    │   ├───biosnap
    │   │   ├───cluster
    │   │   │   ├───source_train.csv
    │   │   │   ├───target_train.csv
    │   │   │   ├───target_test.csv
    │   │   ├───random
    │   │   │   ├───test.csv
    │   │   │   ├───train.csv
    │   │   │   ├───val.csv
    │   │   ├───full.csv
    │   ├───human
    │   │   ├───cold
    │   │   │   ├───test.csv
    │   │   │   ├───train.csv
    │   │   │   ├───val.csv
    │   │   ├───random
    │   │   │   ├───test.csv
    │   │   │   ├───train.csv
    │   │   │   ├───val.csv
    │   │   ├───full.csv
```


### Examples

* Dataset: [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp), [BioSNAP](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets) or [Human](https://github.com/lifanchen-simm/transformerCPI?tab=readme-ov-file#data-sets)
* Data Split: random or cold
* Algorithm: DrugBAN
* Example: BindingDB with random splitting using DrugBAN

`python main.py --cfg "configs/DrugBAN.yaml" --data bindingdb --split random`


* Dataset: [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp) or [BioSNAP](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets)
* Data Split: cluster
* Algorithm: DrugBAN
* Example: BindingDB with cluster splitting using DrugBAN

`python main.py --cfg "configs/DrugBAN_Non_DA.yaml" --data "bindingdb" --split "cluster"`


* Dataset: [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp) or [BioSNAP](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets)
* Data Split: cluster
* Algorithm: DrugBAN with CDAN
* Example: BindingDB with cluster splitting using DrugBAN with CDAN

`python main.py --cfg "configs/DrugBAN_DA.yaml" --data "bindingdb" --split "cluster"`

### 3. Related `kale` API

`kale.embed.ban`: Extract features from drugs using GCN and proteins using CNN, fuse them with bilinear attention, and predict interactions.

`kale.loaddata.drugban_datasets`: Data loaders for DrugBAN datasets.

`kale.pipeline.drugban_trainer`: Pipelines for drug-target interaction prediction.

`kale.predict.class_domain_nets`: Classifier for domain.

`kale.prepdata.chem_transform`: Encode protein sequences.

`kale.pipeline.domain_adapter`: Gradient reversal layer (GRL) for domain adaptation.

`kale.predict.losses`: Compute different types of loss functions for neural network outputs.

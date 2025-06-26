# Drug-target Interaction Prediction: Interpretable bilinear attention network with domain adaptation on BindingDB/BioSNAP dataset

### 1. Description

This example is constructed by refactoring the [Interpretable bilinear attention network with domain adaptation improves drug-target prediction repository](https://github.com/peizhenbai/DrugBAN) for a [Nature Machine Intelligence 2023 paper](https://www.nature.com/articles/s42256-022-00605-1).

### 2. Usage

### Datasets
For downloading the datasets, please refer to the following links: https://github.com/pykale/data/blob/main/molecular/README.md

We have two datasets in this collection:
- [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp)
- [BioSNAP](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets)

where BindingDB and BioSNAP are split into random and cluster splits. The full dataset is also provided for each dataset.

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

```


### Examples

For in-domain prediction tasks using DrugBAN without domain adaptation, run the following command:

`python main.py --cfg "configs/non_DA_in_domain.yaml"`

* `non_DA_in_domain.yaml` is a configuration file, and contains all changeable hyperparameters. It allows users to customize the model accordingly.
* Change `DATA.DATASET` to fit your dataset directory. Available dataset examples are [`bindingdb`](https://www.bindingdb.org/rwd/bind/index.jsp) and [`biosnap`](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets).
* Change `DATA.SPLIT` to fit your data splitting strategy.

<br>

For cross-domain prediction tasks using DrugBAN without domain adaptation, run the following command:

`python main.py --cfg "configs/non_DA_cross_domain.yaml"`

* `non_DA_in_domain.yaml` is a configuration file, and contains all changeable hyperparameters. It allows users to customize the model accordingly.
* Change `DATA.DATASET` to fit your dataset directory. Available dataset examples are [`bindingdb`](https://www.bindingdb.org/rwd/bind/index.jsp), [`biosnap`](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets).

<br>

For cross-domain prediction tasks using DrugBAN with domain adaptation, run the following command:

`python main.py --cfg "configs/DA_cross_domain.yaml"`

* `DA_cross_domain.yaml` is a configuration file, and contains all changeable hyperparameters. It allows users to customize the model accordingly.
* Change `DATA.DATASET` to fit your dataset directory. Available dataset examples are [`bindingdb`](https://www.bindingdb.org/rwd/bind/index.jsp), [`biosnap`](https://github.com/kexinhuang12345/MolTrans?tab=readme-ov-file#datasets).

<br>

The use of `cfg.DA.TASK` in `*.yaml` files is as follows:
* `cfg.DA.TASK = False` refers to **in-domain** splitting strategy, where each experimental dataset is randomly divided into training, validation and test sets with a 7:1:2 ratio.
* `cfg.DA.TASK = True` refers to **cross-domain** splitting strategy, where the single-linkage algorithm is used to cluster drugs and proteins, and randomly selected 60% of the drug clusters and 60% of the protein clusters.
  * All drug-protein pairs in the selected clusters are source domain data. The remaining drug-protein pairs are target domain data.
  * In the setting of domain adaptation, all labelled source domain data and 80% unlabelled target domain data are used for training. The remaining 20% labelled target domain data are used for testing.


### 3. Related `kale` API

`kale.embed.ban`: Extract features from drugs using GCN and proteins using CNN, fuse them with bilinear attention, and predict interactions.

`kale.loaddata.molecular_datasets.py`: Data loaders for DrugBAN datasets.

`kale.pipeline.drugban_trainer`: Pipelines for drug-target interaction prediction. Conditional domain adversarial network (CDAN) can be applied for cross-domain prediction tasks.

`kale.pipeline.domain_adapter`: Gradient reversal layer (GRL) for domain adaptation.

`kale.predict.class_domain_nets`: Classifier for domain.

`kale.predict.losses`: Compute different types of loss functions for neural network outputs.

`kale.prepdata.chem_transform`: Encode protein sequences.

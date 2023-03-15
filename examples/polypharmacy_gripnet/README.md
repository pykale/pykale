# Polypharmacy Side Effect Prediction using GripNet (Link Prediction)

## 1. Description

Polypharmacy is the concurrent use of multiple medications by patients. Its side effects, polypharmacy side effects, are caused by drug combinations rather than by any single drug. In this example, we train a [GripNet model](https://doi.org/10.1016/j.patcog.2022.108973) [1] for predicting polypharmacy side effects.

## 2. GripNet

**Gr**aph **i**nformation **p**ropagation **Net**work ([GripNet](https://doi.org/10.1016/j.patcog.2022.108973)) is an effective and efficient framework for learning node representations on knowledge graphs (Subfigure A) for link prediction and node classification. It introduces a novel supergraph structure (Subfigure B) and constructs a graph neural network that passes messages via a task-specific propagation path based on the supergraph (Subfigure C).

![GripNet Architecture](https://ars.els-cdn.com/content/image/1-s2.0-S0031320322004538-gr2_lrg.jpg)

The original implementation of GripNet is [here](https://github.com/NYXFLOWER/GripNet.git).

## 3. Dataset

The [GripNet](https://doi.org/10.1016/j.patcog.2022.108973) paper studied polypharmacy side effects on the dataset constructed by [Zitnik et al. [2]](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770?login=false) integrating three datasets. This integrated dataset can be regarded as a [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph). It contains the relationships between proteins (P-P), proteins and drugs (P-D), and drugs, labeled with the P-P association, P-D association, and 1,317 side effects, respectively. The data statistics are shown in the table below. A standard usage should include a training set, a validation set, and a test set. The lack of a validation set during model fitting (i.e., training) may cause a warning from `pytorch_lightning`.

| Dataset           | Nodes            | Edges      | #Unique Edge Labels |
| ----------------- | ---------------- | ---------- | ------------------- |
| PP-Decagon        | 19,081(P)        | 715,612    | 1                   |
| ChG-TargetDecagon | 3,648(P), 284(D) | 18,690     | 1                   |
| ChChSe-Decagon    | 645(D)           | 4,649,441  | 1,317               |

Note: In this example, we use a [small subset of the integrated dataset above](https://github.com/pykale/data/blob/main/graphs/pose_pyg_2.pt) to illustrate the usage of GripNet in predicting polypharmacy side effects. Here, we reuse the training set as the validation and test sets for the purpose of demonstration only. We will update this example when an improved example dataset becomes available at a later time.

## 4. Usage

- Dataset download: [[here](https://github.com/pykale/data/tree/main/graphs)]

- Algorithm: GripNet

```python
python main.py
```

We provide a `yaml` config file for a quick testing in the `configs` folder. To use it, run:

```python
python main.py --cfg configs/PoSE_MINI-GripNet.yaml
```

## References

[1] Xu, H., Sang, S., Bai, P., Li, R., Yang, L., & Lu, H. (2022). [GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs](https://doi.org/10.1016/j.patcog.2022.108973). Pattern Recognition.

[2] Zitnik, M., Agrawal, M., & Leskovec, J. (2018). [Modeling polypharmacy side effects with graph convolutional networks](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770?login=false). Bioinformatics, 34(13), i457-i466.

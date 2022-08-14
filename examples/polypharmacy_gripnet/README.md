# Polypharmacy Side Effect Prediction using GripNet (Link Prediction)

## 1. Description

Polypharmacy is the concurrent use of multiple medications by patients. Its side effects, polypharmacy side effects, are caused by drug combinations rather than by any single drug. In this example, we train a GripNet model [[1]](https://doi.org/10.1016/j.patcog.2022.108973) for predicting polypharmacy side effects.

## 2. GripNet

**Gr**aph **i**nformation **p**ropagation **Net**work ([GripNet](#reference)) is an effective and efficient framework for learning node representations on knowledge graphs (Subfigure A) for link prediction and node classification. It introduces a novel supergraph structure (Subfigure B) and constructs a graph neural network that passes messages via a task-specific propagation path based on the supergraph (Subfigure C).

![GripNet Architecture](https://ars.els-cdn.com/content/image/1-s2.0-S0031320322004538-gr2_lrg.jpg)

The original implementation of GripNet is [here](https://github.com/NYXFLOWER/GripNet.git).

## 3. Dataset

We train GripNet on three datasets constructed by [[2]](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770?login=false).
The datasets are downloaded from the [Bio-SNAP](http://snap.stanford.edu/biodata/) dataset collection. The integration of these datasets can be regarded as a knowledge graph, which contains the relations among proteins (P) and drugs (D) labeled with the P-P association, the P-D association and 1,317 side effects. The data statistics are shown below.

| Dataset           | Nodes            | Edges   | #Unique Edge Labels |
| ----------------- | ---------------- | ------- | ------------------- |
| PP-Decagon        | 19,081(P)        | 715,612 | 1                   |
| ChG-TargetDecagon | 3,648(P), 284(D) | 18,690  | 1                   |
| ChChSe-Decagon    | 645(D)           | 63,473  | 1,317               |

## 4. Usage
- Dataset download: [[here](https://github.com/pykale/data/tree/main/graphs)]

- Algorithm: GripNet

`python main.py --cfg configs/Drug_MINI-GripNet.yaml`

## References:

[1] Xu, H., Sang, S., Bai, P., Li, R., Yang, L., & Lu, H. (2022). [GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs](https://doi.org/10.1016/j.patcog.2022.108973). Pattern Recognition.

[2] Zitnik, M., Agrawal, M., & Leskovec, J. (2018). [Modeling polypharmacy side effects with graph convolutional networks](https://academic.oup.com/bioinformatics/article/34/13/i457/5045770?login=false). Bioinformatics, 34(13), i457-i466.
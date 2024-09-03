# Interpretable bilinear attention network with domain adaptation improves drug-target prediction | [Paper](https://doi.org/10.1038/s42256-022-00605-1)

<div align="left">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pz-white/DrugBAN/blob/main/drugban_demo.ipynb)
[![DOI:10.1038/s42256-022-00605-1](https://zenodo.org/badge/DOI/10.1038/s42256-022-00605-1.svg)](https://doi.org/10.1038/s42256-022-00605-1)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/peizhenbai/DrugBAN/blob/main/LICENSE.md)
</div>


## Introduction
This repository contains the PyTorch implementation of **DrugBAN** framework, as described in our *Nature Machine Intelligence* paper "[Interpretable bilinear attention network with domain adaptation improves drug–target prediction](https://doi.org/10.1038/s42256-022-00605-1)".  **DrugBAN** is a deep bilinear attention network (BAN) framework with adversarial domain adaptation to explicitly learn pair-wise local interactions between drugs and targets,
and adapt on out-of-distribution data. It works on two-dimensional (2D) drug molecular graphs and target protein sequences to perform prediction.
## Framework
![DrugBAN](image/DrugBAN.jpg)
## System Requirements
The source code developed in Python 3.8 using PyTorch 2.3.1. The required python dependencies are given below. DrugBAN is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
torch>=2.3.1
dgl>=2.3.0
dgllife>=0.3.2
numpy>=1.24.4
scikit-learn>=1.3.2
pandas>=2.0.3
prettytable>=3.10.2
rdkit~=2024.3.5
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```

## Datasets
The `datasets` folder contains all experimental data used in DrugBAN: [BindingDB](https://www.bindingdb.org/bind/index.jsp) [1], [BioSNAP](https://github.com/kexinhuang12345/MolTrans) [2] and [Human](https://github.com/lifanchen-simm/transformerCPI) [3].
In `datasets/bindingdb` and `datasets/biosnap` folders, we have full data with two random and clustering-based splits for both in-domain and cross-domain experiments.
In `datasets/human` folder, there is full data with random split for the in-domain experiment, and with cold split to alleviate ligand bias.

## Demo
We provide DrugBAN running demo through a cloud Jupyter notebook on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pz-white/DrugBAN/blob/main/drugban_demo.ipynb). Note it is based on a small sample dataset of bindingdb due to the resource limitation of a free colab account. This demo only takes 3 minutes to complete the training and testing process. For running DrugBAN on the full dataset, we advise GPU ram >= 8GB and CPU ram >= 16GB.

The **expected output and run time** of demo has been provided in the colab notebook for verification.


## Run DrugBAN on Our Data to Reproduce Results

To train DrugBAN, where we provide the basic configurations for all hyperparameters in `config.py`. For different in-domain and cross-domain tasks, the customized task configurations can be found in respective `configs/*.yaml` files.

For the in-domain experiments with vanilla DrugBAN, you can directly run the following command. `${dataset}` could either be `bindingdb`, `biosnap` and `human`. `${split_task}` could be `random` and `cold`.
```
$ python main.py --cfg "configs/DrugBAN.yaml" --data ${dataset} --split ${split_task}
```

For the cross-domain experiments with vanilla DrugBAN, you can directly run the following command. `${dataset}` could beither `bindingdb`, `biosnap`.
```
$ python main.py --cfg "configs/DrugBAN_Non_DA.yaml" --data ${dataset} --split "cluster"
```
For the cross-domain experiments with CDAN DrugBAN, you can directly run the following command. `${dataset}` could beither `bindingdb`, `biosnap`.
```
$ python main.py --cfg "configs/DrugBAN_DA.yaml" --data ${dataset} --split "cluster"
```

## Comet ML
[Comet ML](https://www.comet.com/site/) is an online machine learning experimentation platform, which help researchers to track and monitor their ML experiments. We provide Comet ML support to easily monitor training process in our code.
This is **optional to use**. If you want to apply, please follow:

- Sign up [Comet](https://www.comet.com/site/) account and install its package using `pip3 install comet_ml`.

- Save your generated API key into `.comet.config` in your home directory, which can be found in your account setting. The saved file format is as follows:

```
[comet]
api_key=YOUR-API-KEY
```

- Set `_C.COMET.USE` to `True` and change `_C.COMET.WORKSPACE` in `config.py` into the one that you created on Comet.




For more details, please refer the [official documentation](https://www.comet.com/docs/python-sdk/advanced/).

## Acknowledgements
This implementation is inspired and partially based on earlier works [2], [4] and [5].

## Citation
Please cite our [paper](https://arxiv.org/abs/2208.02194) if you find our work useful in your own research.
```
    @article{bai2023drugban,
      title   = {Interpretable bilinear attention network with domain adaptation improves drug-target prediction},
      author  = {Peizhen Bai and Filip Miljkovi{\'c} and Bino John and Haiping Lu},
      journal = {Nature Machine Intelligence},
      year    = {2023},
      publisher={Nature Publishing Group},
      doi     = {10.1038/s42256-022-00605-1}
    }
```

## References
    [1] Liu, Tiqing, Yuhmei Lin, Xin Wen, Robert N. Jorissen, and Michael K. Gilson (2007). BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities. Nucleic acids research, 35(suppl_1), D198-D201.
    [2] Huang, Kexin, Cao Xiao, Lucas M. Glass, and Jimeng Sun (2021). MolTrans: Molecular Interaction Transformer for drug–target interaction prediction. Bioinformatics, 37(6), 830-836.
    [3] Chen, Lifan, et al (2020). TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. Bioinformatics, 36(16), 4406-4414.
    [4] Kim, Jin-Hwa, Jaehyun Jun, and Byoung-Tak Zhang (2018). Bilinear attention networks. Advances in neural information processing systems, 31.
    [5] Haiping Lu, Xianyuan Liu, Shuo Zhou, Robert Turner, Peizhen Bai, ... & Hao Xu (2022). PyKale: Knowledge-Aware Machine Learning from Multiple Sources in Python. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM).

# Multiomics Data Integration using Graph Convolutional Networks (MOGONET)

## 1. Description

Multimodal learning for multiomics data analysis is a promising research area in biomedical studies. Integrating multiple modalities (e.g., genomics, epigenomics, transcriptomics, proteomics, metabolomics, etc.) captures their complementary information and provides a deeper understanding of most complex human diseases. This example is constructed by refactoring the code of [MOGONET](https://doi.org/10.1038/s41467-021-23774-w) [1], a multiomics integrative method for cancer classification tasks, using the PyTorch Geometric and PyTorch Lightning frameworks.


## 2. MOGONET

**M**ulti-**O**mics **G**raph c**O**nvolutional **NET**works ([MOGONET](https://doi.org/10.1038/s41467-021-23774-w)) is
a multiomics fusion framework for cancer classification and biomarker identification that utilizes supervised graph
convolutional networks for omics datasets. The overall framework of MOGONET is illustrated below.

![MOGONET Architecture](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-021-23774-w/MediaObjects/41467_2021_23774_Fig1_HTML.png)

The original implementation of MOGONET is available [here](https://github.com/txWang/MOGONET).

## 3. Dataset

We have tested the MOGONET method using two preprocessed multiomics benchmarks, ROSMAP and BRCA, which have been provided
by the authors of the MOGONET paper in their repository. A brief description of these datasets is shown in the following
tables.

**Table 1**: Characteristics of the preprocessed ROSMAP multiomics dataset.

|      Omics       | #Training samples | #Test samples | #Features  |
|:----------------:|:-----------------:|:-------------:|:----------:|
| mRNA expression  |        245        |      106      |    200     |
| DNA methylation  |        245        |      106      |    200     |
| miRNA expression |        245        |      106      |    200     |



**Table 2**: Characteristics of the preprocessed BRCA multiomics dataset.

|      Omics       | #Training samples | #Test samples | #Features |
|:----------------:|:-----------------:|:-------------:|:---------:|
| mRNA expression  |        612        |      263      |   1000    |
| DNA methylation  |        612        |      263      |   1000    |
| miRNA expression |        612        |      263      |    503    |

Note: These datasets have been processed following the **Preprocessing** section of the original paper.

## 4. Usage

* Datasets: [BRCA, ROSMAP](https://github.com/pykale/data/tree/main/multiomics)
* Algorithm: MOGONET

Run the MOGONET model for the BRCA and ROSMAP datasets using `yaml` configuration files provided in the `configs` folder
for quick testing. To use them, run:

`python main.py --cfg configs/MOGONET_BRCA_quick_test.yaml`

`python main.py --cfg configs/MOGONET_ROSMAP_quick_test.yaml`


## Reference

[1] Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021). [MOGONET integrates multi-omics data
using graph convolutional networks allowing patient classification and biomarker identification](https://doi.org/10.1038/s41467-021-23774-w). Nature Communications.

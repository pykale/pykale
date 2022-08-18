# Autism Detection: Domain Adaptation for Multi-Site Neuroimaging Data Analysis

### 1. Description

This example demonstrates multi-source domain adaptation method with application in neuroimaging data analysis for
autism detection.

### 2. Materials and Methods

- Data: Four largest subsets of ABIDE I

| Site   | Number of Samples |
|--------|-------------------|
| NYU    | 175               |
| UM_1   | 106               |
| UCLA_1 | 72                |
| USM    | 71                |
- Atlas: CC200
- Pre-processing pipeline: cpac
- Classification problem: Autism vs Control
- Pipeline:
  1. Constructing brain networks from resting-state fMRI data
  2. Classification with Ridge Classifier or Covariate Independence Regularized Least Squares (CoIRLS) classifier


### 3. Related `kale` API

`kale.interpret.visualize`: Visualize the results of a model.

`kale.pipeline.multi_domain_adapter.CoIRLS`: Covariate Independence Regularized Least Squares (CoIRLS) classifier.

`kale.utils.download.download_file_by_url`: Download a file from a URL.

### References

[1] Cameron Craddock, Yassine Benhajali, Carlton Chu, Francois Chouinard, Alan Evans, András Jakab, Budhachandra Singh
Khundrakpam, John David Lewis, Qingyang Li, Michael Milham, Chaogan Yan, Pierre Bellec (2013). The Neuro Bureau
Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives. In *Neuroinformatics 2013*,
Stockholm, Sweden.

[2] Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., ... & Varoquaux, G. (2014).
Machine learning for neuroimaging with scikit-learn. *Frontiers in neuroinformatics*, 14.

[3] Zhou, S., Li, W., Cox, C.R., & Lu, H. (2020). [Side Information Dependence as a Regularizer for Analyzing Human Brain Conditions across Cognitive Experiments](https://ojs.aaai.org//index.php/AAAI/article/view/6179). in *AAAI 2020*, New York, USA.

[4] Zhou, S. (2022). [Interpretable Domain-Aware Learning for Neuroimage Classification](https://etheses.whiterose.ac.uk/31044/1/PhD_thesis_ShuoZhou_170272834.pdf) (Doctoral dissertation, University of Sheffield).

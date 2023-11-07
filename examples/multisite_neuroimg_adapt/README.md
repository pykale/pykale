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

[1] Craddock C., Benhajali Y., Chu C., Chouinard F., Evans A., Jakab A., Khundrakpam BS., Lewis JD., Li Q., Milham M., Yan C. and Bellec P. (2013). [The Neuro Bureau Preprocessing Initiative: Open Sharing of Preprocessed Neuroimaging Data and Derivatives](https://doi.org/10.3389/conf.fninf.2013.09.00041). Frontiers in Neuroinformatics, 7.

[2] Abraham A., Pedregosa F., Eickenberg M., Gervais P., Mueller A., Kossaifi J., Gramfort A., Thirion B. and Varoquaux G. (2014). [Machine Learning for Neuroimaging with scikit-learn](https://doi.org/10.3389/fninf.2014.00014). Frontiers in Neuroinformatics, 8.

[3] Zhou S., Li W., Cox C. and Lu H. (2020). [Side Information Dependence as a Regularizer for Analyzing Human Brain Conditions across Cognitive Experiments](https://doi.org/10.1609/aaai.v34i04.6179). Proceedings of the AAAI Conference on Artificial Intelligence, 34(04), 6957-6964.

[4] Zhou S. (2022). [Interpretable Domain-Aware Learning for Neuroimage Classification](https://etheses.whiterose.ac.uk/31044/1/PhD_thesis_ShuoZhou_170272834.pdf) (Doctoral Dissertation, University of Sheffield).

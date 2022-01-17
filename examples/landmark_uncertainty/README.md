# Uncertainty Estimation in Landmark Localisation

### 1. Description
In this example we implement the methods from [Uncertainty Estimation for Heatmap-based Landmark Localisation](placeholder_link). The method is Quantile Binning, which bins landmark predictions by any continuous uncertainty estimation measure. We assign each bin estimated localization error bounds. We can use these bins to filter out the worst predictions, or identify the likely best predictions.

We evaluate how well an uncertainty measure predicts localization error by measuring the Jaccard Index (a similarity measure) between the predicted bins and the ground truth error quantiles. We also evaluate the accuracy of the estimated error bounds. This framework is applicable to any dataset consisting of (*Continuous Uncertainty Measure*, *Continuous Evaluation Metric*) tuples.

This Figure depicts the features exemplified in this example. Note that **a)** is precomputed and provided in tabular form for this example.

![Quantile Binning Framework](figures/quantile_binning.pdf)


### 2. Datasets

We provide two datasets containing landmark localisation error and uncertainty estimation values across 6 landmarks using 3 uncertainty estimation measures. The data is derived from a Cardiac Magnetic Resonance Imaging (CMR) landmark localization task, using data from the [ASPIRE Registry](https://erj.ersjournals.com/content/39/4/945). We have 303 Short Axis View (CMR) scans with 3 landmarks each, and 422 Four Chamber View CMR scans with 3 landmarks each. For each uncertainty measure we provide tuples of (*Continuous Uncertainty Measure*, *Continuous Localization Error*) for each sample in the validation and test set in tabular form. We have split the data into 8 folds and used cross validation to gather validation and test set uncertainty tuples for every sample in the datasets. In this example, we compare the uncertainty measures: Single Maximum Heatmap Activation (S-MHA), Ensemble Maximum Heatmap Activation (E-MHA) and Ensemble Coordinate Prediction Variance (E-CPV). We compare these measures on landmark predictions from a [U-Net model](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf) and a [PHD-Net model](https://ieeexplore.ieee.org/document/9433895/).




### 3. Usage
Run Quantile Binning for the Four Chamber and Short Axis data respectively:
```
python main.py --cfg configs/4ch_data.yaml
python main.py --cfg configs/Sa_data.yaml
```

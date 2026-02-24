# Uncertainty Estimation in Landmark Localization


## 1. Description

In this example we implement the methods from [Uncertainty Estimation for Heatmap-based Landmark Localization](https://arxiv.org/abs/2203.02351) [1]. The method is Quantile Binning, which bins landmark predictions by any continuous uncertainty estimation measure. We assign each bin estimated localization error bounds. We can use these bins to filter out the worst predictions, or identify the likely best predictions.

We evaluate how well an uncertainty measure predicts localization error by measuring the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index) (a similarity measure) between the predicted bins and the ground truth error quantiles. We also evaluate the accuracy of the estimated error bounds. This framework is applicable to any dataset consisting of (*Continuous Uncertainty Measure*, *Continuous Evaluation Metric*) tuples.

Fig. 1 depicts the features exemplified in this example. Note that **a)** and **b)** are precomputed and provided in tabular form for this example.

|![Quantile Binning Framework](figures/quantile_binning.png)|
|:--:|
| Fig. 1: Overview of our general Quantile Binning framework. **a)** We make a prediction using a heatmap-based landmark localization model, and **b)**  extract a continuous uncertainty measure. **c)**  We learn thresholds to categorize predictions into bins of increasing uncertainty, estimating error bounds for each bin. **e)**  We filter out predictions from high uncertainty bins to improve the proportion of acceptable predictions. **d)**  Finally, we evaluate each uncertainty measure's ability to capture the true error quantiles and the accuracy of the estimated error bounds.|

## 2. Datasets

We provide three tabular datasets containing landmark localization error and uncertainty estimation values: 1) 303 Short Axis View Cardiac Magnetic Resonance (CMR) images with 3 landmarks (SA), 422 Four Chamber View CMR images with 3 landmarks (4ch), and 400 Cephalometric Radiology images with 19 landmarks (Cephalometric). The CMR dataset is from the [ASPIRE Registry](https://erj.ersjournals.com/content/39/4/945) [2], and the Cephalometric dataset is from an [ISBI grand challenge](https://www.researchgate.net/publication/296621456_A_benchmark_for_comparison_of_dental_radiography_analysis_algorithms) [3].

For each uncertainty measure we provide tuples of (*Continuous Uncertainty Measure*, *Continuous Localization Error*) for each sample in the validation and test set in tabular form. We have split the data into 8 folds and used cross validation to gather validation and test set uncertainty tuples for every sample in the datasets.

In this example, we compare the uncertainty measures:

- Single Maximum Heatmap Activation (S-MHA).
- Ensemble Maximum Heatmap Activation (E-MHA).
- Ensemble Coordinate Prediction Variance (E-CPV).

We compare these measures on landmark predictions from:

- A [U-Net model](https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf) [4].
- A [PHD-Net model](https://ieeexplore.ieee.org/document/9433895/) [5].

We also provide an example where the ground truth test error is not available under PHD-NET-NO-GT/.

The full README for the datasets can be found [here](https://github.com/pykale/data/tree/main/tabular/cardiac_landmark_uncertainty).

## 3. Usage

Run Quantile Binning for the Four Chamber data, Short Axis data, Cephalometric data, and Four Chamber No Ground Truth Test labels data respectively:

```bash
python main.py --cfg configs/4CH_data.yaml
python main.py --cfg configs/SA_data.yaml
python main.py --cfg configs/isbi_config.yaml
python main.py --cfg configs/no_gt_test_example.yaml
```


Edit the above yaml files for additional configuration options.

 ## 4. Additional Experimental Results
 Find additional experimental results from our paper in this repository [additional results](https://github.com/Schobs/Qbin).

## 5. Quantile Binning Beyond Landmarks
Quantile Binning can be used for purposes beyond landmarks. Simply follow the same format for your data as detailed [here](https://github.com/pykale/data/tree/main/tabular/cardiac_landmark_uncertainty). In a nutshell, each sample in the .csv should contain a column for the continuous uncertainty measure and a column for the continuous evaluation metric. Multiple uncertainty measures or evaluation metrics can be present for one sample (e.g. results from Monte Carlo Dropout and Deep Ensembles).

In your config.yaml file, specify the \<uncertainty_measure, evaluation_metric\> tuples you want to use for Quantile Binning.

If you want to compare multiple metrics against eachother, you can specify multiple tuples in a list.

Specify them in `PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS` and/or `PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS`. See [below](#6-advanced-usage---config-options) for more details on these config options.

The pipeline supports evaluating the uncertainty from multiple targets (e.g. landmarks) at once. This works by formatting your data in lists: one list for each target. See the code in main.py for how this works. In our examples, we select DATASET.LANDMARKS to represent the indicies of the targets.

To change this in your example, observe the following in main.py

```python

    for model in all_models_to_compare:
        for landmark in landmarks:
            # Define Paths for this loop
            landmark_results_path_val = os.path.join(
                cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_val + "_t" + str(landmark)
            )
            landmark_results_path_test = os.path.join(
                cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_test + "_t" + str(landmark)
            )

            fitted_save_at = os.path.join(save_folder, "fitted_quantile_binning", model, dataset)
            os.makedirs(save_folder, exist_ok=True)
            .
            .
            .
            (rest of code)

```

When you loop through the target indicies (landmarks in the example above), you can specify the paths for each target.

If you only have one target (e.g. regression of a house price), follow the config file [one_target_example.yaml](/examples/landmark_uncertainty/configs/one_target_example.yaml) for how to do this. Essentially, you just use a list of length 1.


## 6. Advanced Usage - Config Options
You can add your own config or change the config options in the `.yaml` files.

### Quick tips
- To display figures instead of saving them, set `OUTPUT.SAVE_FIGURES: False`.
- To compare multiple quantile bin settings, edit `PIPELINE.NUM_QUANTILE_BINS`.
- If test errors are not available, set `DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE: False`.
- Use `BOXPLOT` options to control detail level and plot readability.

The configuration options are grouped into:
- A) [Dataset](#a-dataset)
- B) [Pipeline](#b-pipeline)
- C) [Visualization](#c-visualization)
- D) [Boxplot](#d-boxplot)
- E) [Output](#e-output)

### A) Dataset

| Parameter | Description | Default |
| --- | --- | --- |
| `DATASET.SOURCE` | Dataset download URL. | `"https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"` |
| `DATASET.ROOT` | Root directory for data. | `"../../../data/landmarks/"` |
| `DATASET.BASE_DIR` | Dataset folder under `DATASET.ROOT`. | `"Uncertainty_tuples"` |
| `DATASET.FILE_FORMAT` | Download archive format. | `"zip"` |
| `DATASET.CONFIDENCE_INVERT` | Invert flags per uncertainty measure: `[name, invert]`. | `[["S-MHA", True], ["E-MHA", True], ["E-CPV", False]]` |
| `DATASET.DATA` | Dataset name to run (`"4CH"`, `"SA"`, `"ISBI"`, etc.). | `"4CH"` |
| `DATASET.LANDMARKS` | Landmark indices to process. | `[0, 1, 2]` |
| `DATASET.NUM_FOLDS` | Number of cross-validation folds. | `8` |
| `DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE` | Whether test error labels are available. | `True` |
| `DATASET.UE_PAIRS_VAL` | Validation CSV prefix (`<prefix>_tX.csv`). | `"uncertainty_pairs_valid"` |
| `DATASET.UE_PAIRS_TEST` | Test CSV prefix (`<prefix>_tX.csv`). | `"uncertainty_pairs_test"` |

### B) Pipeline

| Parameter | Description | Default |
| --- | --- | --- |
| `PIPELINE.NUM_QUANTILE_BINS` | Q values (number of quantile bins) to run. | `[5, 10, 25]` |
| `PIPELINE.COMPARE_INDIVIDUAL_Q` | Compare models and uncertainties at each Q. | `True` |
| `PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS` | Per-Q tuples: `[name, error_col, uncertainty_col]`. | `[["S-MHA", "S-MHA Error", "S-MHA Uncertainty"], ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"], ["E-CPV", "E-CPV Error", "E-CPV Uncertainty"]]` |
| `PIPELINE.INDIVIDUAL_Q_MODELS` | Models used in per-Q comparison. | `["U-NET", "PHD-NET"]` |
| `PIPELINE.COMPARE_Q_VALUES` | Compare fixed model/uncertainty settings across Q values. | `True` |
| `PIPELINE.COMPARE_Q_MODELS` | Models used in cross-Q comparison. | `["PHD-NET"]` |
| `PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS` | Cross-Q tuples: `[name, error_col, uncertainty_col]`. | `[["E-MHA", "E-MHA Error", "E-MHA Uncertainty"]]` |
| `PIPELINE.COMBINE_MIDDLE_BINS` | Merge middle quantile bins. | `False` |
| `PIPELINE.PIXEL_TO_MM_SCALE` | Error scaling factor (pixel-to-mm multiplier). | `1.0` |
| `PIPELINE.INDIVIDUAL_LANDMARKS_TO_SHOW` | Landmark indices for per-target plots (`[-1]` for all). | `[-1]` |
| `PIPELINE.SHOW_INDIVIDUAL_LANDMARKS` | Enable per-target plots. | `True` |

### C) Visualization

| Parameter | Description | Default |
| --- | --- | --- |
| `IM_KWARGS.colormap` | Colormap name. | `"Set1"` |
| `MARKER_KWARGS.marker` | Marker style. | `"o"` |
| `MARKER_KWARGS.markerfacecolor` | Marker face color. | `(1, 1, 1, 0.1)` |
| `MARKER_KWARGS.markeredgewidth` | Marker edge width. | `1.5` |
| `MARKER_KWARGS.markeredgecolor` | Marker edge color. | `"r"` |
| `WEIGHT_KWARGS.markersize` | Marker size. | `6` |
| `WEIGHT_KWARGS.alpha` | Marker transparency. | `0.7` |

### D) Boxplot

| Parameter | Description | Default |
| --- | --- | --- |
| `BOXPLOT.SAMPLES_AS_DOTS` | Show sample points as dots on boxplots. | `True` |
| `BOXPLOT.ERROR_LIM` | Upper y-axis limit for error plots. | `64` |
| `BOXPLOT.SHOW_SAMPLE_INFO_MODE` | Sample-count label mode (`"All"`, `"Average"`, or `None`). | `"Average"` |

### E) Output

| Parameter | Description | Default |
| --- | --- | --- |
| `OUTPUT.OUT_DIR` | Output root directory. | `"./outputs/"` |
| `OUTPUT.SAVE_PREPEND` | Prefix for output filenames. | `"example"` |
| `OUTPUT.SAVE_FIGURES` | Save figures (`True`) or show interactively (`False`). | `True` |


## 7. References
[1] L. A. Schobs, A. J. Swift and H. Lu, "Uncertainty Estimation for Heatmap-Based Landmark Localization," in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

[2] J. Hurdman, R. Condliffe, C.A. Elliot, C. Davies, C. Hill, J.M. Wild, D. Capener, P. Sephton, N. Hamilton, I.J. Armstrong, C. Billings, A. Lawrie, I. Sabroe, M. Akil, L. O’Toole, D.G. Kiely
European Respiratory Journal 2012 39: 945-955; DOI: 10.1183/09031936.00078411

[3] Wang, Ching-Wei & Huang, Cheng-Ta & Lee, Jia-Hong & Li, Chung-Hsing & Chang, Sheng-Wei & Siao, Ming-Jhih & Lai, Tat-Ming & Ibragimov, Bulat & Vrtovec, Tomaž & Ronneberger, Olaf & Fischer, Philipp & Cootes, Tim & Lindner, Claudia. (2016). A benchmark for comparison of dental radiography analysis algorithms. Medical Image Analysis. 31. 10.1016/j.media.2016.02.004.

[4] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

[5] L. Schobs, S. Zhou, M. Cogliano, A. J. Swift and H. Lu, "Confidence-Quantifying Landmark Localisation For Cardiac MRI," 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), Nice, France, 2021, pp. 985-988, doi: 10.1109/ISBI48211.2021.9433895.

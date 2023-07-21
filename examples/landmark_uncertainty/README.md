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
                cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_val + "_l" + str(landmark)
            )
            landmark_results_path_test = os.path.join(
                cfg.DATASET.ROOT, base_dir, model, dataset, uncertainty_pairs_test + "_l" + str(landmark)
            )

            fitted_save_at = os.path.join(save_folder, "fitted_quantile_binning", model, dataset)
            os.makedirs(save_folder, exist_ok=True)
            .
            .
            .
            (rest of code)

```

When you loop through the target indicies (landmarks in the example above), you can specify the paths for each target.

If you only have one target (e.g. regression of a house price), follow the config file [one_target_example.yaml](/pykale/examples/landmark_uncertainty/configs/one_target_example.yaml) for how to do this. Essentially, you just use a list of length 1.


## 6. Advanced Usage - Config Options
 You can add your own config or change the config options in the .yaml files.
 To use your own data, you can simply change the paths in DATASET, as shown in the examples.

### Quick Tips:
- To display rather than save figures set `OUTPUT.SAVE_FIGURES`: `False` in the .yaml file.
- To compare results for different numbers of Quantile Bins, check out the documentation below for `PIPELINE.NUM_QUANTILE_BINS`
- If test error is not available, set `DATASET.TEST_ERROR_AVAILABLE`: False in the .yaml file.
- Check out the `BOXPLOT` options to turn on/off visualizations of all landmarks on the box plots and adjust error limits.

The configuration options are broken into a few sections:
- A) [Dataset](#a-dataset) - options related to the dataset.
- B) [Pipeline](#b-pipeline) - options related to the pipeline.
- C) [Plotting](#c-visualisation-im_kwargs-marker_kwargs-weight_kwargs) - options related to the plots.
- D) [Boxplot](#d-boxplot) - options related to the boxplot detai.
- E) [Output](#e-output) - options related to output paths.


### A) Dataset

The following are the configuration options related to the dataset.

#### `DATASET.SOURCE`

The URL source of the dataset. This option points to the location where the dataset is hosted. If you have local data, set this to None and put your data under DATASET.ROOT.

Default:
`_C.DATASET.SOURCE` = "https://github.com/pykale/data/raw/main/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip"

#### `DATASET.ROOT`

The root directory where the dataset will be downloaded and extracted to. If you have local data, set this to the directory where your data is stored.

Default:
`_C.DATASET.ROOT` = "../../../data/landmarks/"

#### `DATASET.BASE_DIR`

The base directory within the `DATASET.ROOT` directory where the dataset will be downloaded and extracted to. If you have local data, set this to one level down of ROOT, where the data is stored.

e.g. if you are using `4CH` data (`DATASET.DATA`="4CH") and the full path to the data is "../../../data/landmarks/Uncertainty_tuples/4CH" then:

1) Set `DATASET.BASE_DIR` = "Uncertainty_tuples"
2)  Set `_C.DATASET.ROOT` = "../../../data/landmarks/".


Default:
`_C.DATASET.BASE_DIR` = "Uncertainty_tuples"

#### `DATASET.FILE_FORMAT`

The format of the dataset file. This option is used to specify the format of the dataset file when the file is downloaded and extracted.

Default:
`_C.DATASET.FILE_FORMAT` = "zip"

 #### `DATASET.CONFIDENCE_INVERT`

A list of tuples specifying the uncertainty measures and whether or not to invert their confidence values. The measures are specified by name and the inversion is specified using a boolean value.

Default:
_C.DATASET.CONFIDENCE_INVERT` = [["S-MHA", True], ["E-MHA", True], ["E-CPV", False]]

#### `DATASET.DATA`

The type of dataset to use. This option specifies which subset of the dataset to use for the experiment. For the examples, you can use 4CH, SA or ISBI.

Default:
`_C.DATASET.DATA` = "4CH"

#### `DATASET.LANDMARKS`

A list of landmark indices to use in the experiment.

Default:
`_C.DATASET.LANDMARKS` = [0, 1, 2]


#### `DATASET.NUM_FOLDS`

The number of cross-validation folds to analyze. If no cross-validaiton, set to 1.

Default:
`_C.DATASET.NUM_FOLDS = 8`


#### `DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE`

A boolean indicating whether ground truth test errors are available in the dataset. If false, it will only fit the quantile binning model to the validation set, it won't attempt to evalute the performance on the test set since there is no ground truth test error.

Default:
`_C.DATASET.GROUND_TRUTH_TEST_ERRORS_AVAILABLE = True`


#### `DATASET.UE_PAIRS_VAL`

The name of the file containing the uncertainty pairs for validation. This option specifies the name of the file containing the uncertainty pairs for validation. This should be the preamble name of the .csv files before _lX.csv where X will be the landmark index.

Default:
`_C.DATASET.UE_PAIRS_VAL = "uncertainty_pairs_valid"`

(The program will infer uncertainty_pairs_valid_l0.csv, uncertainty_pairs_valid_l1.csv, uncertainty_pairs_valid_l2.csv, etc.)

#### `DATASET.UE_PAIRS_TEST`

The name of the file containing the uncertainty pairs for testing. This option specifies the name of the file containing the uncertainty pairs for testing. This should be the preamble name of the .csv files before _lX.csv where X will be the landmark index.

Default:
`_C.DATASET.UE_PAIRS_TEST` = "uncertainty_pairs_test"

(The program will infer uncertainty_pairs_test_l0.csv, uncertainty_pairs_test_l1.csv, uncertainty_pairs_test_l2.csv, etc.)

<br/><br/>


### B) Pipeline

The following are the configuration options related to the pipeline.

#### `PIPELINE.NUM_QUANTILE_BINS`

A list of integers specifying the number of quantile bins to use for the uncertainty histogram. Use multiple if you want to compare the performance of different numbers of quantile bins, or just set it to a single integer if you want to use a single number of quantile bins e.g. [5].

Default:
`_C.PIPELINE.NUM_QUANTILE_BINS` = [5, 10, 25]

#### `PIPELINE.COMPARE_INDIVIDUAL_Q`

A boolean indicating whether to compare uncertainty measures and models over each single value of Q (Q= NUM_QUANTILE_BINS).

Default:
`_C.PIPELINE.COMPARE_INDIVIDUAL_Q` = True

#### `PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS`

A list of lists specifying the uncertainty error pairs to compare for each value of Q. Each sublist should contain three elements: the name of the uncertainty measure, the key for the error in the CSV file, and the key for the uncertainty in the CSV file.

Default:
`_C.PIPELINE.INDIVIDUAL_Q_UNCERTAINTY_ERROR_PAIRS` = [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"], ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"], ["E-CPV", "E-CPV Error", "E-CPV Uncertainty"]]

#### `PIPELINE.INDIVIDUAL_Q_MODELS`

A list of model names found in the path to compare for each value of Q.

Default:
_C.PIPELINE.INDIVIDUAL_Q_MODELS = ["U-NET", "PHD-NET"]

#### `PIPELINE.COMPARE_Q_VALUES`

A boolean indicating whether to compare a single uncertainty measure on a single model through various values of Q bins.

Default:
`_C.PIPELINE.COMPARE_Q_VALUES` = True

#### `PIPELINE.COMPARE_Q_MODELS`

A list of model names to compare over values of Q.

Default:
`_C.PIPELINE.COMPARE_Q_MODELS` = ["PHD-NET"]

#### `PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS`

A list of lists specifying the uncertainty error pairs to compare over values of Q. Each sublist should contain three elements: the name of the uncertainty measure, the key for the error in the CSV file, and the key for the uncertainty in the CSV file.

Default:
`_C.PIPELINE.COMPARE_Q_UNCERTAINTY_ERROR_PAIRS` = [["E-MHA", "E-MHA Error", "E-MHA Uncertainty"]]

#### `PIPELINE.COMBINE_MIDDLE_BINS`

A boolean indicating whether to combine the middle quantile bins into a single bin.

Default:
`_C.PIPELINE.COMBINE_MIDDLE_BINS` = False

#### `PIPELINE.PIXEL_TO_MM_SCALE`

A float specifying the scale factor to convert pixel units to millimeter units.

Default:
`_C.PIPELINE.PIXEL_TO_MM_SCALE` = 1.0

#### `PIPELINE.IND_LANDMARKS_TO_SHOW`

A list of landmark indices to show individually. A value of -1 means show all landmarks individually, and an empty list means show none.

Default:
`_C.PIPELINE.IND_LANDMARKS_TO_SHOW` = [-1]

#### `PIPELINE.SHOW_IND_LANDMARKS`

A boolean indicating whether to show results from individual landmarks.

Default:
_C.`PIPELINE.SHOW_IND_LANDMARKS` = True

<br/><br/>

### C) Visualisation: IM_KWARGS, MARKER_KWARGS, WEIGHT_KWARGS
The following are the configuration options related to visualization and plotting.

#### `IM_KWARGS.CMAP`

The color map to use for the image.

Default:
`_C.IM_KWARGS.cmap` = "gray"

#### `MARKER_KWARGS.MARKER`

The marker style to use for the landmark points.

Default:
`_C.MARKER_KWARGS.marker` = "o"

#### `MARKER_KWARGS.MARKERFACECOLOR`

The face color to use for the landmark points.

Default:
_C.MARKER_KWARGS.markerfacecolor = (1, 1, 1, 0.1)

#### `MARKER_KWARGS.MARKEREDGEWIDTH`

The edge width to use for the landmark points.

Default:
`_C.MARKER_KWARGS.markeredgewidth` = 1.5

#### `MARKER_KWARGS.MARKEREDGECOLOR`

The edge color to use for the landmark points.

Default:
`_C.MARKER_KWARGS.markeredgecolor` = "r"

#### `WEIGHT_KWARGS.MARKERSIZE`

The size to use for the weights of the landmark points.

Default:
`_C.WEIGHT_KWARGS.markersize` = 6

#### `WEIGHT_KWARGS.ALPHA`

The transparency to use for the plots.

Default:
`_C.WEIGHT_KWARGS.alpha` = 0.7

<br/><br/>


### D) BOXPLOT
The following are the configuration options related to the box plot.


#### `BOXPLOT.SAMPLES_AS_DOTS`

A boolean indicating whether to show the samples as dots on the box plot (can be expensive if many landmarks.).

Default:
`_C.BOXPLOT.SAMPLES_AS_DOTS` = True

#### `BOXPLOT.ERROR_LIM`

The error limit to use for the box plot.

Default:
`_C.BOXPLOT.ERROR_LIM` = 64

#### `BOXPLOT.SHOW_SAMPLE_INFO_MODE`

The mode for showing sample information on the box plot. The available modes are "None", "All", and "Average".

Default:
_C.BOXPLOT.SHOW_SAMPLE_INFO_MODE = "Average"

<br/><br/>


### E) OUTPUT

The following are miscellaneous configuration options.

#### `OUTPUT.SAVE_FOLDER`

The folder where the experiment results will be saved.

Default:
`_C.OUTPUT.SAVE_FOLDER` = "./outputs/""

#### `OUTPUT.SAVE_PREPEND`

The string to prepend to the output file names.

Default:
`_C.OUTPUT.SAVE_PREPEND` = "8std_27_07_22"

#### `OUTPUT.SAVE_FIGURES`

A boolean indicating whether to save the figures generated during the experiment.
If False, the figures will be shown instead.

Default:
`_C.OUTPUT.SAVE_FIGURES` = True


## 6. References
[1] L. A. Schobs, A. J. Swift and H. Lu, "Uncertainty Estimation for Heatmap-Based Landmark Localization," in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

[2] J. Hurdman, R. Condliffe, C.A. Elliot, C. Davies, C. Hill, J.M. Wild, D. Capener, P. Sephton, N. Hamilton, I.J. Armstrong, C. Billings, A. Lawrie, I. Sabroe, M. Akil, L. O’Toole, D.G. Kiely
European Respiratory Journal 2012 39: 945-955; DOI: 10.1183/09031936.00078411

[3] Wang, Ching-Wei & Huang, Cheng-Ta & Lee, Jia-Hong & Li, Chung-Hsing & Chang, Sheng-Wei & Siao, Ming-Jhih & Lai, Tat-Ming & Ibragimov, Bulat & Vrtovec, Tomaž & Ronneberger, Olaf & Fischer, Philipp & Cootes, Tim & Lindner, Claudia. (2016). A benchmark for comparison of dental radiography analysis algorithms. Medical Image Analysis. 31. 10.1016/j.media.2016.02.004.

[4] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28

[5] L. Schobs, S. Zhou, M. Cogliano, A. J. Swift and H. Lu, "Confidence-Quantifying Landmark Localisation For Cardiac MRI," 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), Nice, France, 2021, pp. 985-988, doi: 10.1109/ISBI48211.2021.9433895.

# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
#         Charles Anjah, cmanjahart@gmail.com
# =============================================================================

"""
Module from the implementation of L. A. Schobs, A. J. Swift and H. Lu,
"Uncertainty Estimation for Heatmap-Based Landmark Localization,"
in IEEE Transactions on Medical Imaging, vol. 42, no. 4, pp. 1021-1034, April 2023, doi: 10.1109/TMI.2022.3222730.

Key Evaluation Approaches:
    A) Jaccard Similarity Analysis: Measures overlap between predicted uncertainty bins
       and ground truth error quantiles (evaluate_jaccard, JaccardEvaluator).
    B) Error Bound Accuracy: Evaluates whether predicted confidence intervals contain
       actual errors (evaluate_bounds, bin_wise_bound_eval).
    C) Bin-wise Error Analysis: Analyzes mean errors and distributions within uncertainty
       bins (get_mean_errors, bin_wise_errors).

Main Classes:
    - BaseEvaluator: Abstract base class defining the evaluation workflow
    - JaccardEvaluator: Concrete evaluator for Jaccard similarity metrics
    - EvaluationConfig: Configuration container for evaluation parameters
    - ResultsContainer: Organizes complex nested evaluation results
    - DataProcessor: Utilities for data extraction and preprocessing
    - QuantileCalculator: Quantile-based error distribution analysis
    - MetricsCalculator: Statistical metrics computation
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch import tensor
from torchmetrics.classification import BinaryJaccardIndex

from kale.prepdata.string_transform import strip_for_bound


class ColumnNames:
    """Constants for DataFrame column names."""

    UID = "uid"
    TARGET_IDX = "Target Index"
    TESTING_FOLD = "Testing Fold"
    ERROR_SUFFIX = " Error"
    UNCERTAINTY_BINS_SUFFIX = " Uncertainty bins"
    UNCERTAINTY_BOUNDS_SUFFIX = " Uncertainty bounds"


class ResultKeys:
    """Constants for result dictionary keys."""

    MEAN_ALL_TARGETS = "mean all targets"
    MEAN_ALL_BINS = "mean all bins"
    ALL_BINS = "all bins"
    ALL_BINS_CONCAT_TARGETS_SEP = "all bins concatenated targets separated"

    # Bounds specific
    ERROR_BOUNDS_ALL = "error_bounds_all"
    ALL_BOUND_PERCENTS_NO_TARGET_SEP = "all_bound_percents_notargetsep"
    ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all errorbound concat bins targets sep foldwise"
    ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_ALL = "all_errorbound_concat_bins_targets_sep_all"

    # Errors specific
    ALL_MEAN_ERROR_BINS_NO_SEP = "all_mean_error_bins_nosep"
    ALL_MEAN_ERROR_BINS_TARGETS_SEP = "all mean error bins targets sep"
    ALL_ERROR_CONCAT_BINS_TARGET_NO_SEP = "all_error_concat_bins_targets_nosep"
    ALL_ERROR_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all error concat bins targets sep foldwise"
    ALL_ERROR_CONCAT_BINS_TARGET_SEP_ALL = "all_error_concat_bins_targets_sep_all"

    # Jaccard specific
    JACCARD_ALL = "jaccard_all"
    JACCARD_TARGETS_SEPARATED = "Jaccard targets separated"
    RECALL_ALL = "recall_all"
    RECALL_TARGETS_SEPARATED = "Recall targets separated"
    PRECISION_ALL = "precision_all"
    PRECISION_TARGETS_SEPARATED = "Precision targets separated"
    ALL_JACC_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all jacc concat bins targets sep foldwise"
    ALL_JACC_CONCAT_BINS_TARGET_SEP_ALL = "all_jaccard_concat_bins_targets_sep_all"


@dataclass
class EvaluationConfig:
    """
    Configuration parameters for uncertainty quantification evaluation.

    Attributes:
        num_folds (int): Number of cross-validation folds for evaluation. Defaults to 8.
        original_num_bins (int): Number of quantile bins for uncertainty evaluation. Defaults to 10. Controls the
            granularity of uncertainty analysis.
        error_scaling_factor (float): Scaling factor applied to prediction errors during evaluation. Defaults to 1.0 (no
            scaling).
        combine_middle_bins (bool): Whether to combine middle uncertainty bins for simplified analysis. Defaults to
            False. When True, reduces evaluation complexity by merging intermediate quantiles.
        combined_num_bins (int): Number of bins when middle bins are combined. Defaults to 3 (low, medium, high
            uncertainty). Only used when combine_middle_bins is True.

    Example:

        .. code-block:: pycon

            >>> config = EvaluationConfig(
            ...     num_folds=5,
            ...     original_num_bins=20,
            ...     combine_middle_bins=True
            ... )
            >>> evaluator = JaccardEvaluator(config)
    """

    num_folds: int = 8
    original_num_bins: int = 10
    error_scaling_factor: float = 1.0
    combine_middle_bins: bool = False
    combined_num_bins: int = 3


@dataclass
class FoldData:
    """
    Container for evaluation data from a single cross-validation fold.

    Attributes:
        errors (pd.DataFrame): DataFrame containing prediction errors for the fold. Expected columns include UID,
            target_idx, and model-specific error columns.
        bins (pd.DataFrame): DataFrame containing uncertainty bin assignments. Expected columns include UID, target_idx,
            and uncertainty bin columns.
        bounds (Optional[List]): Optional list containing error bound information for bound-based evaluation methods.
            Defaults to None.

    Example:

        .. code-block:: pycon

            >>> fold_data = FoldData(
            ...     errors=error_df[error_df['Testing Fold'] == fold_idx],
            ...     bins=bins_df[bins_df['Testing Fold'] == fold_idx],
            ...     bounds=optional_bounds_list
            ... )
    """

    errors: pd.DataFrame
    bins: pd.DataFrame
    bounds: Optional[List] = None


@dataclass
class BinResults:
    """
    Base container for evaluation results from a single fold.

    Attributes:
        mean_all_targets (float): Mean evaluation metric across all targets in the fold.
        mean_all_bins (List[float]): Mean evaluation metric for each bin across targets. Length equals the number of
            bins.
        all_bins (List[List[float]]): Raw evaluation metrics for each bin and target. Outer list represents bins, inner
            lists contain values for each target.
        all_bins_concat_targets_sep (List[List[List[Any]]]): Values grouped by target and bin within one fold.
            Bounds and Jaccard use float values, so each target/bin entry is [metric]. Errors use List[float] values,
            so each entry is [[sample_errors...]], or [] for an empty bin. Bounds and errors store bins from lowest
            to highest uncertainty; Jaccard stores bins in the reverse order.
    """

    mean_all_targets: float
    mean_all_bins: List[float]
    all_bins: List[List[float]]
    all_bins_concat_targets_sep: List[List[List[Any]]]


@dataclass
class JaccardBinResults(BinResults):
    """
    Extended results container for Jaccard similarity evaluation with precision and recall.

    Attributes:
        mean_all_targets_recall (float): Mean recall across all targets in the fold.
        mean_all_bins_recall (List[float]): Mean recall for each bin across targets.
        all_bins_recall (List[List[float]]): Raw recall values for each bin and target.
        mean_all_targets_precision (float): Mean precision across all targets.
        mean_all_bins_precision (List[float]): Mean precision for each bin.
        all_bins_precision (List[List[float]]): Raw precision values for each bin and target.
    """

    mean_all_targets_recall: float = 0.0
    mean_all_bins_recall: List[float] = field(default_factory=list)
    all_bins_recall: List[List[float]] = field(default_factory=list)
    mean_all_targets_precision: float = 0.0
    mean_all_bins_precision: List[float] = field(default_factory=list)
    all_bins_precision: List[List[float]] = field(default_factory=list)


class ResultsContainer:
    """
    Store evaluation results by model, uncertainty type, bin, target, and fold.

    Attributes:
        num_bins (int): Number of uncertainty bins in the evaluation.
        num_targets (int): Number of target labels being evaluated.
        main_results (Dict): Main aggregated results across all targets.
        target_separated_results (Dict): Results separated by individual targets.
        target_sep_foldwise (List[Dict]): Target-separated results for each fold.
        target_sep_all (List[Dict]): Target-separated results aggregated across folds.
        additional_containers (Dict): Container for evaluation-specific results.
        recall_results (Dict): Recall metrics for all model-uncertainty combinations.
        recall_target_separated (Dict): Target-separated recall results.
        precision_results (Dict): Precision metrics for all model-uncertainty combinations.
        precision_target_separated (Dict): Target-separated precision results.

    Example:

        .. code-block:: pycon

            >>> container = ResultsContainer(num_bins=10, num_targets=5)
            >>> container.add_main_result("model1_epistemic", fold_results)
            >>> container.add_target_separated_result("model1_epistemic", target_results)
    """

    def __init__(self, num_bins: int, num_targets: int):
        """
        Initialize ResultsContainer with bin and target dimensions.

        Args:
            num_bins (int): Number of uncertainty bins in the evaluation.
            num_targets (int): Number of target labels being evaluated.
        """
        self.num_bins = num_bins
        self.num_targets = num_targets
        self._init_containers()

    def _init_containers(self):
        """Initialize empty result containers."""
        # Main results
        self.main_results = {}
        self.target_separated_results = {}

        # Target separated containers
        self.target_sep_foldwise = [{} for _ in range(self.num_targets)]
        self.target_sep_all = [{} for _ in range(self.num_targets)]

        # Additional containers for specific evaluations
        self.additional_containers = {}

        # Jaccard metrics
        self.recall_results = {}
        self.recall_target_separated = {}
        self.precision_results = {}
        self.precision_target_separated = {}

    def add_main_result(self, key: str, value):
        """Add a main result."""
        self.main_results[key] = value

    def add_target_separated_result(self, key: str, value):
        """Add a target separated result."""
        self.target_separated_results[key] = value


class DataProcessor:
    """Extract fold data and group predictions by uncertainty bin."""

    @staticmethod
    def extract_fold_data(data_structs: pd.DataFrame, fold: int, uncertainty_type: str) -> FoldData:
        """
        Extract data for a specific cross-validation fold and uncertainty type.

        Filters the input DataFrame to extract only the data belonging to the specified fold and prepares separate
        DataFrames for errors and uncertainty bins.

        Args:
            data_structs (pd.DataFrame): Complete dataset containing all folds and data. Must include columns for fold
                identification, UIDs, target indices, errors, and uncertainty bins.
            fold (int): The specific fold number to extract (0-based indexing).
            uncertainty_type (str): Type of uncertainty to extract (e.g., "epistemic", "aleatoric"). Used to construct
                column names for error and bin data.

        Returns:
            FoldData: Container with filtered errors and bins DataFrames for the specified fold and uncertainty type.
                Contains UIDs, target indices, and corresponding error and bin values.
        """
        fold_mask = data_structs[ColumnNames.TESTING_FOLD] == fold

        errors = data_structs[fold_mask][
            [ColumnNames.UID, ColumnNames.TARGET_IDX, uncertainty_type + ColumnNames.ERROR_SUFFIX]
        ]

        bins = data_structs[fold_mask][
            [ColumnNames.UID, ColumnNames.TARGET_IDX, uncertainty_type + ColumnNames.UNCERTAINTY_BINS_SUFFIX]
        ]

        return FoldData(errors=errors, bins=bins)

    @staticmethod
    def group_data_by_bins(errors_dict: Dict, bins_dict: Dict, num_bins: int) -> Tuple[List[List], List[List]]:
        """
        Group prediction data by their assigned uncertainty bins.

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values. Keys should correspond to unique
                prediction identifiers.
            bins_dict (Dict): Dictionary mapping prediction keys to bin assignments. Values should be bin indices (0 to
                num_bins-1).
            num_bins (int): Total number of uncertainty bins used in the evaluation.

        Returns:
            Tuple[List[List], List[List]]: A tuple containing:
                - bin_keys: List of lists where bin_keys[i] contains all prediction
                  keys assigned to bin i.
                - bin_errors: List of lists where bin_errors[i] contains all error
                  values for predictions assigned to bin i.

        Example:

            .. code-block:: pycon

                >>> errors = {'pred1': 0.1, 'pred2': 0.3, 'pred3': 0.2}
                >>> bins = {'pred1': 0, 'pred2': 1, 'pred3': 0}
                >>> keys, errs = DataProcessor.group_data_by_bins(errors, bins, 2)
                >>> # keys[0] = ['pred1', 'pred3'], keys[1] = ['pred2']
                >>> # errs[0] = [0.1, 0.2], errs[1] = [0.3]
        """
        bin_keys: List[List] = [[] for _ in range(num_bins)]
        bin_errors: List[List] = [[] for _ in range(num_bins)]

        # Compare bins and uids as strings so a uid stored as an int in one frame and a str in the other still lines up.
        index_by_bin_label = {str(bin_idx): bin_idx for bin_idx in range(num_bins)}
        errors_by_str_uid = {}
        for key, value in errors_dict.items():
            errors_by_str_uid.setdefault(str(key), value)

        for key, bin_value in bins_dict.items():
            bin_idx = index_by_bin_label.get(str(bin_value))
            if bin_idx is None:  # Bin assignment outside the requested range.
                continue

            bin_keys[bin_idx].append(key)
            str_uid = str(key)
            if str_uid not in errors_by_str_uid:
                raise ValueError(f"No error found for uid {key!r}")
            bin_errors[bin_idx].append(errors_by_str_uid[str_uid])

        return bin_keys, bin_errors


class QuantileCalculator:
    """Compute quantile thresholds and group errors into quantile-based bins."""

    @staticmethod
    def calculate_error_quantiles(
        errors_dict: Dict, num_bins: int, combine_middle_bins: bool
    ) -> Tuple[List[float], List[List], List[List]]:
        """
        Calculate quantile thresholds and group errors into quantile-based bins.

        Computes quantile boundaries based on the error distribution and groups errors and their corresponding keys into
        bins. Supports combining middle bins for simplified three-bin analysis (low, medium, high error).

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values. Used to compute quantile boundaries
                from the error distribution.
            num_bins (int): Number of quantile bins to create. Determines the granularity of the quantile analysis.
            combine_middle_bins (bool): Whether to combine middle quantiles into a single bin. When True, creates 3 bins
                regardless of num_bins.

        Returns:
            Tuple[List[float], List[List], List[List]]: A tuple containing:
                - quantile_thresholds: List of quantile boundary values
                - error_groups: List of error value lists for each bin (worst to best)
                - key_groups: List of prediction key lists for each bin (worst to best)

        Example:

            .. code-block:: pycon

                >>> errors = {'p1': 0.1, 'p2': 0.5, 'p3': 0.3, 'p4': 0.7, 'p5': 0.2}
                >>> thresholds, err_grps, key_grps = QuantileCalculator.calculate_error_quantiles(
                ...     errors, num_bins=3, combine_middle_bins=False
                ... )
                >>> # Returns thresholds and groups ordered from highest to lowest error
        """
        sorted_errors = sorted(errors_dict.values())

        quantiles = np.arange(1 / num_bins, 1, 1 / num_bins)[: num_bins - 1]
        quantile_thresholds = [np.quantile(sorted_errors, q) for q in quantiles]

        if combine_middle_bins:
            quantile_thresholds = [quantile_thresholds[0], quantile_thresholds[-1]]

        error_groups, key_groups = QuantileCalculator._group_by_quantiles(errors_dict, quantile_thresholds)

        # Reverse to go from worst to best (B5 to B1)
        return quantile_thresholds, error_groups[::-1], key_groups[::-1]

    @staticmethod
    def _group_by_quantiles(errors_dict: Dict, thresholds: List[float]) -> Tuple[List[List], List[List]]:
        """
        Group errors and keys by quantile thresholds.

        Helper method that partitions prediction errors and their corresponding keys into groups based on quantile
        threshold boundaries.

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values.
            thresholds (List[float]): List of quantile threshold values that define the boundaries between groups.

        Returns:
            Tuple[List[List], List[List]]: A tuple containing:
                - error_groups: List of error value lists for each quantile group
                - key_groups: List of prediction key lists for each quantile group
        """
        error_groups = []
        key_groups = []

        for q in range(len(thresholds) + 1):
            group_errors = []
            group_keys = []

            for id_, error in errors_dict.items():
                if QuantileCalculator._is_in_quantile_range(error, q, thresholds):
                    group_errors.append(error)
                    group_keys.append(id_)

            error_groups.append(group_errors)
            key_groups.append(group_keys)

        return error_groups, key_groups

    @staticmethod
    def _is_in_quantile_range(error: float, quantile_idx: int, thresholds: List[float]) -> bool:
        """
        Check if error falls within the specified quantile range.

        Helper method that determines whether a given error value belongs to the specified quantile group based on
        threshold boundaries.

        Args:
            error (float): The error value to classify.
            quantile_idx (int): Index of the quantile group to check (0-based).
            thresholds (List[float]): List of quantile threshold values.

        Returns:
            bool: True if the error falls within the specified quantile range, False otherwise.
        """
        if quantile_idx == 0:
            return error <= thresholds[0]
        elif quantile_idx < len(thresholds):
            return thresholds[quantile_idx - 1] < error <= thresholds[quantile_idx]
        else:
            return error > thresholds[quantile_idx - 1]


class MetricsCalculator:
    """Compute Jaccard similarity, recall, precision, and bound accuracy for uncertainty evaluation."""

    @staticmethod
    def calculate_jaccard_metrics(predicted_keys: List, ground_truth_keys: List) -> Tuple[float, float, float]:
        """
        Calculate Jaccard similarity, recall, and precision for set-based evaluation.

        Computes three key metrics for assessing the overlap between predicted and ground truth sets of prediction keys.
        These metrics provide comprehensive evaluation of uncertainty quantification quality within bins.

        Args:
            predicted_keys (List): List of prediction keys identified by the model as belonging to a specific
                uncertainty or error bin.
            ground_truth_keys (List): List of prediction keys that actually belong to the target bin based on true error
                characteristics.

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - jaccard (float): Jaccard similarity coefficient (intersection over union)
                - recall (float): Recall score (true positives / actual positives)
                - precision (float): Precision score (true positives / predicted positives)

        Metrics Explanation:
            - Jaccard: |intersection| / |union|, measures overall set overlap
            - Recall: How many actual positives were correctly identified
            - Precision: How many predictions were actually correct

        Edge Cases:
            - Empty ground truth: recall = 1.0, precision = 0.0
            - Empty predictions: precision = 0.0
            - Both empty: jaccard = 0.0 (handled by zero_division parameter)

        Example:

            .. code-block:: pycon

                >>> pred = ['sample1', 'sample2', 'sample3']
                >>> gt = ['sample1', 'sample4']
                >>> jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics(pred, gt)
                >>> # jaccard = 1/4 = 0.25, recall = 1/2 = 0.5, precision = 1/3 ≈ 0.33
        """
        all_keys = list(set(predicted_keys + ground_truth_keys))

        jaccard = 0.0
        if len(all_keys) != 0:
            pred_binary = tensor([1 if key in predicted_keys else 0 for key in all_keys])
            gt_binary = tensor([1 if key in ground_truth_keys else 0 for key in all_keys])
            jaccard_metric = BinaryJaccardIndex()
            jaccard = float(jaccard_metric(pred_binary, gt_binary))

        if len(ground_truth_keys) == 0:
            recall = 1.0
            precision = 0.0
        else:
            recall = sum(1 for el in predicted_keys if el in ground_truth_keys) / len(ground_truth_keys)

            if len(predicted_keys) == 0:
                precision = 0.0
            else:
                precision = sum(1 for x in predicted_keys if x in ground_truth_keys) / len(predicted_keys)

        return jaccard, recall, precision

    @staticmethod
    def calculate_bound_accuracy(error: float, bin_idx: int, bounds: List[float]) -> bool:
        """
        Check whether an error falls within its bin's half-open range ``(lower, upper]``.

        Bin 0 covers ``(0, bounds[0]]``, intermediate bin ``i`` covers ``(bounds[i-1], bounds[i]]``, and the last bin
        covers ``(bounds[-1], inf)``.

        Args:
            error (float): The prediction error value to check.
            bin_idx (int): Index of the uncertainty bin (0-based).
            bounds (List[float]): Upper bound of each finite bin, ordered from tightest to loosest.

        Returns:
            bool: True if the error falls within the bin's range.
        """
        if bin_idx == 0:
            return 0 < error <= bounds[bin_idx]
        elif bin_idx < len(bounds):
            return bounds[bin_idx - 1] < error <= bounds[bin_idx]
        else:
            return error > bounds[bin_idx - 1]


class BaseEvaluator(ABC):
    """
    Abstract base class for uncertainty quantification evaluation strategies.

    Evaluate models and uncertainty types across folds, collecting the metrics produced by each subclass.

    Attributes:
        config_ (EvaluationConfig): Configuration containing evaluation parameters
        current_num_bins_ (int): Number of bins for current evaluation (may differ from original)
        current_targets_ (List[int]): Target indices for current evaluation
        current_uncertainty_type_ (str): Current uncertainty type being processed
        current_model_ (str): Current model being evaluated
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize BaseEvaluator with evaluation configuration.

        Args:
            config (EvaluationConfig): Configuration object containing evaluation parameters such as number of folds,
                bins, and processing options.
        """
        self.config_ = config
        self.container_: Optional[ResultsContainer] = None
        self.current_num_bins_: int = config.original_num_bins
        self.current_targets_: List[int] = []
        self.current_uncertainty_type_: str = ""
        self.current_model_: str = ""

    def evaluate(self, bin_predictions: Dict[str, pd.DataFrame], uncertainty_pairs: List, targets: List[int]) -> Dict:
        """
        Evaluate each model and uncertainty type across the configured folds.

        Args:
            bin_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to DataFrames containing bin
                predictions and evaluation data. Each DataFrame should contain columns for UIDs, target indices, errors,
                and uncertainty bins.
            uncertainty_pairs (List): List of uncertainty type pairs to evaluate. Each pair contains uncertainty type
                names (e.g., ['epistemic'], ['aleatoric']).
            targets (List[int]): List of target indices to include in the evaluation. Used to filter data and organize
                results by target.

        Returns:
            Dict: Aggregated metrics and target-separated results in the concrete evaluator's format.
        """
        self.current_targets_ = targets
        self.current_num_bins_ = (
            self.config_.combined_num_bins if self.config_.combine_middle_bins else self.config_.original_num_bins
        )

        self.container_ = ResultsContainer(self.current_num_bins_, len(targets))

        for model, data_structs in bin_predictions.items():
            self.current_model_ = model
            for uncertainty_pair in uncertainty_pairs:
                self.current_uncertainty_type_ = uncertainty_pair[0]
                model_key = f"{model} {self.current_uncertainty_type_}"

                fold_results = self._process_all_folds(data_structs)

                self._aggregate_fold_results(model_key, fold_results)

        return self._finalize_results()

    @abstractmethod
    def _process_single_fold(self, fold_data: FoldData) -> BinResults:
        """
        Process evaluation for a single cross-validation fold.

        Args:
            fold_data (FoldData): Container with errors and bins data for one fold. Contains filtered DataFrames for the
                current fold, uncertainty type, and any additional data needed for evaluation.

        Returns:
            BinResults: Results structure containing evaluation metrics for this fold.
        """
        pass

    @abstractmethod
    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]) -> None:
        """
        Aggregate results across all folds for a model-uncertainty combination.

        Args:
            model_key (str): Identifier for the current model-uncertainty combination (format: "model_name
                uncertainty_type").
            fold_results (List[BinResults]): List of evaluation results from all folds for the current model-uncertainty
                combination.
        """
        pass

    @abstractmethod
    def _finalize_results(self) -> Dict:
        """
        Convert results container into final output format.

        Returns:
            Dict: Final results dictionary with keys matching the expected API format. Structure depends on the
                evaluation type but typically includes main results, target-separated results, and evaluation-specific
                metrics.
        """
        pass

    def _process_all_folds(self, data_structs: pd.DataFrame) -> List[BinResults]:
        """
        Process all cross-validation folds for a given model and uncertainty type.

        Args:
            data_structs (pd.DataFrame): Complete dataset containing all folds and evaluation data for the current
                model. Must include columns for fold identification, UIDs, target indices, errors, and uncertainty bins.

        Returns:
            List[BinResults]: Results in fold order.
        """
        fold_results = []

        for fold in range(self.config_.num_folds):
            fold_data = DataProcessor.extract_fold_data(data_structs, fold, self.current_uncertainty_type_)
            result = self._process_single_fold(fold_data)
            fold_results.append(result)

        return fold_results


class JaccardEvaluator(BaseEvaluator):
    """
    Evaluator for calculating Jaccard similarity metrics for uncertainty quantification.

    Compare predicted uncertainty bins with quantiles of the observed errors using Jaccard similarity, recall,
    and precision.

    The Jaccard similarity is calculated as: J(A, B) = |A ∩ B| / |A ∪ B|

    A and B contain the sample UIDs in the corresponding uncertainty and error bins.

    Example:

        .. code-block:: pycon

            >>> config = EvaluationConfig(original_num_bins=10, num_folds=5)
            >>> evaluator = JaccardEvaluator(config)
            >>> results = evaluator.evaluate(
            ...     bin_predictions={"model1": df},
            ...     uncertainty_pairs=[("epistemic", "error")],
            ...     targets=[0, 1],
            ... )
            >>> print(results["jaccard_all"]["model1 epistemic"])
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config or EvaluationConfig())

    @classmethod
    def create_simple(
        cls, original_num_bins: int, num_folds: int = 8, combine_middle_bins: bool = False
    ) -> "JaccardEvaluator":
        """
        Create a JaccardEvaluator instance with simplified parameters.

        Convenience factory method for creating a JaccardEvaluator with commonly used configuration parameters without
        requiring full EvaluationConfig setup.

        Args:
            original_num_bins (int): Number of quantile bins for uncertainty evaluation. Typical values range from 5 to
                20 depending on dataset size and desired granularity.
            num_folds (int, optional): Number of cross-validation folds for evaluation. Defaults to 8. Higher values
                provide more robust estimates but increase computation time.
            combine_middle_bins (bool, optional): Whether to combine middle uncertainty bins for simplified analysis.
                Defaults to False. When True, reduces the number of bins by merging middle quantiles.

        Returns:
            JaccardEvaluator: Configured evaluator instance ready for uncertainty quantification assessment.

        Example:

            .. code-block:: pycon

                >>> evaluator = JaccardEvaluator.create_simple(
                ...     original_num_bins=10,
                ...     num_folds=5,
                ...     combine_middle_bins=True
                ... )
                >>> results = evaluator.evaluate(
                ...     bin_predictions={"model1": predictions_df},
                ...     uncertainty_pairs=[("epistemic", "error")],
                ...     targets=[0, 1],
                ... )
        """
        config = EvaluationConfig(
            original_num_bins=original_num_bins, num_folds=num_folds, combine_middle_bins=combine_middle_bins
        )
        return cls(config)

    @classmethod
    def create_default(cls) -> "JaccardEvaluator":
        """
        Create a JaccardEvaluator instance with default configuration parameters.

        Factory method that creates an evaluator using the default EvaluationConfig settings, providing a quick way to
        instantiate the evaluator for standard uncertainty quantification evaluation tasks.

        Returns:
            JaccardEvaluator: Evaluator instance configured with default parameters including standard bin counts, fold
                numbers, and evaluation settings.

        Example:

            .. code-block:: pycon

                >>> evaluator = JaccardEvaluator.create_default()
                >>> results = evaluator.evaluate(
                ...     bin_predictions={"model1": data},
                ...     uncertainty_pairs=[("uncertainty", "error")],
                ...     targets=[0],
                ... )
        """
        return cls()

    @staticmethod
    def _format_target_results(bin_jaccard: List[float], bin_recall: List[float], bin_precision: List[float]) -> Dict:
        """
        Format target evaluation results into the expected dictionary format.

        Args:
            bin_jaccard (List[float]): Jaccard values for each bin.
            bin_recall (List[float]): Recall values for each bin.
            bin_precision (List[float]): Precision values for each bin.

        Returns:
            Dict: Formatted results dictionary with mean and bin-wise metrics.
        """
        return {
            "mean_jaccard": np.mean(bin_jaccard),
            "mean_recall": np.mean(bin_recall),
            "mean_precision": np.mean(bin_precision),
            "bin_jaccard": bin_jaccard,
            "bin_recall": bin_recall,
            "bin_precision": bin_precision,
        }

    def _process_single_fold(self, fold_data: FoldData) -> JaccardBinResults:
        """
        Process Jaccard evaluation metrics for a single cross-validation fold.

        Args:
            fold_data (FoldData): Container with errors and bins data for the current fold. Contains filtered DataFrames
                with prediction errors and uncertainty bins for evaluation.

        Returns:
            JaccardBinResults: Mean and per-target Jaccard, recall, and precision metrics in descending bin order.
        """
        all_target_jaccard = []
        all_target_recall = []
        all_target_precision = []

        all_bin_jaccard: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        all_bin_recall: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        all_bin_precision: List[List[float]] = [[] for _ in range(self.current_num_bins_)]

        all_bins_concat_targets_sep: List[List[List[float]]] = [
            [[] for _ in range(self.current_num_bins_)] for _ in range(len(self.current_targets_))
        ]

        for i, target_idx in enumerate(self.current_targets_):
            target_results = self._process_target_jaccard(fold_data, target_idx)

            all_target_jaccard.append(target_results["mean_jaccard"])
            all_target_recall.append(target_results["mean_recall"])
            all_target_precision.append(target_results["mean_precision"])

            for bin_idx in range(self.current_num_bins_):
                all_bin_jaccard[bin_idx].append(target_results["bin_jaccard"][bin_idx])
                all_bin_recall[bin_idx].append(target_results["bin_recall"][bin_idx])
                all_bin_precision[bin_idx].append(target_results["bin_precision"][bin_idx])
                all_bins_concat_targets_sep[i][bin_idx].append(target_results["bin_jaccard"][bin_idx])

        return JaccardBinResults(
            mean_all_targets=np.mean(all_target_jaccard),
            mean_all_bins=[np.mean(x) for x in all_bin_jaccard],
            all_bins=all_bin_jaccard,
            all_bins_concat_targets_sep=all_bins_concat_targets_sep,
            mean_all_targets_recall=np.mean(all_target_recall),
            mean_all_bins_recall=[np.mean(x) for x in all_bin_recall],
            all_bins_recall=all_bin_recall,
            mean_all_targets_precision=np.mean(all_target_precision),
            mean_all_bins_precision=[np.mean(x) for x in all_bin_precision],
            all_bins_precision=all_bin_precision,
        )

    def _process_target_jaccard(self, fold_data: FoldData, target_idx: int) -> Dict:
        """
        Process Jaccard similarity metrics for a specific target within a fold.

        Computes bin-wise Jaccard similarity, precision, and recall for predictions associated with a particular target
        index. This enables target-specific evaluation of uncertainty quantification quality.

        Args:
            fold_data (FoldData): Container with errors and bins data for the current fold.
            target_idx (int): Index of the target to process. Used to filter data for target-specific evaluation.

        Returns:
            Dict: Dictionary containing computed metrics with keys:
                - 'mean_jaccard': Overall Jaccard similarity for this target
                - 'mean_recall': Overall recall for this target
                - 'mean_precision': Overall precision for this target
                - 'bin_jaccard': List of Jaccard values for each bin
                - 'bin_recall': List of recall values for each bin
                - 'bin_precision': List of precision values for each bin
        """
        errors_dict, bins_dict = self._extract_target_data(fold_data, target_idx)

        pred_bin_keys = self._get_predicted_bin_keys(errors_dict, bins_dict)
        gt_key_groups = self._get_ground_truth_quantiles(errors_dict)

        bin_jaccard, bin_recall, bin_precision = self._calculate_bin_wise_metrics(pred_bin_keys, gt_key_groups)

        return JaccardEvaluator._format_target_results(bin_jaccard, bin_recall, bin_precision)

    def _extract_target_data(self, fold_data: FoldData, target_idx: int) -> Tuple[Dict, Dict]:
        """
        Extract and prepare target-specific data for evaluation.

        Args:
            fold_data (FoldData): Container with errors and bins data for the current fold.
            target_idx (int): Index of the target to process.

        Returns:
            Tuple[Dict, Dict]: Tuple containing errors_dict and bins_dict for the target.
        """
        target_errors = fold_data.errors[fold_data.errors[ColumnNames.TARGET_IDX] == target_idx]
        target_bins = fold_data.bins[fold_data.bins[ColumnNames.TARGET_IDX] == target_idx]

        errors_dict = dict(
            zip(
                target_errors[ColumnNames.UID], target_errors[self.current_uncertainty_type_ + ColumnNames.ERROR_SUFFIX]
            )
        )
        bins_dict = dict(
            zip(
                target_bins[ColumnNames.UID],
                target_bins[self.current_uncertainty_type_ + ColumnNames.UNCERTAINTY_BINS_SUFFIX],
            )
        )

        return errors_dict, bins_dict

    def _get_predicted_bin_keys(self, errors_dict: Dict, bins_dict: Dict) -> List[List]:
        """
        Get predicted bin keys ordered from worst to best (B5 to B1).

        Args:
            errors_dict (Dict): Dictionary mapping UIDs to error values.
            bins_dict (Dict): Dictionary mapping UIDs to bin assignments.

        Returns:
            List[List]: Predicted bin keys for each bin, ordered from worst to best.
        """
        pred_bin_keys, _ = DataProcessor.group_data_by_bins(errors_dict, bins_dict, self.current_num_bins_)
        return pred_bin_keys[::-1]  # Reverse for B5 to B1

    def _get_ground_truth_quantiles(self, errors_dict: Dict) -> List[List]:
        """
        Get ground truth quantile groups for error-based evaluation.

        Args:
            errors_dict (Dict): Dictionary mapping UIDs to error values.

        Returns:
            List[List]: Ground truth key groups for each quantile bin.
        """
        _, _, gt_key_groups = QuantileCalculator.calculate_error_quantiles(
            errors_dict, self.config_.original_num_bins, self.config_.combine_middle_bins
        )
        return gt_key_groups

    def _calculate_bin_wise_metrics(
        self, pred_bin_keys: List[List], gt_key_groups: List[List]
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate Jaccard, recall, and precision metrics for each bin.

        Args:
            pred_bin_keys (List[List]): Predicted bin keys for each bin.
            gt_key_groups (List[List]): Ground truth key groups for each bin.

        Returns:
            Tuple[List[float], List[float], List[float]]: Bin-wise jaccard, recall, and precision values.
        """
        bin_jaccard = []
        bin_recall = []
        bin_precision = []

        for bin_idx in range(self.current_num_bins_):
            jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics(
                pred_bin_keys[bin_idx], gt_key_groups[bin_idx]
            )
            bin_jaccard.append(jaccard)
            bin_recall.append(recall)
            bin_precision.append(precision)

        return bin_jaccard, bin_recall, bin_precision

    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]) -> None:
        """
        Aggregate Jaccard evaluation results across all cross-validation folds.

        Args:
            model_key (str): Identifier for the current model-uncertainty combination (format: "model_name
                uncertainty_type").
            fold_results (List[BinResults]): List of JaccardBinResults from all folds for the current model-uncertainty
                combination.
        """
        jaccard_results = cast(List[JaccardBinResults], fold_results)

        aggregated_metrics = self._aggregate_fold_metrics(jaccard_results)

        self._store_main_results(model_key, aggregated_metrics)

        self._store_target_separated_results(model_key, jaccard_results)

    def _aggregate_fold_metrics(self, jaccard_results: List[JaccardBinResults]) -> Dict[str, List[List[float]]]:
        """
        Aggregate metrics across all folds for bin-wise and target-wise analysis.

        Args:
            jaccard_results (List[JaccardBinResults]): Results from all folds.

        Returns:
            Dict[str, List[List[float]]]: Dictionary containing aggregated metrics for each metric type.
        """
        empty_bins_template: List[List[float]] = [[] for _ in range(self.current_num_bins_)]

        fold_jaccard_bins = copy.deepcopy(empty_bins_template)
        fold_recall_bins = copy.deepcopy(empty_bins_template)
        fold_precision_bins = copy.deepcopy(empty_bins_template)

        fold_all_jaccard_bins = copy.deepcopy(empty_bins_template)
        fold_all_recall_bins = copy.deepcopy(empty_bins_template)
        fold_all_precision_bins = copy.deepcopy(empty_bins_template)

        for result in jaccard_results:
            for bin_idx in range(len(result.mean_all_bins)):
                fold_jaccard_bins[bin_idx].append(result.mean_all_bins[bin_idx])
                fold_recall_bins[bin_idx].append(result.mean_all_bins_recall[bin_idx])
                fold_precision_bins[bin_idx].append(result.mean_all_bins_precision[bin_idx])

                fold_all_jaccard_bins[bin_idx].extend(result.all_bins[bin_idx])
                fold_all_recall_bins[bin_idx].extend(result.all_bins_recall[bin_idx])
                fold_all_precision_bins[bin_idx].extend(result.all_bins_precision[bin_idx])

        return {
            "fold_jaccard_bins": fold_jaccard_bins,
            "fold_recall_bins": fold_recall_bins,
            "fold_precision_bins": fold_precision_bins,
            "fold_all_jaccard_bins": fold_all_jaccard_bins,
            "fold_all_recall_bins": fold_all_recall_bins,
            "fold_all_precision_bins": fold_all_precision_bins,
        }

    def _store_main_results(self, model_key: str, aggregated_metrics: Dict[str, List[List[float]]]):
        """
        Store main aggregated results in the results container.

        Args:
            model_key (str): Model-uncertainty combination identifier.
            aggregated_metrics (Dict[str, List[List[float]]]): Aggregated metrics from all folds.
        """
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        self.container_.add_main_result(model_key, aggregated_metrics["fold_jaccard_bins"])
        self.container_.add_target_separated_result(model_key, aggregated_metrics["fold_all_jaccard_bins"])

        self.container_.recall_results[model_key] = aggregated_metrics["fold_recall_bins"]
        self.container_.recall_target_separated[model_key] = aggregated_metrics["fold_all_recall_bins"]

        self.container_.precision_results[model_key] = aggregated_metrics["fold_precision_bins"]
        self.container_.precision_target_separated[model_key] = aggregated_metrics["fold_all_precision_bins"]

    def _store_target_separated_results(self, model_key: str, jaccard_results: List[JaccardBinResults]):
        """
        Store target-separated results for foldwise and overall analysis.

        Args:
            model_key (str): Model-uncertainty combination identifier.
            jaccard_results (List[JaccardBinResults]): Results from all folds.
        """
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        for fold_idx in range(len(jaccard_results)):
            result = jaccard_results[fold_idx]
            self._process_fold_target_separation(model_key, result)

    def _process_fold_target_separation(self, model_key: str, result: JaccardBinResults):
        """
        Process target separation for a single fold's results.

        Args:
            model_key (str): Model-uncertainty combination identifier.
            result (JaccardBinResults): Results from one fold.
        """
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        for target_idx in range(len(result.all_bins_concat_targets_sep)):
            self._initialize_target_containers_if_needed(model_key, target_idx)

            for bin_idx in range(self.current_num_bins_):
                if target_idx < len(self.container_.target_sep_foldwise):
                    self.container_.target_sep_foldwise[target_idx][model_key][bin_idx].extend(
                        result.all_bins_concat_targets_sep[target_idx][bin_idx]
                    )

                    self.container_.target_sep_all[target_idx][model_key][bin_idx].extend(
                        result.all_bins_concat_targets_sep[target_idx][bin_idx]
                    )

    def _initialize_target_containers_if_needed(self, model_key: str, target_idx: int):
        """
        Initialize target containers if they don't exist for the given model key.

        Args:
            model_key (str): Model-uncertainty combination identifier.
            target_idx (int): Index of the target.
        """
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        if target_idx < len(self.container_.target_sep_foldwise):
            if model_key not in self.container_.target_sep_foldwise[target_idx]:
                self.container_.target_sep_foldwise[target_idx][model_key] = [[] for _ in range(self.current_num_bins_)]

        if target_idx < len(self.container_.target_sep_all):
            if model_key not in self.container_.target_sep_all[target_idx]:
                self.container_.target_sep_all[target_idx][model_key] = [[] for _ in range(self.current_num_bins_)]

    def _finalize_results(self) -> Dict:
        """
        Convert aggregated results container into final Jaccard evaluation output format.

        Transforms the results container into the expected API output format with properly organized Jaccard similarity,
        precision, and recall results for all evaluation categories.

        Returns:
            Dict: Final results dictionary with keys defined in ResultKeys class:
                - JACCARD_ALL: Main Jaccard similarity results across all targets
                - JACCARD_TARGETS_SEPARATED: Target-separated Jaccard results
                - RECALL_ALL: Recall metrics for all model-uncertainty combinations
                - RECALL_TARGETS_SEPARATED: Target-separated recall results
                - PRECISION_ALL: Precision metrics for all combinations
                - PRECISION_TARGETS_SEPARATED: Target-separated precision results
                - ALL_JACC_CONCAT_BINS_TARGET_SEP_FOLDWISE: Fold-wise target separation
                - ALL_JACC_CONCAT_BINS_TARGET_SEP_ALL: Overall target separation
        """
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")
        return {
            ResultKeys.JACCARD_ALL: self.container_.main_results,
            ResultKeys.JACCARD_TARGETS_SEPARATED: self.container_.target_separated_results,
            ResultKeys.RECALL_ALL: self.container_.recall_results,
            ResultKeys.RECALL_TARGETS_SEPARATED: self.container_.recall_target_separated,
            ResultKeys.PRECISION_ALL: self.container_.precision_results,
            ResultKeys.PRECISION_TARGETS_SEPARATED: self.container_.precision_target_separated,
            ResultKeys.ALL_JACC_CONCAT_BINS_TARGET_SEP_FOLDWISE: self.container_.target_sep_foldwise,
            ResultKeys.ALL_JACC_CONCAT_BINS_TARGET_SEP_ALL: self.container_.target_sep_all,
        }


class BoundsEvaluator(BaseEvaluator):
    """
    Evaluator for the accuracy of estimated error bounds.

    For each quantile bin, this measures the proportion of predictions whose true error falls inside
    the bound estimated for that bin, which shows how well the estimated bounds are calibrated.

    Args:
        estimated_bounds (Dict[str, pd.DataFrame]): Estimated error bounds per model, keyed by
            ``"<model> Error Bounds"``.
        config (EvaluationConfig, optional): Evaluation settings. Defaults to :class:`EvaluationConfig`.

    Example:

        .. code-block:: pycon

            >>> evaluator = BoundsEvaluator(bounds, EvaluationConfig(original_num_bins=5))
            >>> results = evaluator.evaluate(bin_predictions, [["S-MHA"]], targets=[0, 1])
    """

    def __init__(self, estimated_bounds: Dict[str, pd.DataFrame], config: Optional[EvaluationConfig] = None):
        super().__init__(config or EvaluationConfig())
        self.estimated_bounds_ = estimated_bounds

    def _process_all_folds(self, data_structs: pd.DataFrame) -> List[BinResults]:
        """Process every fold, attaching that fold's estimated bounds to the fold data."""
        error_bounds = self.estimated_bounds_[self.current_model_ + " Error Bounds"]
        bounds_column = self.current_uncertainty_type_ + " Uncertainty bounds"

        fold_results = []
        for fold in range(self.config_.num_folds):
            fold_data = DataProcessor.extract_fold_data(data_structs, fold, self.current_uncertainty_type_)
            fold_data.bounds = strip_for_bound(error_bounds[error_bounds["fold"] == fold][bounds_column].values)
            fold_results.append(self._process_single_fold(fold_data))

        return fold_results

    def _process_single_fold(self, fold_data: FoldData) -> BinResults:
        """Evaluate bound accuracy for one fold."""
        if fold_data.bounds is None:
            raise RuntimeError("Fold data is missing its estimated bounds")
        result = bin_wise_bound_eval(
            fold_data.bounds,
            fold_data.errors,
            fold_data.bins,
            self.current_targets_,
            self.current_uncertainty_type_,
            num_bins=self.current_num_bins_,
        )

        return BinResults(
            mean_all_targets=result[ResultKeys.MEAN_ALL_TARGETS],
            mean_all_bins=result[ResultKeys.MEAN_ALL_BINS],
            all_bins=result["mean all"],
            all_bins_concat_targets_sep=result[ResultKeys.ALL_BINS_CONCAT_TARGETS_SEP],
        )

    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]) -> None:
        """Store bound accuracy by bin, fold, and target.

        For each model key, main results use [bin][fold] and unseparated accuracies use [bin][fold * target],
        ordered by fold then target. Both use descending uncertainty bins. The two target-separated outputs use
        [target][model_key][bin][fold], with ascending uncertainty bins and an accuracy value for every fold.

        Args:
            model_key (str): Model and uncertainty identifier used in the result containers.
            fold_results (List[BinResults]): Results in fold order, with bins ordered from lowest to highest
                uncertainty and targets ordered as in current_targets_.

        Raises:
            RuntimeError: If the results container has not been initialized.
        """
        num_bins = self.current_num_bins_
        num_targets = len(self.current_targets_)

        mean_bins: List[List[float]] = [[] for _ in range(num_bins)]
        bins_targets_not_sep: List[List[float]] = [[] for _ in range(num_bins)]
        targets_sep_foldwise: List[List[List[float]]] = [[[] for _ in range(num_bins)] for _ in range(num_targets)]
        targets_sep_all: List[List[List[float]]] = [[[] for _ in range(num_bins)] for _ in range(num_targets)]

        for fold in fold_results:
            for idx_bin in range(len(fold.mean_all_bins)):
                mean_bins[idx_bin].append(fold.mean_all_bins[idx_bin])
                bins_targets_not_sep[idx_bin].extend(fold.all_bins[idx_bin])

                for target_idx in range(num_targets):
                    fold_bin_values = fold.all_bins_concat_targets_sep[target_idx][idx_bin]
                    targets_sep_foldwise[target_idx][idx_bin].extend(fold_bin_values)
                    targets_sep_all[target_idx][idx_bin].extend(fold_bin_values)

        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        self.container_.add_main_result(model_key, mean_bins[::-1])
        self.container_.additional_containers.setdefault(ResultKeys.ALL_BOUND_PERCENTS_NO_TARGET_SEP, {})[
            model_key
        ] = bins_targets_not_sep[::-1]

        for target_idx in range(num_targets):
            self.container_.target_sep_foldwise[target_idx][model_key] = targets_sep_foldwise[target_idx]
            self.container_.target_sep_all[target_idx][model_key] = targets_sep_all[target_idx]

    def _finalize_results(self) -> Dict:
        """Map the container onto the error bound result keys."""
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        return {
            ResultKeys.ERROR_BOUNDS_ALL: self.container_.main_results,
            ResultKeys.ALL_BOUND_PERCENTS_NO_TARGET_SEP: self.container_.additional_containers.get(
                ResultKeys.ALL_BOUND_PERCENTS_NO_TARGET_SEP, {}
            ),
            ResultKeys.ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_FOLDWISE: self.container_.target_sep_foldwise,
            ResultKeys.ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_ALL: self.container_.target_sep_all,
        }


class ErrorsEvaluator(BaseEvaluator):
    """
    Evaluator for the mean localization error of each quantile bin.

    For each bin, this measures the mean error per target and across all targets, which shows whether
    predictions the model is less certain about do carry larger errors.

    Args:
        config (EvaluationConfig, optional): Evaluation settings, including ``error_scaling_factor``.
            Defaults to :class:`EvaluationConfig`.

    Example:

        .. code-block:: pycon

            >>> evaluator = ErrorsEvaluator(EvaluationConfig(original_num_bins=5))
            >>> results = evaluator.evaluate(bin_predictions, [["S-MHA"]], targets=[0, 1])
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        super().__init__(config or EvaluationConfig())

    def _process_single_fold(self, fold_data: FoldData) -> BinResults:
        """Evaluate mean bin errors for one fold."""
        result = bin_wise_errors(
            fold_data.errors,
            fold_data.bins,
            self.current_num_bins_,
            self.current_targets_,
            self.current_uncertainty_type_,
            error_scaling_factor=self.config_.error_scaling_factor,
        )

        return BinResults(
            mean_all_targets=result[ResultKeys.MEAN_ALL_TARGETS],
            mean_all_bins=result[ResultKeys.MEAN_ALL_BINS],
            all_bins=result[ResultKeys.ALL_BINS],
            all_bins_concat_targets_sep=result[ResultKeys.ALL_BINS_CONCAT_TARGETS_SEP],
        )

    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]) -> None:
        """Store mean and sample errors by bin, fold, and target.

        Main results use [bin][fold], with None for bins empty across all targets. Unseparated values concatenate
        folds, targets, and samples in that order. Target-separated errors use
        [target][model_key][bin][nonempty_fold][sample], or [target][model_key][bin][sample] with folds combined.
        All outputs use descending uncertainty bins; target-separated lists contain only nonempty folds.

        Args:
            model_key (str): Model and uncertainty identifier used in the result containers.
            fold_results (List[BinResults]): Results in fold order, with bins ordered from lowest to highest
                uncertainty and targets ordered as in current_targets_.

        Raises:
            RuntimeError: If the results container has not been initialized.
        """
        num_bins = self.current_num_bins_
        num_targets = len(self.current_targets_)

        mean_bins: List[List[Optional[float]]] = [[] for _ in range(num_bins)]
        all_bins: List[List[float]] = [[] for _ in range(num_bins)]
        concat_targets_no_sep: List[List[float]] = [[] for _ in range(num_bins)]
        targets_sep_foldwise: List[List[List[List[float]]]] = [
            [[] for _ in range(num_bins)] for _ in range(num_targets)
        ]
        targets_sep_all: List[List[List[float]]] = [[[] for _ in range(num_bins)] for _ in range(num_targets)]

        for fold in fold_results:
            for idx_bin in range(len(fold.mean_all_bins)):
                mean_bins[idx_bin].append(fold.mean_all_bins[idx_bin])
                all_bins[idx_bin].extend(fold.all_bins[idx_bin])

                # Flatten this bin's errors across every target, dropping the target separation.
                per_target = [target_bins[idx_bin] for target_bins in fold.all_bins_concat_targets_sep]
                sample_errors = [errors for target_values in per_target for errors in target_values]
                flattened = [value for errors in sample_errors for value in errors]
                concat_targets_no_sep[idx_bin].extend(flattened)

                for target_idx in range(num_targets):
                    fold_bin_values = fold.all_bins_concat_targets_sep[target_idx][idx_bin]
                    targets_sep_foldwise[target_idx][idx_bin].extend(fold_bin_values)
                    if fold_bin_values:
                        targets_sep_all[target_idx][idx_bin].extend(fold_bin_values[0])

        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        self.container_.add_main_result(model_key, mean_bins[::-1])
        self.container_.add_target_separated_result(model_key, all_bins[::-1])
        self.container_.additional_containers.setdefault(ResultKeys.ALL_ERROR_CONCAT_BINS_TARGET_NO_SEP, {})[
            model_key
        ] = concat_targets_no_sep[::-1]

        for target_idx in range(num_targets):
            self.container_.target_sep_foldwise[target_idx][model_key] = targets_sep_foldwise[target_idx][::-1]
            self.container_.target_sep_all[target_idx][model_key] = targets_sep_all[target_idx][::-1]

    def _finalize_results(self) -> Dict:
        """Map the container onto the mean error result keys."""
        if self.container_ is None:
            raise RuntimeError("Results container is not initialized")

        return {
            ResultKeys.ALL_MEAN_ERROR_BINS_NO_SEP: self.container_.main_results,
            ResultKeys.ALL_MEAN_ERROR_BINS_TARGETS_SEP: self.container_.target_separated_results,
            ResultKeys.ALL_ERROR_CONCAT_BINS_TARGET_NO_SEP: self.container_.additional_containers.get(
                ResultKeys.ALL_ERROR_CONCAT_BINS_TARGET_NO_SEP, {}
            ),
            ResultKeys.ALL_ERROR_CONCAT_BINS_TARGET_SEP_FOLDWISE: self.container_.target_sep_foldwise,
            ResultKeys.ALL_ERROR_CONCAT_BINS_TARGET_SEP_ALL: self.container_.target_sep_all,
        }


def evaluate_bounds(
    estimated_bounds: Dict[str, pd.DataFrame],
    bin_predictions: Dict[str, pd.DataFrame],
    uncertainty_pairs: List,
    num_bins: int,
    targets: List[int],
    num_folds: int = 8,
    combine_middle_bins: bool = False,
) -> Dict:
    """
    Evaluate error bound accuracy for uncertainty quantification models.

    Args:
        estimated_bounds (Dict[str, pd.DataFrame]): Estimated bounds keyed by "<model> Error Bounds".
        bin_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to DataFrames containing bin
            predictions and evaluation data with columns for UIDs, target indices, errors, and uncertainty bins.
        uncertainty_pairs (List): List of uncertainty type pairs to evaluate. Each element should be a list/tuple
            containing uncertainty type names.
        num_bins (int): Number of uncertainty bins for evaluation. Controls the granularity of the bound accuracy
            assessment.
        targets (List[int]): List of target indices to include in the evaluation. Used to filter data and organize
            results by target.
        num_folds (int, optional): Number of cross-validation folds for evaluation. Defaults to 8.
        combine_middle_bins (bool, optional): Whether to combine middle uncertainty bins for simplified three-bin
            analysis. Defaults to False.

    Returns:
        Dict: Comprehensive evaluation results dictionary containing:
            - Error bound accuracy statistics across bins and targets
            - Target-separated results for detailed analysis
            - Fold-wise results for cross-validation assessment
            - Bound coverage percentages and reliability metrics

    Example:

        .. code-block:: pycon

            >>> bounds = {'model1 Error Bounds': bounds_df}
            >>> predictions = {'model1': predictions_df}
            >>> results = evaluate_bounds(
            ...     bounds, predictions,
            ...     uncertainty_pairs=[['epistemic']],
            ...     num_bins=5, targets=[0, 1, 2]
            ... )
            >>> results['error_bounds_all']['model1 epistemic']
    """

    config = EvaluationConfig(
        num_folds=num_folds,
        original_num_bins=num_bins,
        combine_middle_bins=combine_middle_bins,
    )
    return BoundsEvaluator(estimated_bounds, config).evaluate(bin_predictions, uncertainty_pairs, targets)


def evaluate_jaccard(bin_predictions, uncertainty_pairs, num_bins, targets, num_folds=8, combine_middle_bins=False):
    """
    Evaluate uncertainty estimation's ability to predict true error quantiles using Jaccard metrics.

    Args:
        bin_predictions: Dictionary of DataFrames containing bin predictions for each model
        uncertainty_pairs: List of uncertainty pairs to evaluate
        num_bins: Number of quantile bins
        targets: List of targets to measure uncertainty estimation
        num_folds: Number of cross-validation folds
        combine_middle_bins: Whether to combine middle bins into one bin

    Returns:
        Dictionary containing Jaccard evaluation results
    """
    evaluator = JaccardEvaluator.create_simple(
        original_num_bins=num_bins, num_folds=num_folds, combine_middle_bins=combine_middle_bins
    )
    return evaluator.evaluate(bin_predictions, uncertainty_pairs, targets)


def _bin_error_bounds(q: int, num_bins: int, fold_bounds: list) -> Tuple[float, float]:
    """Return the ``(lower, upper]`` error bounds for quantile bin ``q``.

    Args:
        q (int): Index of the quantile bin.
        num_bins (int): Total number of quantile bins.
        fold_bounds (list): Estimated error bounds for this target, one per bin edge.

    Returns:
        tuple: The ``(lower, upper)`` bounds.
    """
    if q == 0:
        return 0, fold_bounds[q]
    if q < num_bins - 1:
        return fold_bounds[q - 1], fold_bounds[q]
    return fold_bounds[q - 1], float("inf")


def _count_within_bounds(errors: list, lower: float, upper: float) -> int:
    """Count how many ``errors`` fall in the half-open interval ``(lower, upper]``.

    Args:
        errors (list): Error values in a single bin.
        lower (float): Exclusive lower bound.
        upper (float): Inclusive upper bound.

    Returns:
        int: The number of errors within the bounds.
    """
    return sum(1 for error in errors if lower < error <= upper)


def _bin_accuracy(inbin_errors: list, lower: float, upper: float) -> float:
    """Return the fraction of ``inbin_errors`` within ``(lower, upper]``, or ``1.0`` if the bin is empty.

    Args:
        inbin_errors (list): Errors of the samples assigned to this bin.
        lower (float): Exclusive lower bound.
        upper (float): Inclusive upper bound.

    Returns:
        float: The proportion of errors within the bounds; ``1.0`` when the bin is empty.
    """
    if len(inbin_errors) == 0:
        return 1.0
    return _count_within_bounds(inbin_errors, lower, upper) / len(inbin_errors)


def _weighted_average(values: list, weights: list) -> float:
    """Return the ``weights``-weighted mean of ``values``, or ``0.0`` if the weights sum to zero.

    Args:
        values (list): Per-item values.
        weights (list): Non-negative weights aligned with ``values``.

    Returns:
        float: The weighted mean, or ``0.0`` when all weights are zero.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def bin_wise_bound_eval(
    fold_bounds_all_targets: list,
    fold_errors: pd.DataFrame,
    fold_bins: pd.DataFrame,
    targets: list,
    uncertainty_type: str,
    num_bins: int = 5,
) -> dict:
    """
    Compute error bound accuracy for each target and uncertainty bin in one fold.

    Args:
        fold_bounds_all_targets (list): A list of lists of estimated error bounds for each target.
        fold_errors (pd.DataFrame): A Pandas DataFrame containing the true errors for this fold.
        fold_bins (pd.DataFrame): A Pandas DataFrame containing the predicted quantile bins for this fold.
        targets (list): A list of targets to measure uncertainty estimation.
        uncertainty_type (str): The name of the uncertainty type to calculate accuracy for.
        num_bins (int): The number of quantile bins.

    Returns:
        dict: A dictionary containing the following error bound accuracy statistics:
              - 'mean all targets': The mean accuracy over all targets and quantile bins.
              - 'mean all bins': A list of mean accuracy values for each quantile bin (all targets included),
               weighted by bin size; ``0.0`` for a bin that is empty for every target.
              - 'mean all': Accuracy values organized as [bin][target].
              - 'all bins concatenated targets separated': A list of accuracy values for each quantile bin, concatenated
               for each target separately.

    Raises:
        ValueError: If an in-range prediction has no matching error, or a target has no samples in the fold.

    Example:

        .. code-block:: pycon

            >>> bin_wise_bound_eval(fold_bounds_all_targets, fold_errors, fold_bins, [0,1], 'S-MHA', num_bins=5)
    """
    all_target_perc = []
    all_qs_perc: List[List[float]] = [[] for x in range(num_bins)]
    all_qs_size: List[List[float]] = [[] for x in range(num_bins)]

    all_qs_errorbound_concat_targets_sep: List[List[List[float]]] = [
        [[] for y in range(num_bins)] for x in range(len(targets))
    ]

    for i_ti, target_idx in enumerate(targets):
        true_errors_ti = fold_errors[(fold_errors["Target Index"] == target_idx)][["uid", uncertainty_type + " Error"]]
        pred_bins_ti = fold_bins[(fold_bins["Target Index"] == target_idx)][
            ["uid", uncertainty_type + " Uncertainty bins"]
        ]

        true_errors_ti = dict(zip(true_errors_ti.uid, true_errors_ti[uncertainty_type + " Error"]))
        pred_bins_ti = dict(zip(pred_bins_ti.uid, pred_bins_ti[uncertainty_type + " Uncertainty bins"]))

        fold_bounds = fold_bounds_all_targets[i_ti]

        _, pred_bins_errors = DataProcessor.group_data_by_bins(true_errors_ti, pred_bins_ti, num_bins)

        bins_acc = []
        bins_sizes = []
        for q in range((num_bins)):
            inbin_errors = pred_bins_errors[q]

            lower, upper = _bin_error_bounds(q, num_bins, fold_bounds)
            accuracy_bin = _bin_accuracy(inbin_errors, lower, upper)
            bins_sizes.append(len(inbin_errors))
            bins_acc.append(accuracy_bin)

            all_qs_perc[q].append(accuracy_bin)
            all_qs_size[q].append(len(inbin_errors))
            all_qs_errorbound_concat_targets_sep[i_ti][q].append(accuracy_bin)

        if sum(bins_sizes) == 0:
            raise ValueError(
                f"Target {target_idx} has no samples in this fold for uncertainty type {uncertainty_type}."
            )
        all_target_perc.append(_weighted_average(bins_acc, bins_sizes))

    weighted_ave_binwise = []
    for binidx in range(len(all_qs_perc)):
        weighted_ave_binwise.append(_weighted_average(all_qs_perc[binidx], all_qs_size[binidx]))

    return {
        "mean all targets": np.mean(all_target_perc),
        "mean all bins": weighted_ave_binwise,
        "mean all": all_qs_perc,
        "all bins concatenated targets separated": all_qs_errorbound_concat_targets_sep,
    }


def get_mean_errors(
    bin_predictions: Dict[str, "pd.DataFrame"],
    uncertainty_pairs: List,
    num_bins: int,
    targets: List[int],
    num_folds: int = 8,
    error_scaling_factor: float = 1.0,
    combine_middle_bins: bool = False,
) -> Dict:
    """
    Compute mean localization errors and collect sample errors by uncertainty bin across folds and targets.

    Args:
        bin_predictions (Dict): Dict of Pandas DataFrames where each DataFrame has errors, predicted bins for all
            uncertainty measures for a model.
        uncertainty_pairs (List[Tuple[str, str]]): List of tuples describing the different uncertainty combinations to test.
        num_bins (int): Number of quantile bins.
        targets (List[str]): List of targets to measure uncertainty estimation.
        num_folds (int, optional): Number of folds. Defaults to 8.
        error_scaling_factor (int, optional): Scale error factor. Defaults to 1.
        combine_middle_bins (bool, optional): Combine middle bins if True. Defaults to False.

    Returns:
        Dict[str, Union[Dict[str, List[List[float]]], List[Dict[str, List[float]]]]]: Dictionary with mean error for all
         targets combined and targets separated.
            Keys that are returned:
                "all_mean_error_bins_nosep":  For every fold, the mean error for each bin. All targets are combined in the same list.
                "all mean error bins targets sep":   For every fold, the mean error for each bin. Each target is in a separate list.
                "all_error_concat_bins_targets_nosep":  For every fold, every error value in a list. Each target is in the same list. The list is flattened for all the folds.
                "all error concat bins targets sep foldwise":  For every fold, every error value in a list. Each target is in a separate list. Each list has a list of results by fold.
                "all_error_concat_bins_targets_sep_all": For every fold, every error value in a list. Each target is in a separate list. The list is flattened for all the folds.

    """
    config = EvaluationConfig(
        num_folds=num_folds,
        original_num_bins=num_bins,
        error_scaling_factor=error_scaling_factor,
        combine_middle_bins=combine_middle_bins,
    )
    return ErrorsEvaluator(config).evaluate(bin_predictions, uncertainty_pairs, targets)


def bin_wise_errors(fold_errors, fold_bins, num_bins, targets, uncertainty_key, error_scaling_factor):
    """
    Compute mean and sample errors for each target and uncertainty bin in one fold.

    Args:
        fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
        fold_bins (Pandas Dataframe): Pandas Dataframe of predicted quantile bins for this fold.
        num_bins (int): Number of quantile bins.
        targets (list): list of targets to measure uncertainty estimation.
        uncertainty_key (string): Uncertainty type identifying the error and bin columns.
        error_scaling_factor (float): Factor to scale the errors by.


    Returns:
        [Dict]: Dict with mean error statistics.

    Raises:
        ValueError: If an in-range prediction has no matching error.
    """

    all_target_error = []
    all_qs_error = [[] for x in range(num_bins)]
    all_qs_error_concat_targets_sep = [[[] for y in range(num_bins)] for x in range(len(targets))]

    for i, target_idx in enumerate(targets):
        true_errors_ti = fold_errors[(fold_errors["Target Index"] == target_idx)][["uid", uncertainty_key + " Error"]]
        pred_bins_ti = fold_bins[(fold_bins["Target Index"] == target_idx)][
            ["uid", uncertainty_key + " Uncertainty bins"]
        ]

        true_errors_ti = dict(
            zip(true_errors_ti.uid, true_errors_ti[uncertainty_key + " Error"] * error_scaling_factor)
        )
        pred_bins_ti = dict(zip(pred_bins_ti.uid, pred_bins_ti[uncertainty_key + " Uncertainty bins"]))

        _, pred_bins_errors = DataProcessor.group_data_by_bins(true_errors_ti, pred_bins_ti, num_bins)

        inner_errors = []
        for bin in range(num_bins):
            pred_b_errors = pred_bins_errors[bin]

            if pred_b_errors == []:
                continue

            mean_error = np.mean(pred_b_errors)
            all_qs_error[bin].append(mean_error)
            all_qs_error_concat_targets_sep[i][bin].append(pred_b_errors)
            inner_errors.append(mean_error)

        all_target_error.append(np.mean(inner_errors))

    mean_all_targets = np.mean(all_target_error)
    mean_all_bins = []
    for x in all_qs_error:
        if x == []:
            mean_all_bins.append(None)
        else:
            mean_all_bins.append(np.mean(x))

    return {
        "mean all targets": mean_all_targets,
        "mean all bins": mean_all_bins,
        "all bins": all_qs_error,
        "all bins concatenated targets separated": all_qs_error_concat_targets_sep,
    }

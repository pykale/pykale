# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
# =============================================================================
"""
Module from the implementation of L. A. Schobs, A. J. Swift and H. Lu, "Uncertainty Estimation for Heatmap-Based Landmark Localization,"
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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

from kale.prepdata.string_transform import strip_for_bound


class ColumnNames:
    """Constants for DataFrame column names."""

    UID = "uid"
    TARGET_IDX = "target_idx"
    TESTING_FOLD = "Testing Fold"
    ERROR_SUFFIX = " Error"
    UNCERTAINTY_BINS_SUFFIX = " Uncertainty bins"
    UNCERTAINTY_BOUNDS_SUFFIX = " Uncertainty bounds"


class ResultKeys:
    """Constants for result dictionary keys."""

    MEAN_ALL_TARGETS = "mean all targets"
    MEAN_ALL_BINS = "mean all bins"
    ALL_BINS = "all bins"
    ALL_BINS_CONCAT_TARGETS_SEP = "all bins concatenated targets seperated"

    # Bounds specific
    ERROR_BOUNDS_ALL = "Error Bounds All"
    ALL_BOUND_PERCENTS_NO_TARGET_SEP = "all_bound_percents_notargetsep"
    ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all errorbound concat bins targets sep foldwise"
    ALL_ERROR_BOUND_CONCAT_BINS_TARGET_SEP_ALL = "all errorbound concat bins targets sep all"

    # Errors specific
    ALL_MEAN_ERROR_BINS_NO_SEP = "all mean error bins nosep"
    ALL_MEAN_ERROR_BINS_TARGETS_SEP = "all mean error bins targets sep"
    ALL_ERROR_CONCAT_BINS_TARGET_NO_SEP = "all error concat bins targets nosep"
    ALL_ERROR_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all error concat bins targets sep foldwise"
    ALL_ERROR_CONCAT_BINS_TARGET_SEP_ALL = "all error concat bins targets sep all"

    # Jaccard specific
    JACCARD_ALL = "Jaccard All"
    JACCARD_TARGETS_SEPARATED = "Jaccard targets seperated"
    RECALL_ALL = "Recall All"
    RECALL_TARGETS_SEPARATED = "Recall targets seperated"
    PRECISION_ALL = "Precision All"
    PRECISION_TARGETS_SEPARATED = "Precision targets seperated"
    ALL_JACC_CONCAT_BINS_TARGET_SEP_FOLDWISE = "all jacc concat bins targets sep foldwise"
    ALL_JACC_CONCAT_BINS_TARGET_SEP_ALL = "all jacc concat bins targets sep all"


@dataclass
class EvaluationConfig:
    """
    Configuration parameters for uncertainty quantification evaluation.

    This dataclass defines the settings and parameters used throughout the
    evaluation process for uncertainty quantification metrics. It provides
    default values for common evaluation scenarios while allowing customization
    for specific research needs.

    Attributes:
        num_folds (int): Number of cross-validation folds for evaluation.
            Defaults to 8. Higher values provide more robust statistical
            estimates but increase computational cost.
        original_num_bins (int): Number of quantile bins for uncertainty
            evaluation. Defaults to 10. Controls the granularity of
            uncertainty analysis.
        error_scaling_factor (float): Scaling factor applied to prediction
            errors during evaluation. Defaults to 1.0 (no scaling).
        combine_middle_bins (bool): Whether to combine middle uncertainty
            bins for simplified analysis. Defaults to False. When True,
            reduces evaluation complexity by merging intermediate quantiles.
        combined_num_bins (int): Number of bins when middle bins are combined.
            Defaults to 3 (low, medium, high uncertainty). Only used when
            combine_middle_bins is True.

    Example:
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

    This dataclass organizes the data required for evaluating a single fold
    in cross-validation, including prediction errors, uncertainty bins, and
    optional error bounds information.

    Attributes:
        errors (pd.DataFrame): DataFrame containing prediction errors for the fold.
            Expected columns include UID, target_idx, and model-specific error columns.
        bins (pd.DataFrame): DataFrame containing uncertainty bin assignments.
            Expected columns include UID, target_idx, and uncertainty bin columns.
        bounds (Optional[List]): Optional list containing error bound information
            for bound-based evaluation methods. Defaults to None.

    Example:
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

    This dataclass stores the fundamental evaluation metrics computed for a single
    cross-validation fold, organized by targets and bins. Serves as the base class
    for specialized result containers like JaccardBinResults.

    Attributes:
        mean_all_targets (float): Mean evaluation metric across all targets in the fold.
        mean_all_bins (List[float]): Mean evaluation metric for each bin across targets.
            Length equals the number of bins.
        all_bins (List[List[float]]): Raw evaluation metrics for each bin and target.
            Outer list represents bins, inner lists contain values for each target.
        all_bins_concat_targets_sep (List[List[List[float]]]): Evaluation metrics
            organized for target-separated analysis. Structure: [target][bin][values].

    Note:
        This base class provides the common structure for fold-level results.
        Subclasses can extend it with additional metrics specific to their
        evaluation type (e.g., precision and recall for Jaccard evaluation).
    """

    mean_all_targets: float
    mean_all_bins: List[float]
    all_bins: List[List[float]]
    all_bins_concat_targets_sep: List[List[List[float]]]


@dataclass
class JaccardBinResults(BinResults):
    """
    Extended results container for Jaccard similarity evaluation with precision and recall.

    This specialized container extends BinResults to include additional metrics
    specific to Jaccard similarity evaluation: precision and recall. These metrics
    provide comprehensive assessment of uncertainty quantification quality.

    Attributes:
        Inherits from BinResults:
            - mean_all_targets: Mean Jaccard similarity across all targets
            - mean_all_bins: Mean Jaccard similarity for each bin
            - all_bins: Raw Jaccard similarities for each bin and target
            - all_bins_concat_targets_sep: Target-separated Jaccard similarities

        Additional Jaccard-specific attributes:
            mean_all_targets_recall (float): Mean recall across all targets in the fold.
            mean_all_bins_recall (List[float]): Mean recall for each bin across targets.
            all_bins_recall (List[List[float]]): Raw recall values for each bin and target.
            mean_all_targets_precision (float): Mean precision across all targets.
            mean_all_bins_precision (List[float]): Mean precision for each bin.
            all_bins_precision (List[List[float]]): Raw precision values for each bin and target.

    Note:
        Precision measures the accuracy of positive predictions within each bin,
        while recall measures the coverage of actual positive cases. Combined with
        Jaccard similarity, these metrics provide a complete picture of uncertainty
        quantification performance.
    """

    mean_all_targets_recall: float = 0.0
    mean_all_bins_recall: List[float] = field(default_factory=list)
    all_bins_recall: List[List[float]] = field(default_factory=list)
    mean_all_targets_precision: float = 0.0
    mean_all_bins_precision: List[float] = field(default_factory=list)
    all_bins_precision: List[List[float]] = field(default_factory=list)


class ResultsContainer:
    """
    Container for organizing and managing complex nested evaluation results.

    This class provides a structured way to organize evaluation results across
    different dimensions: models, uncertainty types, bins, targets, and folds.
    It handles both main aggregated results and target-separated detailed results.

    The container supports multiple evaluation metrics and provides methods for
    adding results in an organized manner. It's designed to work with the
    Template Method pattern in BaseEvaluator.

    Attributes:
        num_bins (int): Number of uncertainty bins in the evaluation.
        num_targets (int): Number of target labels being evaluated.
        main_results (Dict): Main aggregated results across all targets.
        target_separated_results (Dict): Results separated by individual targets.
        target_sep_foldwise (List[Dict]): Target-separated results for each fold.
        target_sep_all (List[Dict]): Target-separated results aggregated across folds.
        additional_containers (Dict): Container for evaluation-specific results.

    Jaccard-specific attributes:
        recall_results (Dict): Recall metrics for all model-uncertainty combinations.
        recall_target_separated (Dict): Target-separated recall results.
        precision_results (Dict): Precision metrics for all combinations.
        precision_target_separated (Dict): Target-separated precision results.

    Example:
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
        """
        Initialize all result containers with empty data structures.

        Sets up the internal data structures needed to organize evaluation results
        across different dimensions (main results, target-separated results,
        fold-wise results, and evaluation-specific containers).
        """
        # Main results
        self.main_results = {}
        self.target_separated_results = {}

        # Target separated containers
        self.target_sep_foldwise = [{} for _ in range(self.num_targets)]
        self.target_sep_all = [{} for _ in range(self.num_targets)]

        # Additional containers for specific evaluations
        self.additional_containers = {}

        # Add missing attributes for JaccardEvaluator
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

    def update_target_containers(self, key: str, foldwise_data, all_data):
        """Update target-separated containers."""
        for target_idx in range(self.num_targets):
            self.target_sep_foldwise[target_idx][key] = foldwise_data[target_idx]
            self.target_sep_all[target_idx][key] = all_data[target_idx]

    def to_dict(self) -> Dict:
        """Convert to dictionary format matching original API."""
        result = {}
        result.update(self.main_results)
        result.update(self.target_separated_results)
        result.update(self.additional_containers)
        return result


class DataProcessor:
    """
    Utility class for data processing operations in uncertainty evaluation.

    This class provides static methods for extracting and filtering data from
    DataFrames for evaluation purposes. It handles fold-specific data extraction,
    target filtering, and data structure preparation for evaluation workflows.

    The class is designed as a collection of utility methods that can be used
    across different evaluator implementations without maintaining state.

    Key Operations:
        - Extract data for specific cross-validation folds
        - Filter data by target indices
        - Prepare data structures for evaluation processing
        - Handle column name mapping and data type conversions

    Example:
        >>> processor = DataProcessor()
        >>> fold_data = processor.extract_fold_data(df, fold=0, uncertainty_type="epistemic")
        >>> filtered_data = processor.filter_by_target(fold_data.errors, target_idx=1)
    """

    @staticmethod
    def extract_fold_data(data_structs: pd.DataFrame, fold: int, uncertainty_type: str) -> FoldData:
        """
        Extract data for a specific cross-validation fold and uncertainty type.

        Filters the input DataFrame to extract only the data belonging to the specified
        fold and prepares separate DataFrames for errors and uncertainty bins.

        Args:
            data_structs (pd.DataFrame): Complete dataset containing all folds and data.
                Must include columns for fold identification, UIDs, target indices,
                errors, and uncertainty bins.
            fold (int): The specific fold number to extract (0-based indexing).
            uncertainty_type (str): Type of uncertainty to extract (e.g., "epistemic", "aleatoric").
                Used to construct column names for error and bin data.

        Returns:
            FoldData: Container with filtered errors and bins DataFrames for the specified
                fold and uncertainty type. Contains UIDs, target indices, and corresponding
                error and bin values.

        Note:
            The method expects specific column naming conventions:
            - Error columns: "{uncertainty_type} Error"
            - Bin columns: "{uncertainty_type} Uncertainty bins"
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

        Organizes prediction keys and errors into bin-wise groups based on their
        uncertainty bin assignments. This enables bin-wise evaluation of prediction
        quality and uncertainty calibration.

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values.
                Keys should correspond to unique prediction identifiers.
            bins_dict (Dict): Dictionary mapping prediction keys to bin assignments.
                Values should be bin indices (0 to num_bins-1).
            num_bins (int): Total number of uncertainty bins used in the evaluation.

        Returns:
            Tuple[List[List], List[List]]: A tuple containing:
                - bin_keys: List of lists where bin_keys[i] contains all prediction
                  keys assigned to bin i.
                - bin_errors: List of lists where bin_errors[i] contains all error
                  values for predictions assigned to bin i.

        Example:
            >>> errors = {'pred1': 0.1, 'pred2': 0.3, 'pred3': 0.2}
            >>> bins = {'pred1': 0, 'pred2': 1, 'pred3': 0}
            >>> keys, errs = DataProcessor.group_data_by_bins(errors, bins, 2)
            >>> # keys[0] = ['pred1', 'pred3'], keys[1] = ['pred2']
            >>> # errs[0] = [0.1, 0.2], errs[1] = [0.3]
        """
        bin_keys: List[List] = [[] for _ in range(num_bins)]
        bin_errors: List[List] = [[] for _ in range(num_bins)]

        for bin_idx in range(num_bins):
            keys = [key for key, val in bins_dict.items() if str(bin_idx) == str(val)]
            errors = [errors_dict[key] for key in keys if key in errors_dict]

            bin_keys[bin_idx] = keys
            bin_errors[bin_idx] = errors

        return bin_keys, bin_errors


class QuantileCalculator:
    """
    Utility class for calculating quantile-based error distributions and thresholds.

    This class provides methods for computing quantile boundaries and grouping
    prediction errors into quantile-based bins. It supports both standard
    quantile binning and combined middle bin configurations for simplified analysis.

    The quantile approach enables analysis of prediction quality across different
    uncertainty levels by creating bins that contain equal numbers of predictions
    but varying error characteristics.

    Key Features:
        - Automatic quantile threshold calculation
        - Flexible bin combination for middle quantiles
        - Error and key grouping by quantile boundaries
        - Support for worst-to-best ordering (B5 to B1)

    Example:
        >>> calculator = QuantileCalculator()
        >>> errors = {'pred1': 0.1, 'pred2': 0.5, 'pred3': 0.3}
        >>> thresholds, err_groups, key_groups = calculator.calculate_error_quantiles(
        ...     errors, num_bins=3, combine_middle_bins=False
        ... )
    """

    @staticmethod
    def calculate_error_quantiles(
        errors_dict: Dict, num_bins: int, combine_middle_bins: bool
    ) -> Tuple[List[float], List[List], List[List]]:
        """
        Calculate quantile thresholds and group errors into quantile-based bins.

        Computes quantile boundaries based on the error distribution and groups
        errors and their corresponding keys into bins. Supports combining middle
        bins for simplified three-bin analysis (low, medium, high error).

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values.
                Used to compute quantile boundaries from the error distribution.
            num_bins (int): Number of quantile bins to create. Determines the
                granularity of the quantile analysis.
            combine_middle_bins (bool): Whether to combine middle quantiles into
                a single bin. When True, creates 3 bins regardless of num_bins.

        Returns:
            Tuple[List[float], List[List], List[List]]: A tuple containing:
                - quantile_thresholds: List of quantile boundary values
                - error_groups: List of error value lists for each bin (worst to best)
                - key_groups: List of prediction key lists for each bin (worst to best)

        Note:
            Results are ordered from worst to best performance (B5 to B1 convention)
            to match the expected evaluation output format.

        Example:
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

        Helper method that partitions prediction errors and their corresponding keys
        into groups based on quantile threshold boundaries.

        Args:
            errors_dict (Dict): Dictionary mapping prediction keys to error values.
            thresholds (List[float]): List of quantile threshold values that define
                the boundaries between groups.

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

        Helper method that determines whether a given error value belongs to
        the specified quantile group based on threshold boundaries.

        Args:
            error (float): The error value to classify.
            quantile_idx (int): Index of the quantile group to check (0-based).
            thresholds (List[float]): List of quantile threshold values.

        Returns:
            bool: True if the error falls within the specified quantile range,
                False otherwise.
        """
        if quantile_idx == 0:
            return error <= thresholds[0]
        elif quantile_idx < len(thresholds):
            return thresholds[quantile_idx - 1] < error <= thresholds[quantile_idx]
        else:
            return error > thresholds[quantile_idx - 1]


class MetricsCalculator:
    """
    Utility class for calculating evaluation metrics in uncertainty quantification.

    This class provides static methods for computing various metrics used to assess
    the quality of uncertainty quantification, including Jaccard similarity, precision,
    recall, and bound accuracy. The methods are designed to work with different
    evaluation strategies and provide consistent metric calculations.

    Key Metrics:
        - Jaccard Similarity: Measures overlap between predicted and ground truth sets
        - Precision: Accuracy of positive predictions within bins
        - Recall: Coverage of actual positive cases by predictions
        - Bound Accuracy: Whether errors fall within expected confidence bounds

    The class supports both binary classification metrics (for set-based evaluation)
    and regression-style bound checking for error prediction accuracy.

    Example:
        >>> calculator = MetricsCalculator()
        >>> jaccard, recall, precision = calculator.calculate_jaccard_metrics(
        ...     predicted_keys=['p1', 'p2'], ground_truth_keys=['p1', 'p3']
        ... )
        >>> # jaccard ≈ 0.33, recall = 0.5, precision = 0.5
    """

    @staticmethod
    def calculate_jaccard_metrics(predicted_keys: List, ground_truth_keys: List) -> Tuple[float, float, float]:
        """
        Calculate Jaccard similarity, recall, and precision for set-based evaluation.

        Computes three key metrics for assessing the overlap between predicted and
        ground truth sets of prediction keys. These metrics provide comprehensive
        evaluation of uncertainty quantification quality within bins.

        Args:
            predicted_keys (List): List of prediction keys identified by the model
                as belonging to a specific uncertainty or error bin.
            ground_truth_keys (List): List of prediction keys that actually belong
                to the target bin based on true error characteristics.

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
            >>> pred = ['sample1', 'sample2', 'sample3']
            >>> gt = ['sample1', 'sample4']
            >>> jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics(pred, gt)
            >>> # jaccard = 1/4 = 0.25, recall = 1/2 = 0.5, precision = 1/3 ≈ 0.33
        """
        all_keys = list(set(predicted_keys + ground_truth_keys))

        jaccard = 0.0
        if len(all_keys) != 0:
            pred_binary = [1 if key in predicted_keys else 0 for key in all_keys]
            gt_binary = [1 if key in ground_truth_keys else 0 for key in all_keys]
            jaccard = jaccard_score(gt_binary, pred_binary, zero_division=0)

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
        Check if error falls within expected bounds for the bin.

        Determines whether a prediction error falls within the expected error bounds
        for a specific uncertainty bin. This method is used to assess the accuracy
        of uncertainty-based error predictions.

        Args:
            error (float): The prediction error value to check.
            bin_idx (int): Index of the uncertainty bin (0-based).
            bounds (List[float]): List of error boundary values that define
                the expected error ranges for each bin.

        Returns:
            bool: True if the error falls within the expected bounds for the
                specified bin, False otherwise.

        Note:
            Bin boundaries are exclusive on the lower bound and inclusive on the
            upper bound, except for bin 0 which starts from 0 (exclusive).
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

    This class implements the Template Method pattern to provide a consistent evaluation
    framework while allowing specialized implementations for different metrics (Jaccard,
    error bounds, etc.). It manages the evaluation workflow across multiple folds,
    models, and uncertainty types.

    Design Pattern:
        Uses Template Method pattern where the main evaluation flow is defined in the
        base class, while specific evaluation logic is implemented by subclasses.

    Key Features:
        - Cross-validation fold processing
        - Multi-model and multi-uncertainty type support
        - Configurable bin combining and scaling
        - Result aggregation and formatting

    Workflow:
        1. Process each model and uncertainty type combination
        2. Extract data for each cross-validation fold
        3. Apply subclass-specific evaluation (_process_single_fold)
        4. Aggregate results across folds (_aggregate_fold_results)
        5. Format final results (_finalize_results)

    Attributes:
        config_ (EvaluationConfig): Configuration containing evaluation parameters
        current_num_bins_ (int): Number of bins for current evaluation (may differ from original)
        current_targets_ (List[int]): Target indices for current evaluation
        current_uncertainty_type_ (str): Current uncertainty type being processed

    Note:
        This is an abstract base class. Use concrete implementations like JaccardEvaluator
        for actual evaluations.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize BaseEvaluator with evaluation configuration.

        Args:
            config (EvaluationConfig): Configuration object containing evaluation
                parameters such as number of folds, bins, and processing options.
        """
        self.config_ = config
        # Instance variables to reduce parameter passing
        self.container_: ResultsContainer
        self.current_num_bins_: int = config.original_num_bins
        self.current_targets_: List[int] = []
        self.current_uncertainty_type_: str = ""

    def evaluate(self, bin_predictions: Dict[str, pd.DataFrame], uncertainty_pairs: List, targets: List[int]) -> Dict:
        """
        Main evaluation method implementing the template method pattern.

        Orchestrates the complete evaluation process across all models, uncertainty types,
        and cross-validation folds. This method defines the evaluation workflow while
        delegating specific evaluation logic to subclass implementations.

        Args:
            bin_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to
                DataFrames containing bin predictions and evaluation data. Each DataFrame
                should contain columns for UIDs, target indices, errors, and uncertainty bins.
            uncertainty_pairs (List): List of uncertainty type pairs to evaluate.
                Each pair contains uncertainty type names (e.g., ['epistemic'], ['aleatoric']).
            targets (List[int]): List of target indices to include in the evaluation.
                Used to filter data and organize results by target.

        Returns:
            Dict: Comprehensive evaluation results dictionary. Structure depends on the
                specific evaluator implementation but typically includes:
                - Main results across all folds and targets
                - Target-separated results for detailed analysis
                - Additional metrics specific to the evaluation type

        Workflow:
            1. Initialize result containers for data organization
            2. For each model and uncertainty type combination:
               a. Process all cross-validation folds
               b. Aggregate fold results
            3. Finalize and format results for output

        Note:
            This method coordinates the evaluation process but delegates the actual
            evaluation logic to abstract methods implemented by subclasses.
        """
        # Set instance variables to reduce parameter passing
        self.current_targets_ = targets
        self.current_num_bins_ = (
            self.config_.combined_num_bins if self.config_.combine_middle_bins else self.config_.original_num_bins
        )

        self.container_ = ResultsContainer(self.current_num_bins_, len(targets))

        for model, data_structs in bin_predictions.items():
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

        This abstract method must be implemented by subclasses to define how
        evaluation metrics are calculated for data from a single fold.

        Args:
            fold_data (FoldData): Container with errors and bins data for one fold.
                Contains filtered DataFrames for the current fold, uncertainty type,
                and any additional data needed for evaluation.

        Returns:
            BinResults: Results structure containing evaluation metrics for this fold.
                The specific subclass of BinResults depends on the evaluation type
                (e.g., JaccardBinResults for Jaccard evaluation).

        Note:
            Subclasses should implement their specific evaluation logic here,
            such as calculating Jaccard similarity, error bounds, or other metrics.
        """
        pass

    @abstractmethod
    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]):
        """
        Aggregate results across all folds for a model-uncertainty combination.

        This abstract method defines how fold-level results are combined and
        stored in the results container for final output formatting.

        Args:
            model_key (str): Identifier for the current model-uncertainty combination
                (format: "model_name uncertainty_type").
            fold_results (List[BinResults]): List of evaluation results from all folds
                for the current model-uncertainty combination.

        Note:
            Subclasses should implement aggregation logic specific to their evaluation
            type, including proper handling of target separation and result formatting.
        """
        pass

    @abstractmethod
    def _finalize_results(self) -> Dict:
        """
        Convert results container into final output format.

        This abstract method handles the final formatting of evaluation results
        to match the expected API output format.

        Returns:
            Dict: Final results dictionary with keys matching the expected API format.
                Structure depends on the evaluation type but typically includes
                main results, target-separated results, and evaluation-specific metrics.

        Note:
            Subclasses should map container data to appropriate result keys
            defined in ResultKeys class.
        """
        pass

    def _process_all_folds(self, data_structs: pd.DataFrame) -> List[BinResults]:
        """
        Process all cross-validation folds for a given model and uncertainty type.

        Iterates through all configured cross-validation folds, extracting data for each
        fold and applying the subclass-specific evaluation logic to compute fold-level
        results. This method coordinates the fold-wise evaluation process within the
        Template Method pattern.

        Args:
            data_structs (pd.DataFrame): Complete dataset containing all folds and evaluation
                data for the current model. Must include columns for fold identification,
                UIDs, target indices, errors, and uncertainty bins.

        Returns:
            List[BinResults]: List of evaluation results from all folds, where each element
                contains the evaluation metrics (e.g., Jaccard similarity, error bounds)
                computed for one fold. The specific type of BinResults depends on the
                evaluator implementation (e.g., JaccardBinResults for Jaccard evaluation).

        Note:
            This method uses the current instance variables (current_uncertainty_type_,
            config_.num_folds) to determine which data to extract and how many folds
            to process. It delegates the actual evaluation logic to _process_single_fold.
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

    This evaluator computes Jaccard similarity between prediction confidence bins
    and error bins to assess the quality of uncertainty quantification. It measures
    how well the model's confidence aligns with actual prediction accuracy.

    The Jaccard similarity is calculated as:
        J(A, B) = |A ∩ B| / |A ∪ B|

    Where A represents the high-confidence predictions and B represents
    the correct predictions within each bin.

    Attributes:
        config_ (EvaluationConfig): Configuration object containing evaluation parameters
            including bin counts, confidence thresholds, and target separation settings.
        data_processor (DataProcessor): Handles data filtering and preprocessing operations.
        quantile_calculator (QuantileCalculator): Computes quantile-based bin boundaries.
        metrics_calculator (MetricsCalculator): Calculates evaluation metrics including
            Jaccard similarity and statistical measures.

    Example:
        >>> config = EvaluationConfig(n_bins=10, targets_to_separate=['label1', 'label2'])
        >>> evaluator = JaccardEvaluator(config)
        >>> results = evaluator.evaluate(df, uncertainty_columns=['epistemic', 'aleatoric'])
        >>> print(results['jaccard_main']['model1_epistemic'])

    Note:
        This evaluator implements the Template Method pattern defined in BaseEvaluator,
        providing specific implementations for Jaccard similarity calculation and
        bin-wise evaluation of uncertainty quantification quality.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize JaccardEvaluator with configuration and required components.

        Sets up the evaluator with configuration parameters and calls the parent
        class constructor to initialize the evaluation framework.

        Args:
            config (Optional[EvaluationConfig]): Configuration object containing
                evaluation parameters such as bin counts, thresholds, and settings
                for target separation. If None, uses default EvaluationConfig values.

        Note:
            Inherits utility components (data_processor, quantile_calculator,
            metrics_calculator) from BaseEvaluator initialization.
        """
        self.config_ = config or EvaluationConfig()
        super().__init__(self.config_)

    @classmethod
    def create_simple(
        cls, original_num_bins: int, num_folds: int = 8, combine_middle_bins: bool = False
    ) -> "JaccardEvaluator":
        """
        Create a JaccardEvaluator instance with simplified parameters.

        Convenience factory method for creating a JaccardEvaluator with commonly
        used configuration parameters without requiring full EvaluationConfig setup.

        Args:
            original_num_bins (int): Number of quantile bins for uncertainty evaluation.
                Typical values range from 5 to 20 depending on dataset size and
                desired granularity.
            num_folds (int, optional): Number of cross-validation folds for evaluation.
                Defaults to 8. Higher values provide more robust estimates but
                increase computation time.
            combine_middle_bins (bool, optional): Whether to combine middle uncertainty
                bins for simplified analysis. Defaults to False. When True, reduces
                the number of bins by merging middle quantiles.

        Returns:
            JaccardEvaluator: Configured evaluator instance ready for uncertainty
                quantification assessment.

        Example:
            >>> evaluator = JaccardEvaluator.create_simple(
            ...     original_num_bins=10,
            ...     num_folds=5,
            ...     combine_middle_bins=True
            ... )
            >>> results = evaluator.evaluate(predictions_df, uncertainty_columns=['epistemic'])
        """
        config = EvaluationConfig(
            original_num_bins=original_num_bins, num_folds=num_folds, combine_middle_bins=combine_middle_bins
        )
        return cls(config)

    @classmethod
    def create_default(cls) -> "JaccardEvaluator":
        """
        Create a JaccardEvaluator instance with default configuration parameters.

        Factory method that creates an evaluator using the default EvaluationConfig
        settings, providing a quick way to instantiate the evaluator for standard
        uncertainty quantification evaluation tasks.

        Returns:
            JaccardEvaluator: Evaluator instance configured with default parameters
                including standard bin counts, fold numbers, and evaluation settings.

        Example:
            >>> evaluator = JaccardEvaluator.create_default()
            >>> results = evaluator.evaluate(data, uncertainty_columns=['uncertainty'])
        """
        return cls()

    def _process_single_fold(self, fold_data: FoldData) -> JaccardBinResults:
        """
        Process Jaccard evaluation metrics for a single cross-validation fold.

        Computes Jaccard similarity, precision, and recall metrics for each bin
        and target within a single fold. This method implements the core evaluation
        logic for assessing uncertainty quantification quality.

        Args:
            fold_data (FoldData): Container with errors and bins data for the current fold.
                Contains filtered DataFrames with prediction errors and uncertainty bins
                for evaluation.

        Returns:
            JaccardBinResults: Results container with computed metrics including:
                - bin_jaccard: List of Jaccard similarities for each bin
                - bin_recall: List of recall values for each bin
                - bin_precision: List of precision values for each bin
                - target_metrics: Target-specific evaluation results
                - bins_targets_separated: Bin results separated by target

        Note:
            This method processes each target individually and aggregates results
            across all targets for comprehensive evaluation. Handles both combined
            and target-separated result generation based on configuration.
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

        Computes bin-wise Jaccard similarity, precision, and recall for predictions
        associated with a particular target index. This enables target-specific
        evaluation of uncertainty quantification quality.

        Args:
            fold_data (FoldData): Container with errors and bins data for the current fold.
            target_idx (int): Index of the target to process. Used to filter data
                for target-specific evaluation.

        Returns:
            Dict: Dictionary containing computed metrics with keys:
                - 'target_jaccard': Overall Jaccard similarity for this target
                - 'target_recall': Overall recall for this target
                - 'target_precision': Overall precision for this target
                - 'bin_jaccard': List of Jaccard values for each bin
                - 'bin_recall': List of recall values for each bin
                - 'bin_precision': List of precision values for each bin

        Note:
            This method processes data specific to one target, enabling detailed
            analysis of uncertainty quantification performance across different
            prediction targets or classes.
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

        # Get predicted bins
        pred_bin_keys, _ = DataProcessor.group_data_by_bins(errors_dict, bins_dict, self.current_num_bins_)
        pred_bin_keys = pred_bin_keys[::-1]  # Reverse for B5 to B1

        # Get ground truth quantiles
        _, _, gt_key_groups = QuantileCalculator.calculate_error_quantiles(
            errors_dict, self.config_.original_num_bins, self.config_.combine_middle_bins
        )

        # Calculate metrics for each bin
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

        return {
            "mean_jaccard": np.mean(bin_jaccard),
            "mean_recall": np.mean(bin_recall),
            "mean_precision": np.mean(bin_precision),
            "bin_jaccard": bin_jaccard,
            "bin_recall": bin_recall,
            "bin_precision": bin_precision,
        }

    def _aggregate_fold_results(self, model_key: str, fold_results: List[BinResults]):
        """
        Aggregate Jaccard evaluation results across all cross-validation folds.

        Combines fold-level Jaccard similarity, precision, and recall results into
        aggregated statistics for a specific model-uncertainty combination. Handles
        both main results and target-separated results based on configuration.

        Args:
            model_key (str): Identifier for the current model-uncertainty combination
                (format: "model_name uncertainty_type").
            fold_results (List[BinResults]): List of JaccardBinResults from all folds
                for the current model-uncertainty combination.

        Note:
            This method aggregates results by computing means and standard deviations
            across folds for both bin-wise and target-wise metrics. Results are
            stored in the container for final formatting.
        """
        # Cast to JaccardBinResults since we know that's what JaccardEvaluator produces
        jaccard_results = cast(List[JaccardBinResults], fold_results)

        fold_jaccard_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        fold_recall_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        fold_precision_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]

        fold_all_jaccard_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        fold_all_recall_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]
        fold_all_precision_bins: List[List[float]] = [[] for _ in range(self.current_num_bins_)]

        for result in jaccard_results:
            for bin_idx in range(len(result.mean_all_bins)):
                fold_jaccard_bins[bin_idx].append(result.mean_all_bins[bin_idx])
                fold_recall_bins[bin_idx].append(result.mean_all_bins_recall[bin_idx])
                fold_precision_bins[bin_idx].append(result.mean_all_bins_precision[bin_idx])

                fold_all_jaccard_bins[bin_idx].extend(result.all_bins[bin_idx])
                fold_all_recall_bins[bin_idx].extend(result.all_bins_recall[bin_idx])
                fold_all_precision_bins[bin_idx].extend(result.all_bins_precision[bin_idx])

        # Store results in container
        self.container_.add_main_result(model_key, fold_jaccard_bins)
        self.container_.add_target_separated_result(model_key, fold_all_jaccard_bins)

        # Store recall results
        self.container_.recall_results[model_key] = fold_recall_bins
        self.container_.recall_target_separated[model_key] = fold_all_recall_bins

        # Store precision results
        self.container_.precision_results[model_key] = fold_precision_bins
        self.container_.precision_target_separated[model_key] = fold_all_precision_bins

        # Store target separated data for foldwise and all
        for fold_idx in range(len(jaccard_results)):
            result = jaccard_results[fold_idx]
            for target_idx in range(len(result.all_bins_concat_targets_sep)):
                for bin_idx in range(self.current_num_bins_):
                    if target_idx < len(self.container_.target_sep_foldwise):
                        if model_key not in self.container_.target_sep_foldwise[target_idx]:
                            self.container_.target_sep_foldwise[target_idx][model_key] = [
                                [] for _ in range(self.current_num_bins_)
                            ]
                        self.container_.target_sep_foldwise[target_idx][model_key][bin_idx].extend(
                            result.all_bins_concat_targets_sep[target_idx][bin_idx]
                        )

                        if model_key not in self.container_.target_sep_all[target_idx]:
                            self.container_.target_sep_all[target_idx][model_key] = [
                                [] for _ in range(self.current_num_bins_)
                            ]
                        self.container_.target_sep_all[target_idx][model_key][bin_idx].extend(
                            result.all_bins_concat_targets_sep[target_idx][bin_idx]
                        )

    def _finalize_results(self) -> Dict:
        """
        Convert aggregated results container into final Jaccard evaluation output format.

        Transforms the results container into the expected API output format with
        properly organized Jaccard similarity, precision, and recall results for
        all evaluation categories.

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

        Note:
            This method maps the container's organized data structure to the specific
            result keys expected by the Jaccard evaluation API.
        """
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

    This function assesses how well predicted error bounds capture actual prediction
    errors across different uncertainty bins. It provides a comprehensive evaluation
    of bound reliability for uncertainty quantification in machine learning models.

    Args:
        estimated_bounds (Dict[str, pd.DataFrame]): Dictionary mapping model names to
            DataFrames containing estimated error bounds for each prediction.
        bin_predictions (Dict[str, pd.DataFrame]): Dictionary mapping model names to
            DataFrames containing bin predictions and evaluation data with columns
            for UIDs, target indices, errors, and uncertainty bins.
        uncertainty_pairs (List): List of uncertainty type pairs to evaluate.
            Each element should be a list/tuple containing uncertainty type names.
        num_bins (int): Number of uncertainty bins for evaluation. Controls the
            granularity of the bound accuracy assessment.
        targets (List[int]): List of target indices to include in the evaluation.
            Used to filter data and organize results by target.
        num_folds (int, optional): Number of cross-validation folds for evaluation.
            Defaults to 8. Higher values provide more robust estimates.
        combine_middle_bins (bool, optional): Whether to combine middle uncertainty
            bins for simplified three-bin analysis. Defaults to False.

    Returns:
        Dict: Comprehensive evaluation results dictionary containing:
            - Error bound accuracy statistics across bins and targets
            - Target-separated results for detailed analysis
            - Fold-wise results for cross-validation assessment
            - Bound coverage percentages and reliability metrics

    Example:
        >>> bounds = {'model1': bounds_df}
        >>> predictions = {'model1': predictions_df}
        >>> results = evaluate_bounds(
        ...     bounds, predictions,
        ...     uncertainty_pairs=[['epistemic']],
        ...     num_bins=5, targets=[0, 1, 2]
        ... )
        >>> print(results['Error Bounds All']['model1_epistemic'])

    Note:
        This function complements Jaccard-based evaluation by focusing on the
        accuracy of predicted confidence intervals rather than bin overlap.
    """

    if combine_middle_bins:
        num_bins = 3

    # Initialize results dicts
    all_bound_percents = {}
    all_bound_percents_notargetsep = {}

    all_concat_errorbound_bins_target_sep_foldwise = [{} for x in range(len(targets))]  # type: List[Dict]
    all_concat_errorbound_bins_target_sep_all = [{} for x in range(len(targets))]  # type: List[Dict]

    # Loop over combinations of models (model) and uncertainty types (uncert_pair)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        error_bounds = estimated_bounds[model + " Error Bounds"]

        for uncert_pair in uncertainty_pairs:
            uncertainty_type = uncert_pair[0]

            fold_learned_bounds_mean_targets = []
            fold_learned_bounds_mean_bins = [[] for x in range(num_bins)]  # type: List[List]
            fold_learned_bounds_bins_targetsnotsep = [[] for x in range(num_bins)]  # type: List[List]
            fold_all_bins_concat_targets_sep_foldwise = [
                [[] for y in range(num_bins)] for x in range(len(targets))
            ]  # type: List[List]
            fold_all_bins_concat_targets_sep_all = [
                [[] for y in range(num_bins)] for x in range(len(targets))
            ]  # type: List[List]

            for fold in range(num_folds):
                # Get the ids for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "target_idx", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "target_idx", uncertainty_type + " Uncertainty bins"]
                ]
                fold_bounds = strip_for_bound(
                    error_bounds[error_bounds["fold"] == fold][uncertainty_type + " Uncertainty bounds"].values
                )

                return_dict = bin_wise_bound_eval(
                    fold_bounds, fold_errors, fold_bins, targets, uncertainty_type, num_bins=num_bins
                )
                fold_learned_bounds_mean_targets.append(return_dict["mean all targets"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_learned_bounds_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_learned_bounds_bins_targetsnotsep[idx_bin] = (
                        fold_learned_bounds_bins_targetsnotsep[idx_bin] + return_dict["mean all"][idx_bin]
                    )

                    for target_idx in range(len(targets)):
                        fold_all_bins_concat_targets_sep_foldwise[target_idx][idx_bin] = (
                            fold_all_bins_concat_targets_sep_foldwise[target_idx][idx_bin]
                            + return_dict["all bins concatenated targets seperated"][target_idx][idx_bin]
                        )
                        combined = (
                            fold_all_bins_concat_targets_sep_all[target_idx][idx_bin]
                            + return_dict["all bins concatenated targets seperated"][target_idx][idx_bin]
                        )

                        fold_all_bins_concat_targets_sep_all[target_idx][idx_bin] = combined

            # Reverses order so they are worst to best i.e. B5 -> B1
            all_bound_percents[model + " " + uncertainty_type] = fold_learned_bounds_mean_bins[::-1]
            all_bound_percents_notargetsep[model + " " + uncertainty_type] = fold_learned_bounds_bins_targetsnotsep[
                ::-1
            ]

            for target_idx in range(len(all_concat_errorbound_bins_target_sep_foldwise)):
                all_concat_errorbound_bins_target_sep_foldwise[target_idx][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_targets_sep_foldwise[target_idx]
                all_concat_errorbound_bins_target_sep_all[target_idx][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_targets_sep_all[target_idx]

    return {
        "Error Bounds All": all_bound_percents,
        "all_bound_percents_notargetsep": all_bound_percents_notargetsep,
        "all errorbound concat bins targets sep foldwise": all_concat_errorbound_bins_target_sep_foldwise,
        "all errorbound concat bins targets sep all": all_concat_errorbound_bins_target_sep_all,
    }


# Convenience functions for backward compatibility
def evaluate_jaccard(bin_predictions, uncertainty_pairs, num_bins, targets, num_folds=8, combine_middle_bins=False):
    """
    Evaluate uncertainty estimation's ability to predict true error quantiles using Jaccard metrics.

    This is a convenience function that uses the new JaccardEvaluator class
    while maintaining backward compatibility with the original function signature.

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


def bin_wise_bound_eval(
    fold_bounds_all_targets: list,
    fold_errors: pd.DataFrame,
    fold_bins: pd.DataFrame,
    targets: list,
    uncertainty_type: str,
    num_bins: int = 5,
) -> dict:
    """
    Helper function for `evaluate_bounds`. Evaluates the accuracy of estimated error bounds for each quantile bin
    for a given uncertainty type, over a single fold and for multiple targets.

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
              - 'mean all bins': A list of mean accuracy values for each quantile bin (all targets included).
              - 'mean all': A list of accuracy values for each quantile bin and target, weighted by # targets in each bin.
              - 'all bins concatenated targets separated': A list of accuracy values for each quantile bin, concatenated
               for each target separately.

    Example:
        >>> bin_wise_bound_eval(fold_bounds_all_targets, fold_errors, fold_bins, [0,1], 'S-MHA', num_bins=5)
    """
    all_target_perc = []
    all_qs_perc: List[List[float]] = [[] for x in range(num_bins)]  #
    all_qs_size: List[List[float]] = [[] for x in range(num_bins)]

    all_qs_errorbound_concat_targets_sep: List[List[List[float]]] = [
        [[] for y in range(num_bins)] for x in range(len(targets))
    ]

    for i_ti, target_idx in enumerate(targets):
        true_errors_ti = fold_errors[(fold_errors["target_idx"] == target_idx)][["uid", uncertainty_type + " Error"]]
        pred_bins_ti = fold_bins[(fold_errors["target_idx"] == target_idx)][
            ["uid", uncertainty_type + " Uncertainty bins"]
        ]

        # Zip to dictionary
        true_errors_ti = dict(zip(true_errors_ti.uid, true_errors_ti[uncertainty_type + " Error"]))
        pred_bins_ti = dict(zip(pred_bins_ti.uid, pred_bins_ti[uncertainty_type + " Uncertainty bins"]))

        # The error bounds are from B1 -> B5 i.e. best quantile of predictions to worst quantile of predictions
        fold_bounds = fold_bounds_all_targets[i_ti]

        # For each bin, see what % of targets are between the error bounds.
        # If bin=0 then lower bound = 0, if bin=Q then no upper bound
        # Keep track of #samples in each bin for weighted mean.

        # turn dictionary of predicted bins into [[num_bins]] array
        pred_bins_keys = []
        pred_bins_errors = []
        for i in range(num_bins):
            inner_list_bin = list([key for key, val in pred_bins_ti.items() if str(i) == str(val)])
            inner_list_errors = []

            for id_ in inner_list_bin:
                inner_list_errors.append(list([val for key, val in true_errors_ti.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list_bin)

        bins_acc = []
        bins_sizes = []
        for q in range((num_bins)):
            inner_bin_correct = 0

            inbin_errors = pred_bins_errors[q]

            for error in inbin_errors:
                if q == 0:
                    lower = 0
                    upper = fold_bounds[q]

                    if error <= upper and error > lower:
                        inner_bin_correct += 1

                elif q < (num_bins) - 1:
                    lower = fold_bounds[q - 1]
                    upper = fold_bounds[q]

                    if error <= upper and error > lower:
                        inner_bin_correct += 1

                else:
                    lower = fold_bounds[q - 1]
                    upper = 999999999999999999999999999999

                    if error > lower:
                        inner_bin_correct += 1

            if inner_bin_correct == 0:
                accuracy_bin = 0.0
            elif len(inbin_errors) == 0:
                accuracy_bin = 1.0
            else:
                accuracy_bin = inner_bin_correct / len(inbin_errors)
            bins_sizes.append(len(inbin_errors))
            bins_acc.append(accuracy_bin)

            all_qs_perc[q].append(accuracy_bin)
            all_qs_size[q].append(len(inbin_errors))
            all_qs_errorbound_concat_targets_sep[i_ti][q].append(accuracy_bin)

        # Weighted average over all bins
        weighted_mean_ti = 0.0
        total_weights = 0.0
        for l_idx in range(len(bins_sizes)):
            bin_acc = bins_acc[l_idx]
            bin_size = bins_sizes[l_idx]
            weighted_mean_ti += bin_acc * bin_size
            total_weights += bin_size
        weighted_ave = weighted_mean_ti / total_weights
        all_target_perc.append(weighted_ave)

    # Weighted average for each of the quantile bins.
    weighted_ave_binwise = []
    for binidx in range(len(all_qs_perc)):
        bin_accs = all_qs_perc[binidx]
        bin_asizes = all_qs_size[binidx]

        weighted_mean_bin = 0.0
        total_weights_bin = 0.0
        for l_idx in range(len(bin_accs)):
            b_acc = bin_accs[l_idx]
            b_siz = bin_asizes[l_idx]
            weighted_mean_bin += b_acc * b_siz
            total_weights_bin += b_siz

        # Avoid div by 0
        if weighted_mean_bin == 0 or total_weights_bin == 0:
            weighted_ave_bin = 0.0
        else:
            weighted_ave_bin = weighted_mean_bin / total_weights_bin
        weighted_ave_binwise.append(weighted_ave_bin)

    # No weighted average, just normal average
    normal_ave_bin_wise = []
    for binidx in range(len(all_qs_perc)):
        bin_accs = all_qs_perc[binidx]
        normal_ave_bin_wise.append(np.mean(bin_accs))

    return {
        "mean all targets": np.mean(all_target_perc),
        "mean all bins": weighted_ave_binwise,
        "mean all": all_qs_perc,
        "all bins concatenated targets seperated": all_qs_errorbound_concat_targets_sep,
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
    Evaluate uncertainty estimation's mean error of each bin.
    For each bin, we calculate the mean localization error for each target and for all targets.
    We calculate the mean error for each dictionary in the bin_predictions dict. For each bin, we calculate: a) the mean
    and std over all folds and all targets b) the mean and std for each target over all folds.

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
                "all mean error bins nosep":  For every fold, the mean error for each bin. All targets are combined in the same list.
                "all mean error bins targets sep":   For every fold, the mean error for each bin. Each target is in a separate list.
                "all error concat bins targets nosep":  For every fold, every error value in a list. Each target is in the same list. The list is flattened for all the folds.
                "all error concat bins targets sep foldwise":  For every fold, every error value in a list. Each target is in a separate list. Each list has a list of results by fold.
                "all error concat bins targets sep all": For every fold, every error value in a list. Each target is in a separate list. The list is flattened for all the folds.

    """
    # If we are combining the middle bins, we only have the 2 edge bins and the middle bins are combined into 1 bin.
    if combine_middle_bins:
        num_bins = 3

    # initialize empty dicts
    all_mean_error_bins = {}
    all_mean_error_bins_targets_sep = {}
    all_concat_error_bins_target_sep_foldwise: List[Dict] = [{} for x in range(len(targets))]
    all_concat_error_bins_target_sep_all: List[Dict] = [{} for x in range(len(targets))]

    all_concat_error_bins_target_nosep = {}
    # Loop over models (model) and uncertainty methods (uncert_pair)
    for i, (model, data_structs) in enumerate(bin_predictions.items()):
        for uncert_pair in uncertainty_pairs:  # uncert_pair = [pair name, error name , uncertainty name]
            uncertainty_type = uncert_pair[0]

            # Initialize lists to store fold-wise results
            fold_mean_targets = []
            fold_mean_bins: List[List[float]] = [[] for x in range(num_bins)]
            fold_all_bins: List[List[float]] = [[] for x in range(num_bins)]
            fold_all_bins_concat_targets_sep_foldwise: List[List[List[float]]] = [
                [[] for y in range(num_bins)] for x in range(len(targets))
            ]
            fold_all_bins_concat_targets_sep_all: List[List[List[float]]] = [
                [[] for y in range(num_bins)] for x in range(len(targets))
            ]

            fold_all_bins_concat_targets_nosep: List[List[float]] = [[] for x in range(num_bins)]

            for fold in range(num_folds):
                # Get the errors and predicted bins for this fold
                fold_errors = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "target_idx", uncertainty_type + " Error"]
                ]
                fold_bins = data_structs[(data_structs["Testing Fold"] == fold)][
                    ["uid", "target_idx", uncertainty_type + " Uncertainty bins"]
                ]

                return_dict = bin_wise_errors(
                    fold_errors,
                    fold_bins,
                    num_bins,
                    targets,
                    uncertainty_type,
                    error_scaling_factor=error_scaling_factor,
                )
                fold_mean_targets.append(return_dict["mean all targets"])

                for idx_bin in range(len(return_dict["mean all bins"])):
                    fold_mean_bins[idx_bin].append(return_dict["mean all bins"][idx_bin])
                    fold_all_bins[idx_bin] = fold_all_bins[idx_bin] + return_dict["all bins"][idx_bin]

                    concat_no_sep = [x[idx_bin] for x in return_dict["all bins concatenated targets seperated"]]

                    flattened_concat_no_sep = [x for sublist in concat_no_sep for x in sublist]
                    flattened_concat_no_sep = [x for sublist in flattened_concat_no_sep for x in sublist]

                    fold_all_bins_concat_targets_nosep[idx_bin] = (
                        fold_all_bins_concat_targets_nosep[idx_bin] + flattened_concat_no_sep
                    )

                    for target_idx in range(len(targets)):
                        fold_all_bins_concat_targets_sep_foldwise[target_idx][idx_bin] = (
                            fold_all_bins_concat_targets_sep_foldwise[target_idx][idx_bin]
                            + return_dict["all bins concatenated targets seperated"][target_idx][idx_bin]
                        )

                        if return_dict["all bins concatenated targets seperated"][target_idx][idx_bin] != []:
                            combined = (
                                fold_all_bins_concat_targets_sep_all[target_idx][idx_bin]
                                + return_dict["all bins concatenated targets seperated"][target_idx][idx_bin][0]
                            )
                        else:
                            combined = fold_all_bins_concat_targets_sep_all[target_idx][idx_bin]

                        fold_all_bins_concat_targets_sep_all[target_idx][idx_bin] = combined

            # reverse orderings
            fold_mean_bins = fold_mean_bins[::-1]
            fold_all_bins = fold_all_bins[::-1]
            fold_all_bins_concat_targets_nosep = fold_all_bins_concat_targets_nosep[::-1]
            fold_all_bins_concat_targets_sep_foldwise = [x[::-1] for x in fold_all_bins_concat_targets_sep_foldwise]
            fold_all_bins_concat_targets_sep_all = [x[::-1] for x in fold_all_bins_concat_targets_sep_all]

            all_mean_error_bins[model + " " + uncertainty_type] = fold_mean_bins
            all_mean_error_bins_targets_sep[model + " " + uncertainty_type] = fold_all_bins

            all_concat_error_bins_target_nosep[model + " " + uncertainty_type] = fold_all_bins_concat_targets_nosep

            for target_idx in range(len(fold_all_bins_concat_targets_sep_foldwise)):
                all_concat_error_bins_target_sep_foldwise[target_idx][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_targets_sep_foldwise[target_idx]
                all_concat_error_bins_target_sep_all[target_idx][
                    model + " " + uncertainty_type
                ] = fold_all_bins_concat_targets_sep_all[target_idx]

    return {
        "all mean error bins nosep": all_mean_error_bins,
        "all mean error bins targets sep": all_mean_error_bins_targets_sep,
        "all error concat bins targets nosep": all_concat_error_bins_target_nosep,
        "all error concat bins targets sep foldwise": all_concat_error_bins_target_sep_foldwise,
        "all error concat bins targets sep all": all_concat_error_bins_target_sep_all,
    }


def bin_wise_errors(fold_errors, fold_bins, num_bins, targets, uncertainty_key, error_scaling_factor):
    """
    Helper function for get_mean_errors. Calculates the mean error for each bin and for each target.

    Args:
        fold_errors (Pandas Dataframe): Pandas Dataframe of errors for this fold.
        fold_bins (Pandas Dataframe): Pandas Dataframe of predicted quantile bins for this fold.
        num_bins (int): Number of quantile bins,
        targets (list) list of targets to measure uncertainty estimation,
        uncertainty_key (string): Name of uncertainty type to calculate accuracy for,


    Returns:
        [Dict]: Dict with mean error statistics.
    """

    all_target_error = []
    all_qs_error = [[] for x in range(num_bins)]
    all_qs_error_concat_targets_sep = [[[] for y in range(num_bins)] for x in range(len(targets))]

    for i, target_idx in enumerate(targets):
        true_errors_ti = fold_errors[(fold_errors["target_idx"] == target_idx)][["uid", uncertainty_key + " Error"]]
        pred_bins_ti = fold_bins[(fold_errors["target_idx"] == target_idx)][
            ["uid", uncertainty_key + " Uncertainty bins"]
        ]

        # Zip to dictionary
        true_errors_ti = dict(
            zip(true_errors_ti.uid, true_errors_ti[uncertainty_key + " Error"] * error_scaling_factor)
        )
        pred_bins_ti = dict(zip(pred_bins_ti.uid, pred_bins_ti[uncertainty_key + " Uncertainty bins"]))

        pred_bins_keys = []
        pred_bins_errors = []

        # This is saving them from best quantile of predictions to worst quantile of predictions in terms of uncertainty
        for j in range(num_bins):
            inner_list = list([key for key, val in pred_bins_ti.items() if str(j) == str(val)])
            inner_list_errors = []

            for id_ in inner_list:
                inner_list_errors.append(list([val for key, val in true_errors_ti.items() if str(key) == str(id_)])[0])

            pred_bins_errors.append(inner_list_errors)
            pred_bins_keys.append(inner_list)

        # Now for each bin, get the mean error
        inner_errors = []
        for bin in range(num_bins):
            # pred_b_keys = pred_bins_keys[bin]
            pred_b_errors = pred_bins_errors[bin]

            # test for empty bin, it would've created a mean_error==nan , so don't add it!
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
        "all bins concatenated targets seperated": all_qs_error_concat_targets_sep,
    }

# =============================================================================
# Author: Lawrence Schobs, lawrenceschobs@gmail.com
#         Zhongwei Ji, jizhongwei1999@outlook.com
#         Charles Anjah, cmanjahart@gmail.com
# =============================================================================

"""Column names and result keys shared by the evaluation modules.

These constants name the DataFrame columns consumed by the evaluators and the keys of the result
dictionaries they return. They live here rather than in
:mod:`kale.evaluate.uncertainty_metrics` so that code needing only the names does not import that
module's heavy dependencies (pandas, torch and torchmetrics).

Both classes are re-exported from :mod:`kale.evaluate.uncertainty_metrics`, so existing imports
continue to work.

Example:
    .. code-block:: pycon

        >>> from kale.evaluate.constants import ColumnNames, ResultKeys
        >>> ColumnNames.UID
        'uid'
        >>> ResultKeys.MEAN_ALL_TARGETS
        'mean all targets'
"""


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

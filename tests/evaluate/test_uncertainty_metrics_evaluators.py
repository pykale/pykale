import pandas as pd
import pytest

from kale.evaluate.uncertainty_metrics import bin_wise_bound_eval, bin_wise_errors, evaluate_bounds, get_mean_errors


@pytest.mark.parametrize("metric", ["errors", "bounds"])
def test_evaluators_reject_missing_error(metric):
    """Both metrics raise an error for unmatched prediction UIDs."""
    fold_errors = pd.DataFrame({"uid": ["a"], "Target Index": 0, "U Error": [1.0]})
    fold_bins = pd.DataFrame({"uid": ["a", "b"], "Target Index": 0, "U Uncertainty bins": 0})

    with pytest.raises(ValueError, match="No error found for uid 'b'"):
        if metric == "errors":
            bin_wise_errors(fold_errors, fold_bins, 2, [0], "U", error_scaling_factor=1.0)
        else:
            bin_wise_bound_eval([[5.0]], fold_errors, fold_bins, [0], "U", num_bins=2)


@pytest.fixture
def evaluation_data():
    """Two folds and targets with unequal bin sizes and an empty middle bin."""
    errors = [
        (0, 0, 0, [1.0, 3.0]),
        (0, 0, 2, [7.0]),
        (0, 1, 0, [5.0]),
        (0, 1, 2, [9.0, 11.0]),
        (1, 0, 0, [2.0]),
        (1, 0, 2, [6.0, 10.0]),
        (1, 1, 0, [4.0, 6.0]),
        (1, 1, 2, [8.0]),
    ]
    rows = [
        (fold, target, f"{bin_idx}-{sample}", error, bin_idx)
        for fold, target, bin_idx, values in errors
        for sample, error in enumerate(values)
    ]
    predictions = pd.DataFrame(rows, columns=["Testing Fold", "Target Index", "uid", "U Error", "U Uncertainty bins"])
    bounds = pd.DataFrame({"fold": [0, 0, 1, 1], "U Uncertainty bounds": ["[4.0, 8.0]"] * 4})
    return {"model": predictions}, {"model Error Bounds": bounds}


@pytest.mark.parametrize("combine_middle_bins", [False, True])
def test_mean_error_results(evaluation_data, combine_middle_bins):
    """Check every result field against hand-calculated fold and target aggregates."""
    predictions, _ = evaluation_data
    result = get_mean_errors(
        predictions,
        [["U"]],
        10 if combine_middle_bins else 3,
        [0, 1],
        num_folds=2,
        combine_middle_bins=combine_middle_bins,
    )

    assert result == {
        "all_mean_error_bins_nosep": {"model U": [[8.5, 8.0], [None, None], [3.5, 3.5]]},
        "all mean error bins targets sep": {"model U": [[7.0, 10.0, 8.0, 8.0], [], [2.0, 5.0, 2.0, 5.0]]},
        "all_error_concat_bins_targets_nosep": {
            "model U": [[7.0, 9.0, 11.0, 6.0, 10.0, 8.0], [], [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]]
        },
        "all error concat bins targets sep foldwise": [
            {"model U": [[[7.0], [6.0, 10.0]], [], [[1.0, 3.0], [2.0]]]},
            {"model U": [[[9.0, 11.0], [8.0]], [], [[5.0], [4.0, 6.0]]]},
        ],
        "all_error_concat_bins_targets_sep_all": [
            {"model U": [[7.0, 6.0, 10.0], [], [1.0, 3.0, 2.0]]},
            {"model U": [[9.0, 11.0, 8.0], [], [5.0, 4.0, 6.0]]},
        ],
    }


def test_mean_error_scaling(evaluation_data):
    """The public wrapper applies scaling to both means and individual errors."""
    predictions, _ = evaluation_data
    result = get_mean_errors(predictions, [["U"]], 3, [0, 1], num_folds=2, error_scaling_factor=2.0)

    assert result["all_mean_error_bins_nosep"] == {"model U": [[17.0, 16.0], [None, None], [7.0, 7.0]]}
    assert result["all_error_concat_bins_targets_nosep"] == {
        "model U": [[14.0, 18.0, 22.0, 12.0, 20.0, 16.0], [], [2.0, 6.0, 10.0, 4.0, 8.0, 12.0]]
    }


@pytest.mark.parametrize("combine_middle_bins", [False, True])
def test_bound_results(evaluation_data, combine_middle_bins):
    """Check weighted accuracy, empty bins, and target bin ordering."""
    predictions, bounds = evaluation_data
    result = evaluate_bounds(
        bounds,
        predictions,
        [["U"]],
        10 if combine_middle_bins else 3,
        [0, 1],
        num_folds=2,
        combine_middle_bins=combine_middle_bins,
    )

    targets = [
        {"model U": [[1.0, 1.0], [1.0, 1.0], [0.0, 0.5]]},
        {"model U": [[0.0, 0.5], [1.0, 1.0], [1.0, 0.0]]},
    ]
    assert result == {
        "error_bounds_all": {"model U": [[2 / 3, 1 / 3], [0.0, 0.0], [2 / 3, 2 / 3]]},
        "all_bound_percents_notargetsep": {
            "model U": [[0.0, 1.0, 0.5, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.5]]
        },
        "all errorbound concat bins targets sep foldwise": targets,
        "all_errorbound_concat_bins_targets_sep_all": targets,
    }

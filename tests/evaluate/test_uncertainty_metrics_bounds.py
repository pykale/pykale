import pandas as pd
import pytest

from kale.evaluate.uncertainty_metrics import bin_wise_bound_eval, ColumnNames


class TestBinWiseBoundEvalScoring:
    """Scoring behaviour of ``bin_wise_bound_eval`` for empty and open-ended bins (#549, #550)."""

    UNCERTAINTY = "S-MHA"

    @classmethod
    def _frames(cls, errors, bins):
        """Build matching error and bin frames for two predictions, using the column-name constants."""
        errors_df = pd.DataFrame(
            {
                ColumnNames.UID: ["u0", "u1"],
                ColumnNames.TARGET_IDX: [0, 0],
                cls.UNCERTAINTY + ColumnNames.ERROR_SUFFIX: errors,
            }
        )
        bins_df = pd.DataFrame(
            {
                ColumnNames.UID: ["u0", "u1"],
                ColumnNames.TARGET_IDX: [0, 0],
                cls.UNCERTAINTY + ColumnNames.UNCERTAINTY_BINS_SUFFIX: bins,
            }
        )
        return errors_df, bins_df

    def test_empty_bin_scores_one(self):
        """An empty quantile bin is scored 1.0, not 0.0 (issue #549).

        Both samples are placed in bin 0, leaving bin 1 empty. The empty bin must take the
        dedicated empty-bin score. Because an empty bin has size 0, it must not affect the
        size-weighted means.
        """
        errors_df, bins_df = self._frames(errors=[1.0, 3.0], bins=[0, 0])

        result = bin_wise_bound_eval(
            [[2.0]], errors_df, bins_df, targets=[0], uncertainty_type=self.UNCERTAINTY, num_bins=2
        )

        # Bin 1 is empty and is scored 1.0.
        assert result["mean all"][1] == [1.0]
        # Bin 0 holds errors 1.0 (within bound 2.0) and 3.0 (outside) -> 0.5.
        assert result["mean all"][0] == [0.5]
        # Size-weighted means are unaffected by the empty bin (its weight is 0).
        assert result["mean all targets"] == pytest.approx(0.5)
        assert result["mean all bins"] == pytest.approx([0.5, 0.0])

    def test_last_bin_has_no_upper_bound(self):
        """The final bin is unbounded above, so an arbitrarily large error is still correct (#550)."""
        errors_df, bins_df = self._frames(errors=[1.0, 1e31], bins=[0, 1])

        result = bin_wise_bound_eval(
            [[2.0]], errors_df, bins_df, targets=[0], uncertainty_type=self.UNCERTAINTY, num_bins=2
        )

        # 1.0 falls inside bound 2.0; 1e31 exceeds the lower bound of the open-ended last bin.
        assert result["mean all targets"] == pytest.approx(1.0)
        assert result["mean all"][1] == [1.0]

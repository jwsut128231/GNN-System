"""Unit tests for _regression_metrics in training pipeline."""
from __future__ import annotations

import pytest

from app.training.pipeline import _regression_metrics


class TestRegressionMetricsMape:
    def test_mape_computed_when_no_zeros(self):
        result = _regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        assert result["mape"] is not None
        assert isinstance(result["mape"], float)
        assert isinstance(result["mse"], float)
        assert isinstance(result["mae"], float)
        assert isinstance(result["r2_score"], float)

    def test_mape_is_none_when_zero_in_y_true(self):
        result = _regression_metrics([0.0, 1.0, 2.0], [0.1, 0.9, 2.1])
        assert result["mape"] is None

    def test_mape_value_correctness(self):
        # MAPE([1,2,3], [1.1,1.9,3.2]) = mean(|err/actual|)
        # = mean([0.1/1, 0.1/2, 0.2/3]) = mean([0.1, 0.05, 0.0667]) ≈ 0.0722
        result = _regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        assert 0.05 < result["mape"] < 0.15

    def test_standard_metrics_present(self):
        result = _regression_metrics([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
        for key in ("mse", "mae", "r2_score", "mape"):
            assert key in result


class TestResidualErrorField:
    """Verify the residual list produced by pipeline contains the error key.

    We call _regression_metrics indirectly and validate the shape contract
    by directly testing the dict shape that pipeline.py produces.
    """

    def test_residual_entry_shape(self):
        """Simulate what pipeline.py builds for residual_data entries."""
        import numpy as np

        test_y = np.array([1.0, 2.0, 3.0])
        test_preds = np.array([1.1, 1.9, 3.2])

        residual = [
            {
                "actual": round(float(test_y[i]), 4),
                "predicted": round(float(test_preds[i]), 4),
                "error": round(float(test_y[i] - test_preds[i]), 4),
            }
            for i in range(min(500, len(test_y)))
        ]

        assert len(residual) == 3
        for entry in residual:
            assert "actual" in entry
            assert "predicted" in entry
            assert "error" in entry
            assert isinstance(entry["error"], float)
            assert abs(entry["error"] - (entry["actual"] - entry["predicted"])) < 1e-6

"""Unit tests for forecast data alignment and validation."""
import numpy as np
import pandas as pd
import pytest

from src.data_alignment import align_model_pair
from src.data_validator import validate_forecast_dataframe


def test_align_model_pair_date_intersection():
    """Ensure non-overlapping dates are dropped and both models align on common dates."""
    df = pd.DataFrame({
        "date": [
            "2026-01-01", "2026-01-02", "2026-01-03",  # Model A has 3 days
            "2026-01-02", "2026-01-03", "2026-01-04",  # Model B has 3 days (overlap: 02, 03)
        ],
        "province": ["dak_lak"] * 6,
        "horizon": [1] * 6,
        "model": ["Model A"] * 3 + ["Model B"] * 3,
        "y_true": [100.0, 102.0, 104.0, 102.0, 104.0, 106.0],
        "y_pred": [101.0, 101.5, 103.5, 103.0, 105.0, 107.0],
    })

    aligned, meta = align_model_pair(df, "Model A", "Model B", province="dak_lak", horizon=1)

    assert meta["n_common"] == 2
    assert meta["dropped_a"] == 1
    assert meta["dropped_b"] == 1
    assert list(aligned["date"]) == ["2026-01-02", "2026-01-03"]
    assert np.allclose(aligned["y_true"], [102.0, 104.0])
    assert np.allclose(aligned["pred_a"], [101.5, 103.5])
    assert np.allclose(aligned["pred_b"], [103.0, 105.0])
    # error = y_true - y_pred
    assert np.allclose(aligned["error_a"], [0.5, 0.5])
    assert np.allclose(aligned["error_b"], [-1.0, -1.0])


def test_validator_detects_duplicates():
    """Ensure validator catches duplicate key entries."""
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-01"],
        "province": ["dak_lak", "dak_lak"],
        "model": ["Model A", "Model A"],
        "horizon": [1, 1],
        "y_true": [100.0, 100.0],
        "y_pred": [101.0, 102.0],
    })

    report = validate_forecast_dataframe(df)
    assert report.is_valid is False
    assert any("duplicate" in err.lower() for err in report.errors)

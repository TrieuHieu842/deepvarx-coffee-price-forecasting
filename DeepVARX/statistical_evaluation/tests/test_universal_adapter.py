"""Unit tests for Universal Forecast Adapter."""
import numpy as np
import pandas as pd
import pytest

from src.universal_adapter import UniversalForecastAdapter


def test_adapter_identify_formats():
    """Verify detection of long, wide, and summary formats."""
    # Summary metrics
    df_summary = pd.DataFrame({
        "Model": ["VAR", "LSTM"],
        "RMSE": [1500.0, 1400.0],
        "MAE": [1000.0, 950.0],
        "MAPE (%)": [1.5, 1.4],
    })
    assert UniversalForecastAdapter.identify_format(df_summary) == "summary_metrics"

    # Long format
    df_long = pd.DataFrame({
        "model": ["VAR", "VAR"],
        "target_date": ["2026-01-01", "2026-01-02"],
        "actual": [100.0, 102.0],
        "predicted": [101.0, 103.0],
    })
    assert UniversalForecastAdapter.identify_format(df_long) == "long_format"

    # Wide format
    df_wide = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "actual": [100.0, 102.0],
        "pred_var": [101.0, 103.0],
        "pred_lstm": [100.5, 102.5],
    })
    assert UniversalForecastAdapter.identify_format(df_wide) == "wide_format"


def test_adapter_normalize_long_format_heterogeneous_columns():
    """Verify normalization of diverse column naming into standard schema."""
    df_messy = pd.DataFrame({
        "FORECASTER": ["DeepVARX 3"] * 3,
        "location": ["lam_dong"] * 3,
        "forecast_date": ["2026-05-01", "2026-05-02", "2026-05-03"],
        "price_vnd_kg": [125000.0, 126000.0, 125500.0],
        "y_hat": [124800.0, 126100.0, 125400.0],
        "sigma": [1500.0, 1520.0, 1510.0],
    })

    df_norm = UniversalForecastAdapter.normalize_long_format(df_messy)

    expected_cols = ["model", "province", "forecast_origin", "target_date", "horizon", "actual", "predicted"]
    for c in expected_cols:
        assert c in df_norm.columns

    assert df_norm["model"].iloc[0] == "DeepVARX 3"
    assert df_norm["province"].iloc[0] == "lam_dong"
    assert df_norm["actual"].iloc[0] == 125000.0
    assert df_norm["predicted"].iloc[0] == 124800.0
    assert df_norm["pred_std"].iloc[0] == 1500.0
    assert df_norm["horizon"].iloc[0] == 1


def test_adapter_normalize_wide_format():
    """Verify pivoting wide-format multi-model table into standard schema."""
    df_wide = pd.DataFrame({
        "date": ["2026-02-01", "2026-02-02"],
        "province": ["dak_lak", "dak_lak"],
        "actual": [100000.0, 101000.0],
        "pred_var": [99800.0, 101200.0],
        "pred_lstm": [100100.0, 100900.0],
    })

    df_norm = UniversalForecastAdapter.normalize_wide_format(df_wide)

    assert len(df_norm) == 4  # 2 dates x 2 models
    assert set(df_norm["model"].unique()) == {"var", "lstm"}
    assert "target_date" in df_norm.columns
    assert "actual" in df_norm.columns
    assert "predicted" in df_norm.columns

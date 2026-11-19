"""
Unit tests for modules/result_manager.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from modules.result_manager import (
    save_model_run,
    load_all_saved_results,
    init_results_storage_if_empty,
    RESULTS_DIR
)


def test_init_and_load_saved_results():
    init_results_storage_if_empty()
    assert RESULTS_DIR.exists()
    
    results = load_all_saved_results()
    assert len(results) >= 5
    for m in ["VAR", "VARX", "LSTM", "DeepVAR", "DeepVARX"]:
        assert m in results
        data = results[m]
        assert "pred" in data
        assert isinstance(data["pred"], pd.DataFrame)
        assert "y_true" in data["pred"].columns
        assert "y_pred" in data["pred"].columns
        assert "metrics" in data
        assert "RMSE" in data["metrics"]


def test_save_model_run_updates_files():
    # Mock model output
    dates = pd.date_range("2025-01-01", periods=10)
    mock_pred = pd.DataFrame({
        "date": dates,
        "province": ["Đắk Lắk"] * 10,
        "y_true": np.array([100000.0] * 10),
        "y_pred": np.array([100500.0] * 10),
        "y_prev": np.array([99800.0] * 10),
    })
    mock_by_prov = pd.DataFrame([{
        "province": "Đắk Lắk",
        "RMSE": 500.0,
        "MAE": 500.0,
        "MAPE (%)": 0.5,
        "sMAPE (%)": 0.5,
        "R²": 0.99,
        "Đúng hướng (%)": 80.0
    }])
    mock_out = {
        "pred": mock_pred,
        "by_prov": mock_by_prov,
        "metrics": {"RMSE": 500.0, "MAE": 500.0, "MAPE (%)": 0.5, "sMAPE (%)": 0.5, "R²": 0.99, "Đúng hướng (%)": 80.0},
        "seconds": 5.2,
        "log": "Test run",
    }
    
    saved = save_model_run("MOCK_TEST_MODEL", mock_out, alloc_info={"scope": "Wide"})
    assert "predictions" in saved
    assert Path(saved["predictions"]).exists()
    assert "by_province" in saved
    assert Path(saved["by_province"]).exists()
    assert "metrics_json" in saved
    assert Path(saved["metrics_json"]).exists()
    
    # Clean up mock files
    for p in saved.values():
        Path(p).unlink(missing_ok=True)
    
    # Remove mock from models_summary.csv
    sum_p = RESULTS_DIR / "models_summary.csv"
    if sum_p.exists():
        df_sum = pd.read_csv(sum_p)
        df_sum = df_sum[df_sum["Mô hình"] != "MOCK_TEST_MODEL"]
        df_sum.to_csv(sum_p, index=False)

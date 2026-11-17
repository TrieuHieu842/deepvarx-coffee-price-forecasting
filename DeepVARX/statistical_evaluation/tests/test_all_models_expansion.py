"""Integration test verifying expansion across all 17 models, data integrity, and DM results."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_CSV = PROJ_ROOT / "statistical_evaluation" / "data" / "input" / "all_models_forecasts_test_542days.csv"
SUMMARY_CSV = PROJ_ROOT / "statistical_evaluation" / "data" / "output" / "dm_summary_results.csv"
DISCOVERY_CSV = PROJ_ROOT / "statistical_evaluation" / "data" / "output" / "model_discovery_report.csv"


def test_input_dataset_completeness_17_models():
    """Ensure all 17 models are present with exactly 542 test days and 3 provinces."""
    assert INPUT_CSV.exists(), f"Input dataset not found: {INPUT_CSV}"
    df = pd.read_csv(INPUT_CSV)

    expected_models = {
        "Naive (Random Walk)",
        "VAR(1)", "VAR(2)", "VAR(5)",
        "VARX 1", "VARX 2", "VARX 3", "VARX-L 2 (Ridge)",
        "LSTM 1", "LSTM 2", "LSTM 3",
        "DeepVAR 1", "DeepVAR 2", "DeepVAR 3",
        "DeepVARX 1", "DeepVARX 2", "DeepVARX 3",
    }
    found_models = set(df["model"].unique())
    assert found_models == expected_models, f"Missing models: {expected_models - found_models}"

    expected_provinces = {"dak_lak", "lam_dong", "gia_lai"}
    assert set(df["province"].unique()) == expected_provinces

    # Verify counts per model: 3 provinces * 542 test days = 1626 observations
    counts = df.groupby("model").size()
    for m in expected_models:
        assert counts[m] == 1626, f"Model {m} has {counts[m]} observations, expected 1626"


def test_ground_truth_consistency_across_models():
    """Ground truth actual values must be 100% identical across all models for identical date & province."""
    df = pd.read_csv(INPUT_CSV)
    grouped = df.groupby(["province", "target_date"])["actual"]
    std_per_date = grouped.std().dropna()
    assert (std_per_date < 1e-4).all(), "Ground truth discrepancy detected across models!"


def test_probabilistic_models_have_valid_pred_std():
    """DeepVAR and DeepVARX models must have valid non-NaN predictive std dev."""
    df = pd.read_csv(INPUT_CSV)
    prob_models = [
        "DeepVAR 1", "DeepVAR 2", "DeepVAR 3",
        "DeepVARX 1", "DeepVARX 2", "DeepVARX 3",
    ]
    for pm in prob_models:
        sub = df[df["model"] == pm]
        assert "pred_std" in sub.columns
        assert sub["pred_std"].notna().all(), f"{pm} has missing pred_std"
        assert (sub["pred_std"] > 0).all(), f"{pm} has non-positive pred_std"


def test_summary_results_integrity():
    """Verify dm_summary_results.csv contains zero NaNs in dm_stat and includes Clark-West metrics."""
    assert SUMMARY_CSV.exists()
    df_sum = pd.read_csv(SUMMARY_CSV)

    assert len(df_sum) == 861  # (136*2 + 15) * 3 provinces = 861
    assert df_sum["dm_stat"].notna().all()
    assert df_sum["hln_stat"].notna().all()
    assert df_sum["hln_p_value"].notna().all()
    assert df_sum["holm_p_value"].notna().all()
    assert (df_sum["n_observations"] == 542).all()

    # Nested models have Clark-West statistics computed under MSE
    nested_mse = df_sum[(df_sum["is_nested"] == True) & (df_sum["loss_function"] == "mse")]
    assert len(nested_mse) > 0
    assert nested_mse["clark_west_stat"].notna().all()
    assert nested_mse["clark_west_p_value"].notna().all()


def test_discovery_report_integrity():
    """Verify model_discovery_report.csv has all 17 models with RMSE and MAE."""
    assert DISCOVERY_CSV.exists()
    df_disc = pd.read_csv(DISCOVERY_CSV)

    assert len(df_disc) == 17
    assert (df_disc["evaluation_status"] == "Included in DM Test").all()
    assert (df_disc["test_rmse_vnd"] > 1000.0).all()
    assert (df_disc["test_mae_vnd"] > 500.0).all()


def test_audit_attributes_and_price_columns_presence():
    """Verify that all recommended audit attributes and price columns are present and populated."""
    # 1. Input dataset
    df_in = pd.read_csv(INPUT_CSV)
    for col in ["actual_price", "predicted_price", "experiment_id", "data_frequency", "forecast_strategy", "actual_source", "model_family", "model_config"]:
        assert col in df_in.columns, f"Missing '{col}' in {INPUT_CSV}"
        assert df_in[col].notna().all(), f"NaNs found in '{col}' of {INPUT_CSV}"

    # Verify price values in input
    assert (df_in["actual_price"] > 40000.0).all()
    assert (df_in["predicted_price"] > 40000.0).all()

    # 2. Summary results
    df_sum = pd.read_csv(SUMMARY_CSV)
    for col in ["model_a_family", "model_b_family", "model_a_config", "model_b_config", "model_a_type", "model_b_type", "mean_actual_price", "mean_pred_price_a", "mean_pred_price_b"]:
        assert col in df_sum.columns, f"Missing '{col}' in {SUMMARY_CSV}"
        assert df_sum[col].notna().all(), f"NaNs found in '{col}' of {SUMMARY_CSV}"

    # 3. Detailed observations
    detail_csv = PROJ_ROOT / "statistical_evaluation" / "data" / "output" / "dm_detail_results.csv"
    assert detail_csv.exists()
    df_det = pd.read_csv(detail_csv, nrows=100)
    for col in ["actual_price", "pred_price_a", "pred_price_b", "model_a_family", "model_b_family"]:
        assert col in df_det.columns, f"Missing '{col}' in {detail_csv}"

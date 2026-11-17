"""Integration test: Execute full DM evaluation on real project coffee test data."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.comparison import run_model_comparisons
from src.data_loader import load_forecast_csv
from src.report_generator import export_evaluation_results


def test_full_coffee_pipeline_integration(tmp_path):
    """Run full pipeline on actual coffee forecasts and verify output consistency."""
    input_file = Path("statistical_evaluation/data/input/coffee_forecasts_test_542days.csv")
    assert input_file.exists(), "Test data file must exist"

    df = load_forecast_csv(input_file)
    assert len(df) == 13008

    # Run comparisons for a subset of models: Naive vs VAR(5) vs VARX 2
    models = ["Naive (Random Walk)", "VAR(5)", "VARX 2"]
    provinces = ["dak_lak", "lam_dong", "gia_lai"]
    horizons = [1]
    loss_functions = ["mse", "mae"]

    results = run_model_comparisons(
        df=df,
        models=models,
        provinces=provinces,
        horizons=horizons,
        loss_functions=loss_functions,
        alpha=0.05,
        hac_lag=5,
        apply_holm=True,
    )

    summary = results.summary_df
    detailed = results.detailed_df

    # 3 pairs * 3 provinces * 1 horizon * 2 loss functions = 18 tests
    assert len(summary) == 18
    # Detailed observations: 18 tests * 542 observations = 9,756 rows
    assert len(detailed) == 18 * 542

    # Check export
    exported = export_evaluation_results(results, output_dir=tmp_path)
    assert Path(exported["summary_csv"]).exists()
    assert Path(exported["detailed_csv"]).exists()
    assert Path(exported["markdown_report"]).exists()

    # Verify VARX 2 outperforms Naive on Dak Lak (MSE)
    sub = summary[
        (summary["province"] == "dak_lak")
        & (summary["model_a"] == "Naive (Random Walk)")
        & (summary["model_b"] == "VARX 2")
        & (summary["loss_function"] == "mse")
    ]
    assert len(sub) == 1
    row = sub.iloc[0]
    # Model A is Naive, Model B is VARX 2. VARX 2 has much lower loss than Naive,
    # so loss_a - loss_b > 0 => mean_loss_diff > 0 and dm_stat > 0
    assert row["mean_loss_diff"] > 0
    assert row["dm_stat"] > 0
    assert row["better_model_by_loss"] == "VARX 2"
    # Raw HLN p-value is significant at 5% level
    assert row["hln_p_value"] < 0.05
    # Holm adjustment controls FWER, so holm_p >= hln_p
    assert row["holm_p_value"] >= row["hln_p_value"]

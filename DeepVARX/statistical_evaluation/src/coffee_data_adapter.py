"""Adapter to extract and export standardized test forecasts from project models.

Extracts test forecasts (542 days x 4 target provinces) for:
- Naive (Random Walk)
- VAR(1), VAR(2), VAR(5)
- VARX 1, VARX 2, VARX-L 2 (Ridge), VARX 3
from the verified master dataset 'data/data_daily_panel_clean_v2.csv'.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path to access utils
_curr_dir = Path(__file__).resolve().parent
_proj_root = _curr_dir.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from utils.var_model import CoffeeVAR, TARGET_PROVINCES, load_and_prepare_coffee_data as load_var_bundle
from utils.varx_model import CoffeeVARX, load_and_prepare_coffee_data as load_varx_bundle


def generate_coffee_test_forecasts(
    data_csv_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate standardized test-set predictions for linear and benchmark models.

    Returns:
        DataFrame in standard long-format:
        [date, province, horizon, model, y_true, y_pred, y_prev]
    """
    if data_csv_path is None:
        data_csv_path = str(_proj_root / "data" / "data_daily_panel_clean_v2.csv")

    if not Path(data_csv_path).exists():
        raise FileNotFoundError(f"Input panel data not found at: {data_csv_path}")

    # Load data bundles
    bundle_var = load_var_bundle(data_csv_path)
    bundle_varx = load_varx_bundle(data_csv_path)

    Y = bundle_var.ret_df.values
    P = bundle_var.price_df.values
    dates = bundle_var.dates
    tr_end = bundle_var.train_end
    va_end = bundle_var.val_end
    T = len(Y)
    te_idx = np.arange(va_end, T)
    te_dates = dates[te_idx].strftime("%Y-%m-%d").tolist()
    n_test = len(te_idx)

    p_true = P[te_idx]       # (N_test, 3)
    p_prev = P[te_idx - 1]   # (N_test, 3)

    model_predictions: Dict[str, np.ndarray] = {
        "Naive (Random Walk)": p_prev,
    }

    # 1. Fit VAR models
    for p in [1, 2, 5]:
        var_mdl = CoffeeVAR(p=p).fit(Y[:tr_end])
        pred_ret = var_mdl.predict_rolling(Y, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        model_predictions[f"VAR({p})"] = pred_p

    # 2. Fit VARX models
    varx_configs = [
        ("VARX 1", "VARX 1", 5, 2, "none", 0.0),
        ("VARX 2", "VARX 2", 5, 2, "none", 0.0),
        ("VARX-L 2 (Ridge)", "VARX 2", 5, 2, "ridge", 0.1),
        ("VARX 3", "VARX 3", 5, 2, "none", 0.0),
    ]

    for model_name, feat_key, p, q, penalty, alpha in varx_configs:
        X = bundle_varx.feature_sets[feat_key].values
        varx_mdl = CoffeeVARX(p=p, q=q, penalty=penalty, alpha=alpha).fit(
            Y[:tr_end], X[:tr_end]
        )
        pred_ret = varx_mdl.predict_rolling(Y, X, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        model_predictions[model_name] = pred_p

    # Reshape into long-format DataFrame
    rows = []
    for model_name, pred_mat in model_predictions.items():
        for prov_idx, prov in enumerate(TARGET_PROVINCES):
            for t in range(n_test):
                rows.append({
                    "date": te_dates[t],
                    "province": prov,
                    "horizon": 1,
                    "model": model_name,
                    "y_true": float(p_true[t, prov_idx]),
                    "y_pred": float(pred_mat[t, prov_idx]),
                    "y_prev": float(p_prev[t, prov_idx]),
                })

    forecast_df = pd.DataFrame(rows)

    if output_csv_path is not None:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(out_p, index=False, encoding="utf-8-sig")
        print(f"Exported {len(forecast_df):,} forecast rows across {len(model_predictions)} models to: {out_p}")

    return forecast_df


if __name__ == "__main__":
    out_default = _proj_root / "statistical_evaluation" / "data" / "input" / "coffee_forecasts_test_542days.csv"
    generate_coffee_test_forecasts(output_csv_path=str(out_default))

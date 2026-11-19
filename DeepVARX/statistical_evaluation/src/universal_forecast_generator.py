"""Universal Model Forecast Generator for all 17 models in the DeepVARX project.

Covers:
1. Benchmark: Naive (Random Walk)
2. VAR Family: VAR(1), VAR(2), VAR(5)
3. VARX Family: VARX 1, VARX 2, VARX-L 2 (Ridge), VARX 3
4. LSTM Family: LSTM 1, LSTM 2, LSTM 3
5. DeepVAR Family: DeepVAR 1, DeepVAR 2, DeepVAR 3 (configured with 2 LSTM layers)
6. DeepVARX Family: DeepVARX 1, DeepVARX 2, DeepVARX 3 (configured with 2 LSTM layers)

All models evaluated strictly on the 542 test days (2024-10-03 to 2026-10-30) across 4 provinces:
dak_lak, lam_dong, gia_lai, dak_nong.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to sys.path
_curr_dir = Path(__file__).resolve().parent
_proj_root = _curr_dir.parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from utils.var_model import CoffeeVAR, TARGET_PROVINCES, load_and_prepare_coffee_data as load_var_bundle
from utils.varx_model import CoffeeVARX, load_and_prepare_coffee_data as load_varx_bundle
from utils.lstm_model import fit_single_lstm_seed, load_and_prepare_coffee_data as load_lstm_bundle
from utils.deepvar_model import fit_single_deepvar_seed, load_and_prepare_coffee_data as load_dv_bundle
from utils.deepvarx_model import (
    CoffeeDeepVARX,
    fit_single_deepvarx_seed,
    load_and_prepare_coffee_data as load_dvx_bundle,
)

logger = logging.getLogger("ForecastGen")


class SessionCoffeeDeepVARX2Layer(CoffeeDeepVARX):
    """Session configuration for DeepVARX with 2 LSTM layers without touching original codebase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lstm_in_dim = self.k_endog if (self.use_cross_attn or self.m_exog == 0) else (self.k_endog + self.m_exog)
        # Configure 2 LSTM layers
        self.lstm_y = nn.LSTM(lstm_in_dim, self.mu_head.in_features, num_layers=2, batch_first=True)


def generate_all_model_forecasts(
    data_csv_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    deep_epochs: int = 15,
    deep_patience: int = 4,
    random_seed: int = 42,
    experiment_id: str = "EXP_SESSION_2LAYER_20261113",
) -> Tuple[pd.DataFrame, Dict]:
    """Train/evaluate all 17 models on the 542 test days and export to standard schema.

    Returns:
        forecast_df: DataFrame with audit and prediction columns.
        model_metadata: Dict with metadata per model.
    """
    t0 = time.time()
    if data_csv_path is None:
        data_csv_path = str(_proj_root / "data" / "data_daily_panel_clean_v2.csv")

    logger.info(f"Loading data bundles from: {data_csv_path}")
    bundle_var = load_var_bundle(data_csv_path)
    bundle_varx = load_varx_bundle(data_csv_path)
    bundle_lstm = load_lstm_bundle(data_csv_path)
    bundle_dv = load_dv_bundle(data_csv_path)
    bundle_dvx = load_dvx_bundle(data_csv_path)

    Y = bundle_var.ret_df.values
    P = bundle_var.price_df.values
    dates = bundle_var.dates
    tr_end = bundle_var.train_end
    va_end = bundle_var.val_end
    T = len(Y)
    te_idx = np.arange(va_end, T)
    te_dates = dates[te_idx].strftime("%Y-%m-%d").tolist()
    origin_dates = dates[te_idx - 1].strftime("%Y-%m-%d").tolist()
    n_test = len(te_idx)

    p_true = P[te_idx]       # (542, 3)
    p_prev = P[te_idx - 1]   # (542, 3)

    records = []
    metadata = {}

    def _append_model_preds(
        model_name: str,
        pred_mat: np.ndarray,
        std_mat: Optional[np.ndarray] = None,
        family: str = "Linear",
        details: str = "",
        model_config: str = "",
        prediction_type: str = "point",
    ):
        for prov_i, prov in enumerate(TARGET_PROVINCES):
            for t in range(n_test):
                entry = {
                    "experiment_id": experiment_id,
                    "data_frequency": "daily",
                    "forecast_strategy": "rolling_1step_ahead",
                    "actual_source": "GiaCaPhe_Clean_V2",
                    "model": model_name,
                    "model_family": family,
                    "model_config": model_config or details,
                    "prediction_type": prediction_type,
                    "province": prov,
                    "forecast_origin": origin_dates[t],
                    "target_date": te_dates[t],
                    "horizon": 1,
                    "actual_price": float(p_true[t, prov_i]),
                    "predicted_price": float(pred_mat[t, prov_i]),
                    "prev_price": float(p_prev[t, prov_i]),
                    "actual": float(p_true[t, prov_i]),
                    "predicted": float(pred_mat[t, prov_i]),
                }
                if std_mat is not None:
                    entry["pred_std"] = float(std_mat[t, prov_i])
                records.append(entry)

        rmse = float(np.sqrt(np.mean((p_true - pred_mat) ** 2)))
        mae = float(np.mean(np.abs(p_true - pred_mat)))
        metadata[model_name] = {
            "family": family,
            "rmse": rmse,
            "mae": mae,
            "has_std": std_mat is not None,
            "details": details,
            "model_config": model_config or details,
            "prediction_type": prediction_type,
        }
        logger.info(f"[{family}] {model_name:<20} | Test RMSE: {rmse:,.1f} | MAE: {mae:,.1f} | {details}")

    # =========================================================================
    # 1. Benchmark: Naive (Random Walk)
    # =========================================================================
    _append_model_preds(
        "Naive (Random Walk)",
        p_prev,
        family="Benchmark",
        details="P_hat_t = P_{t-1} (No parameters)",
        model_config="p=0, no_params, P_hat_t=P_{t-1}",
        prediction_type="point",
    )

    # =========================================================================
    # 2. VAR Models (Vector Autoregression)
    # =========================================================================
    for p in [1, 2, 5]:
        var_mdl = CoffeeVAR(p=p).fit(Y[:tr_end])
        pred_ret = var_mdl.predict_rolling(Y, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        _append_model_preds(
            f"VAR({p})",
            pred_p,
            family="VAR",
            details=f"p={p} lags",
            model_config=f"p={p}_lags, endog_only(4_provinces), OLS",
            prediction_type="point",
        )

    # =========================================================================
    # 3. VARX Models (Vector Autoregression with Exogenous Variables)
    # =========================================================================
    varx_cfgs = [
        ("VARX 1", "VARX 1", 5, 2, "none", 0.0, "5 vars (London + FX)"),
        ("VARX 2", "VARX 2", 5, 2, "none", 0.0, "11 core vars (Weather + Econ)"),
        ("VARX-L 2 (Ridge)", "VARX 2", 5, 2, "ridge", 0.1, "11 vars + Ridge alpha=0.1"),
        ("VARX 3", "VARX 3", 5, 2, "none", 0.0, "20 full exogenous vars"),
    ]
    for mname, fkey, p, q, pen, alpha, desc in varx_cfgs:
        X = bundle_varx.feature_sets[fkey].values
        varx_mdl = CoffeeVARX(p=p, q=q, penalty=pen, alpha=alpha).fit(Y[:tr_end], X[:tr_end])
        pred_ret = varx_mdl.predict_rolling(Y, X, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        _append_model_preds(
            mname,
            pred_p,
            family="VARX",
            details=desc,
            model_config=f"p={p}, q={q}, penalty={pen}, alpha={alpha}, {desc}",
            prediction_type="point",
        )

    # =========================================================================
    # 4. LSTM Models (LSTM 1, 2, 3)
    # =========================================================================
    lstm_cfgs = [
        ("LSTM 1", "LSTM 1", 10, 1, 0.0, 0.0, "Univariate return baseline"),
        ("LSTM 2", "LSTM 2", 10, 1, 0.1, 0.0, "Multivariate 11 vars"),
        ("LSTM 3", "LSTM 3", 10, 1, 0.2, 0.2, "Full exogenous 20 vars"),
    ]
    for mname, fkey, seq, num_l, drop, in_drop, desc in lstm_cfgs:
        X_df = bundle_lstm.feature_sets[fkey]
        res_l = fit_single_lstm_seed(
            X_df=X_df,
            Y_df=bundle_lstm.ret_df,
            price_df=bundle_lstm.price_df,
            train_end=tr_end,
            val_end=va_end,
            seq_len=seq,
            hidden_dim=64,
            num_layers=num_l,
            dropout=drop,
            in_dropout=in_drop,
            batch_size=32,
            lr=1e-3,
            max_epochs=deep_epochs,
            patience=deep_patience,
            scaler_type="standard",
            seed=random_seed,
        )
        _append_model_preds(
            mname,
            res_l["test_pred_price"],
            family="LSTM",
            details=desc,
            model_config=f"seq={seq}, hidden=64, layers={num_l}, drop={drop}, in_drop={in_drop}, {desc}",
            prediction_type="point",
        )

    # =========================================================================
    # 5. DeepVAR Models (DeepVAR 1, 2, 3 with 2 LSTM layers)
    # =========================================================================
    dv_cfgs = [
        ("DeepVAR 1", "DeepVAR 1", 10, 2, 2, 0.0, 0.0, "Probabilistic baseline (2 layers)"),
        ("DeepVAR 2", "DeepVAR 2", 10, 2, 2, 0.1, 0.0, "Multivariate 11 vars (2 layers)"),
        ("DeepVAR 3", "DeepVAR 3", 10, 2, 2, 0.2, 0.2, "Full 20 vars (2 layers)"),
    ]
    for mname, fkey, seq, num_l, rk, drop, in_drop, desc in dv_cfgs:
        X_df = bundle_dv.feature_sets[fkey]
        res_dv = fit_single_deepvar_seed(
            X_df=X_df,
            Y_df=bundle_dv.ret_df,
            price_df=bundle_dv.price_df,
            train_end=tr_end,
            val_end=va_end,
            seq_len=seq,
            hidden_dim=64,
            cov_hidden_dim=32,
            num_layers=num_l,
            rank=rk,
            dropout=drop,
            in_dropout=in_drop,
            batch_size=32,
            lr=1e-3,
            max_epochs=deep_epochs,
            patience=deep_patience,
            seed=random_seed,
        )
        std_r = res_dv.get("test_std_ret_cal")
        std_p = res_dv["test_pred_price"] * std_r if std_r is not None else None
        _append_model_preds(
            mname,
            res_dv["test_pred_price"],
            std_mat=std_p,
            family="DeepVAR",
            details=desc,
            model_config=f"seq={seq}, hidden=64, cov_hidden=32, layers={num_l}, rank={rk}, drop={drop}, in_drop={in_drop}, Gaussian_NLL, {desc}",
            prediction_type="probabilistic",
        )

    # =========================================================================
    # 6. DeepVARX Models (DeepVARX 1, 2, 3 with 2 LSTM layers)
    # =========================================================================
    # Temporarily monkey-patch CoffeeDeepVARX in utils for this session run
    import utils.deepvarx_model
    orig_dvx_cls = utils.deepvarx_model.CoffeeDeepVARX
    utils.deepvarx_model.CoffeeDeepVARX = SessionCoffeeDeepVARX2Layer

    dvx_cfgs = [
        ("DeepVARX 1", "DeepVARX 1", 10, 2, 0.0, 0.0, True, "TimeXer baseline (2 layers)"),
        ("DeepVARX 2", "DeepVARX 2", 10, 2, 0.15, 0.1, True, "TimeXer Dual-Stream 11 vars (2 layers)"),
        ("DeepVARX 3", "DeepVARX 3", 10, 2, 0.2, 0.15, True, "TimeXer Full 20 vars (2 layers)"),
    ]
    try:
        for mname, fkey, seq, rk, drop, in_drop, use_ca, desc in dvx_cfgs:
            X_df = bundle_dvx.feature_sets[fkey]
            res_dvx = fit_single_deepvarx_seed(
                X_df=X_df,
                Y_df=bundle_dvx.ret_df,
                price_df=bundle_dvx.price_df,
                train_end=tr_end,
                val_end=va_end,
                seq_len=seq,
                hidden_dim=64,
                cov_hidden_dim=32,
                rank=rk,
                dropout=drop,
                in_dropout=in_drop,
                batch_size=32,
                lr=1e-3,
                max_epochs=deep_epochs,
                patience=deep_patience,
                use_cross_attn=use_ca,
                seed=random_seed,
            )
            std_r = res_dvx.get("test_std_ret_cal")
            std_p = res_dvx["test_pred_price"] * std_r if std_r is not None else None
            _append_model_preds(
                mname,
                res_dvx["test_pred_price"],
                std_mat=std_p,
                family="DeepVARX",
                details=desc,
                model_config=f"seq={seq}, hidden=64, cov_hidden=32, layers=2, rank={rk}, drop={drop}, in_drop={in_drop}, cross_attn={use_ca}, Gaussian_NLL, {desc}",
                prediction_type="probabilistic",
            )
    finally:
        utils.deepvarx_model.CoffeeDeepVARX = orig_dvx_cls

    df_forecasts = pd.DataFrame(records)
    logger.info(f"Generated {len(df_forecasts):,} total rows across {len(metadata)} models in {time.time() - t0:.1f}s.")

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_forecasts.to_csv(out_p, index=False, encoding="utf-8-sig")
        logger.info(f"Saved complete forecast dataset to: {out_p}")

    return df_forecasts, metadata


if __name__ == "__main__":
    out_default = _proj_root / "statistical_evaluation" / "data" / "input" / "all_models_forecasts_test_542days.csv"
    generate_all_model_forecasts(output_csv_path=str(out_default))

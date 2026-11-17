"""Multi-model comparison coordinator and statistical evaluation engine."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .clark_west import clark_west_test
from .data_alignment import align_model_pair
from .dm_test import DMTestResult, diebold_mariano_test
from .holm_correction import holm_bonferroni_correction
from .loss_functions import compute_loss_differential, get_loss_function

logger = logging.getLogger(__name__)


def check_is_nested(model_a: str, model_b: str) -> Tuple[bool, Optional[str], Optional[str], str]:
    """Check if a pair of models represents nested specifications.

    Returns:
        (is_nested, restricted_model, unrestricted_model, warning_msg)
    """
    import re

    a = model_a.lower().strip()
    b = model_b.lower().strip()

    # Rule 1: Naive (Random Walk) is restricted vs all linear / autoregressive models
    if "naive" in a and ("var" in b or "lstm" in b):
        return True, model_a, model_b, f"'{model_a}' (restricted) is nested in '{model_b}' (unrestricted). Clark-West (2007) test applicable."
    if "naive" in b and ("var" in a or "lstm" in a):
        return True, model_b, model_a, f"'{model_b}' (restricted) is nested in '{model_a}' (unrestricted). Clark-West (2007) test applicable."

    # Rule 2: Pure VAR lag hierarchy: VAR(1) is nested in VAR(2) and VAR(5); VAR(2) is nested in VAR(5)
    if "var(" in a and "var(" in b and "varx" not in a and "varx" not in b:
        m_a = re.search(r"var\((\d+)\)", a)
        m_b = re.search(r"var\((\d+)\)", b)
        if m_a and m_b:
            p_a, p_b = int(m_a.group(1)), int(m_b.group(1))
            if p_a < p_b:
                return True, model_a, model_b, f"'{model_a}' (restricted lag={p_a}) is nested in '{model_b}' (unrestricted lag={p_b})."
            elif p_b < p_a:
                return True, model_b, model_a, f"'{model_b}' (restricted lag={p_b}) is nested in '{model_a}' (unrestricted lag={p_a})."

    # Rule 3: VAR without X is nested in VARX (with exogenous features)
    if "var(" in a and "varx" in b and "deep" not in b:
        return True, model_a, model_b, f"'{model_a}' (VAR) is nested in '{model_b}' (VARX)."
    if "var(" in b and "varx" in a and "deep" not in a:
        return True, model_b, model_a, f"'{model_b}' (VAR) is nested in '{model_a}' (VARX)."

    # Rule 4: Feature subset hierarchy for same family (VARX, LSTM, DeepVAR, DeepVARX)
    for prefix in ["varx", "lstm", "deepvar", "deepvarx"]:
        is_a_fam = prefix in a and not ("deepvarx" in a and prefix == "deepvar")
        is_b_fam = prefix in b and not ("deepvarx" in b and prefix == "deepvar")
        if is_a_fam and is_b_fam:
            m_a = re.search(r"\b(\d+)\b", a)
            m_b = re.search(r"\b(\d+)\b", b)
            if m_a and m_b:
                n_a, n_b = int(m_a.group(1)), int(m_b.group(1))
                if n_a < n_b:
                    return True, model_a, model_b, f"'{model_a}' is nested in '{model_b}' (feature subset hierarchy)."
                elif n_b < n_a:
                    return True, model_b, model_a, f"'{model_b}' is nested in '{model_a}' (feature subset hierarchy)."

    return False, None, None, ""


def infer_model_family(model_name: str) -> str:
    """Infer family category from model name."""
    m = model_name.lower().strip()
    if "naive" in m:
        return "Benchmark"
    if "deepvarx" in m:
        return "DeepVARX"
    if "deepvar" in m:
        return "DeepVAR"
    if "varx" in m:
        return "VARX"
    if "var" in m:
        return "VAR"
    if "lstm" in m:
        return "LSTM"
    return "Other"


def infer_model_config(model_name: str) -> str:
    """Infer concise configuration string from model name."""
    m = model_name.lower().strip()
    if "naive" in m:
        return "p=0, no_params, P_hat_t=P_{t-1}"
    if "var(1)" in m:
        return "p=1_lag, endog_only(3_provinces), OLS"
    if "var(2)" in m:
        return "p=2_lags, endog_only(3_provinces), OLS"
    if "var(5)" in m:
        return "p=5_lags, endog_only(3_provinces), OLS"
    if "varx 1" in m:
        return "p=5, q=2, vars=5(London+FX), penalty=none, OLS"
    if "varx-l 2" in m:
        return "p=5, q=2, vars=11(Weather+Econ), penalty=ridge, alpha=0.1"
    if "varx 2" in m:
        return "p=5, q=2, vars=11(Weather+Econ), penalty=none, OLS"
    if "varx 3" in m:
        return "p=5, q=2, vars=20(Full_exogenous), penalty=none, OLS"
    if "lstm 1" in m:
        return "seq=10, hidden=64, layers=1, drop=0.0, vars=3(univariate_returns)"
    if "lstm 2" in m:
        return "seq=10, hidden=64, layers=1, drop=0.1, vars=11(core_exog)"
    if "lstm 3" in m:
        return "seq=10, hidden=64, layers=1, drop=0.2, in_drop=0.2, vars=20(full_exog)"
    if "deepvar 1" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.0, Gaussian_NLL, vars=3(returns)"
    if "deepvar 2" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.1, Gaussian_NLL, vars=11(core)"
    if "deepvar 3" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.2, in_drop=0.2, Gaussian_NLL, vars=20(full)"
    if "deepvarx 1" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.0, cross_attn=True, Gaussian_NLL, vars=3(baseline)"
    if "deepvarx 2" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.15, in_drop=0.1, cross_attn=True, Gaussian_NLL, vars=11(core)"
    if "deepvarx 3" in m:
        return "seq=10, hidden=64, cov_hidden=32, layers=2, rank=2, drop=0.2, in_drop=0.15, cross_attn=True, Gaussian_NLL, vars=20(full)"
    return "default_config"


def infer_prediction_type(model_name: str) -> str:
    """Infer output format type: point or probabilistic."""
    m = model_name.lower().strip()
    return "probabilistic" if "deep" in m else "point"


@dataclass
class ComparisonSuiteResult:
    summary_df: pd.DataFrame
    detailed_df: pd.DataFrame
    metadata: Dict


def run_model_comparisons(
    df: pd.DataFrame,
    models: Optional[List[str]] = None,
    provinces: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
    loss_functions: Optional[List[str]] = None,
    alpha: float = 0.05,
    hac_lag: Optional[int] = None,
    apply_holm: bool = True,
    warn_nested: bool = True,
    min_observations: int = 10,
) -> ComparisonSuiteResult:
    """Execute pairwise DM tests across all specified models, provinces, horizons, and loss functions.

    Parameters:
        df: Input standardized forecast DataFrame.
        models: Subset of models to compare (None = all models found).
        provinces: Subset of provinces (None = all provinces found).
        horizons: Forecast horizons (None = all horizons found).
        loss_functions: Loss functions to evaluate (default ['mse', 'mae']).
        alpha: Statistical significance threshold.
        hac_lag: Bandwidth for Newey-West HAC (None = automatic).
        apply_holm: Whether to apply Holm-Bonferroni correction.
        warn_nested: Whether to flag nested model pairs.
    """
    if models is None:
        models = sorted(df["model"].dropna().unique().tolist())
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models for pairwise comparison, got: {models}")

    if provinces is None:
        provinces = sorted(df["province"].dropna().unique().tolist())
    if horizons is None:
        horizons = sorted(df["horizon"].dropna().unique().tolist()) if "horizon" in df.columns else [1]
    if loss_functions is None:
        loss_functions = ["mse", "mae"]

    summary_rows = []
    detailed_frames = []

    model_pairs = list(itertools.combinations(models, 2))

    for prov in provinces:
        for h in horizons:
            for loss_name in loss_functions:
                family_results = []
                family_detailed = []

                for model_a, model_b in model_pairs:
                    is_nested, restr_m, unrestr_m, nested_msg = check_is_nested(model_a, model_b)
                    if is_nested and warn_nested:
                        logger.warning(nested_msg)

                    # Align pair
                    aligned, align_meta = align_model_pair(
                        df, model_a=model_a, model_b=model_b, province=prov, horizon=h
                    )

                    if len(aligned) < min_observations:
                        logger.warning(
                            f"Skipping {model_a} vs {model_b} on {prov} (h={h}): "
                            f"Only {len(aligned)} observations (< {min_observations})."
                        )
                        continue

                    # Extract predictive standard deviation if available
                    std_a = aligned["pred_std_a"].to_numpy() if "pred_std_a" in aligned.columns else None
                    std_b = aligned["pred_std_b"].to_numpy() if "pred_std_b" in aligned.columns else None

                    has_valid_std_a = std_a is not None and not np.isnan(std_a).all()
                    has_valid_std_b = std_b is not None and not np.isnan(std_b).all()

                    # If CRPS requested, both models must supply valid predictive standard deviation
                    if loss_name == "crps" and (not has_valid_std_a or not has_valid_std_b):
                        logger.debug(
                            f"Skipping CRPS for {model_a} vs {model_b}: probabilistic predictions required."
                        )
                        continue

                    pass_std_a = std_a if loss_name == "crps" else None
                    pass_std_b = std_b if loss_name == "crps" else None

                    # Run DM test
                    res = diebold_mariano_test(
                        y_true=aligned["y_true"].to_numpy(),
                        y_pred_a=aligned["pred_a"].to_numpy(),
                        y_pred_b=aligned["pred_b"].to_numpy(),
                        model_a_name=model_a,
                        model_b_name=model_b,
                        loss_func=loss_name,
                        horizon=h,
                        hac_lag=hac_lag,
                        pred_std_a=pass_std_a,
                        pred_std_b=pass_std_b,
                        min_observations=min_observations,
                    )

                    # Compute detailed loss rows
                    loss_diff, loss_a, loss_b = compute_loss_differential(
                        y_true=aligned["y_true"].to_numpy(),
                        y_pred_a=aligned["pred_a"].to_numpy(),
                        y_pred_b=aligned["pred_b"].to_numpy(),
                        loss_func=loss_name,
                        pred_std_a=pass_std_a,
                        pred_std_b=pass_std_b,
                    )

                    # Clark-West (2007) test for nested models under MSE
                    cw_stat = np.nan
                    cw_p_val = np.nan
                    if is_nested and loss_name in ("mse", "squared_error"):
                        y_tr = aligned["y_true"].to_numpy()
                        if restr_m == model_a:
                            y_r = aligned["pred_a"].to_numpy()
                            y_u = aligned["pred_b"].to_numpy()
                        else:
                            y_r = aligned["pred_b"].to_numpy()
                            y_u = aligned["pred_a"].to_numpy()
                        cws, cwp, _, _ = clark_west_test(y_tr, y_r, y_u, horizon=h, hac_lag=hac_lag)
                        cw_stat = cws
                        cw_p_val = cwp

                    t_dt = aligned["target_date"] if "target_date" in aligned.columns else aligned.get("date", "")
                    f_orig = aligned["forecast_origin"] if "forecast_origin" in aligned.columns else ""

                    exp_id = aligned["experiment_id"].iloc[0] if "experiment_id" in aligned.columns else "EXP_SESSION_2LAYER_20261113"
                    freq = aligned["data_frequency"].iloc[0] if "data_frequency" in aligned.columns else "daily"
                    strat = aligned["forecast_strategy"].iloc[0] if "forecast_strategy" in aligned.columns else "rolling_1step_ahead"
                    src = aligned["actual_source"].iloc[0] if "actual_source" in aligned.columns else "GiaCaPhe_Clean_V2"

                    fam_a = aligned["model_a_family"].iloc[0] if "model_a_family" in aligned.columns else infer_model_family(model_a)
                    fam_b = aligned["model_b_family"].iloc[0] if "model_b_family" in aligned.columns else infer_model_family(model_b)
                    cfg_a = aligned["model_a_config"].iloc[0] if "model_a_config" in aligned.columns else infer_model_config(model_a)
                    cfg_b = aligned["model_b_config"].iloc[0] if "model_b_config" in aligned.columns else infer_model_config(model_b)
                    typ_a = aligned["model_a_type"].iloc[0] if "model_a_type" in aligned.columns else infer_prediction_type(model_a)
                    typ_b = aligned["model_b_type"].iloc[0] if "model_b_type" in aligned.columns else infer_prediction_type(model_b)

                    detail_df = pd.DataFrame({
                        "experiment_id": exp_id,
                        "data_frequency": freq,
                        "forecast_strategy": strat,
                        "actual_source": src,
                        "target_date": t_dt,
                        "forecast_origin": f_orig,
                        "province": prov,
                        "horizon": h,
                        "loss_function": loss_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "model_a_family": fam_a,
                        "model_b_family": fam_b,
                        "actual_price": aligned["y_true"],
                        "pred_price_a": aligned["pred_a"],
                        "pred_price_b": aligned["pred_b"],
                        "actual_value": aligned["y_true"],
                        "pred_a": aligned["pred_a"],
                        "pred_b": aligned["pred_b"],
                        "error_a": aligned["error_a"],
                        "error_b": aligned["error_b"],
                        "loss_a": loss_a,
                        "loss_b": loss_b,
                        "loss_differential": loss_diff,
                    })
                    family_detailed.append(detail_df)

                    row_entry = {
                        "experiment_id": exp_id,
                        "data_frequency": freq,
                        "forecast_strategy": strat,
                        "actual_source": src,
                        "province": prov,
                        "horizon": h,
                        "model_a": model_a,
                        "model_b": model_b,
                        "model_a_family": fam_a,
                        "model_b_family": fam_b,
                        "model_a_config": cfg_a,
                        "model_b_config": cfg_b,
                        "model_a_type": typ_a,
                        "model_b_type": typ_b,
                        "loss_function": loss_name,
                        "n_observations": res.n_observations,
                        "mean_actual_price": round(float(np.mean(aligned["y_true"])), 2),
                        "mean_pred_price_a": round(float(np.mean(aligned["pred_a"])), 2),
                        "mean_pred_price_b": round(float(np.mean(aligned["pred_b"])), 2),
                        "mean_loss_a": res.mean_loss_a,
                        "mean_loss_b": res.mean_loss_b,
                        "mean_loss_diff": res.mean_loss_diff,
                        "dm_stat": res.dm_stat,
                        "dm_p_value": res.dm_p_value_two_sided,
                        "hln_stat": res.hln_stat,
                        "hln_p_value": res.hln_p_value_two_sided,
                        "hac_lag": res.hac_lag,
                        "better_model_by_loss": res.better_model_by_loss,
                        "is_nested": is_nested,
                        "nested_restricted_model": restr_m or "",
                        "nested_unrestricted_model": unrestr_m or "",
                        "clark_west_stat": cw_stat,
                        "clark_west_p_value": cw_p_val,
                        "nested_warning": nested_msg,
                    }
                    family_results.append(row_entry)

                if not family_results:
                    continue

                # Apply Holm-Bonferroni correction over family of tests
                family_df = pd.DataFrame(family_results)
                if apply_holm and len(family_df) > 0:
                    raw_p = family_df["hln_p_value"].to_numpy()
                    family_df["holm_p_value"] = holm_bonferroni_correction(raw_p)
                else:
                    family_df["holm_p_value"] = family_df["hln_p_value"]

                # Statistical Significance and Final Interpretation
                for idx, row in family_df.iterrows():
                    eval_p = row["holm_p_value"] if apply_holm else row["hln_p_value"]
                    is_sig = bool(eval_p < alpha) and not row["better_model_by_loss"].startswith("Equal")

                    stat_winner = "None (No significant difference)"
                    if is_sig:
                        if row["mean_loss_diff"] < 0:
                            stat_winner = row["model_a"]
                            interp = f"{row['model_a']} significantly outperforms {row['model_b']} (p={eval_p:.4f} < {alpha})"
                        else:
                            stat_winner = row["model_b"]
                            interp = f"{row['model_b']} significantly outperforms {row['model_a']} (p={eval_p:.4f} < {alpha})"
                    else:
                        interp = f"No significant difference between {row['model_a']} and {row['model_b']} (p={eval_p:.4f} >= {alpha})"

                    family_df.loc[idx, "significance_level"] = alpha
                    family_df.loc[idx, "is_statistically_significant"] = is_sig
                    family_df.loc[idx, "better_model_statistical"] = stat_winner
                    family_df.loc[idx, "final_interpretation"] = interp

                summary_rows.append(family_df)
                if family_detailed:
                    detailed_frames.extend(family_detailed)

    summary_combined = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    detailed_combined = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()

    meta = {
        "models_evaluated": models,
        "provinces_evaluated": provinces,
        "horizons_evaluated": horizons,
        "loss_functions": loss_functions,
        "alpha": alpha,
        "hac_lag": hac_lag,
        "total_pairwise_tests": len(summary_combined),
        "total_observations_evaluated": len(detailed_combined),
    }

    return ComparisonSuiteResult(
        summary_df=summary_combined,
        detailed_df=detailed_combined,
        metadata=meta,
    )

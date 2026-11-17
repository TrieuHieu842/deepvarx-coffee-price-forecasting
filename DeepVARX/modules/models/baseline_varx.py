"""Baseline tuyến tính: VAR (không X), VARX (có X), VARX-L (VARX + phạt Lasso/Ridge).
Cài đặt kết nối trực tiếp với utils.var_model (VARModel) và utils.varx_model (VARXModel).
Dự báo 1 bước (rolling one-step-ahead) trên đoạn TEST, rồi quy đổi về mức giá gốc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_utils import difference, reconstruct_level
from utils.var_model import VARModel
from utils.varx_model import VARXModel


def _design(endog: np.ndarray, exog: np.ndarray | None, p: int, q: int, m: int):
    T = endog.shape[0]
    cols = [endog[m - k:T - k] for k in range(1, p + 1)]
    if exog is not None and q > 0:
        cols += [exog[m - k:T - k] for k in range(1, q + 1)]
    return np.hstack(cols), endog[m:]


def select_order(endog: np.ndarray, exog: np.ndarray | None, rows_end: int, max_p: int, max_q: int,
                 criterion: str = "AIC") -> tuple[pd.DataFrame, tuple[int, int]]:
    """Lưới (p,q) chọn theo AIC/BIC trên mẫu chung."""
    m = max(max_p, max_q if exog is not None else 0)
    ne = endog.shape[1]
    nx = 0 if exog is None else exog.shape[1]
    rows = []
    for p in range(1, max_p + 1):
        for q in (range(1, max_q + 1) if exog is not None else [0]):
            X, Y = _design(endog[:rows_end], exog[:rows_end] if exog is not None else None, p, q, m)
            Z = np.column_stack([np.ones(len(X)), X])
            beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
            res = Y - Z @ beta
            N = len(Y)
            sigma = np.atleast_2d(np.cov(res.T, bias=True))
            k = ne * (1 + ne * p + nx * q)
            pen = 2 * k if criterion == "AIC" else k * np.log(N)
            rows.append({"p": p, "q": q, criterion: N * np.log(np.linalg.det(sigma) + 1e-300) + pen})
    grid = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    best = (int(grid.loc[0, "p"]), int(grid.loc[0, "q"]))
    return grid, best


def _fit_and_forecast_linear(endog: np.ndarray, exog: np.ndarray | None, endog_names: list[str],
                             exog_names: list[str], rows_end: int, p: int, q: int,
                             lasso_alpha: float | None):
    """Huấn luyện VARModel hoặc VARXModel trên [:rows_end] và dự báo rolling 1 bước trên [rows_end:]."""
    if exog is None or exog.shape[1] == 0:
        mdl = VARModel(lags=p, trend="c")
        tr_df = pd.DataFrame(endog[:rows_end], columns=endog_names)
        te_df = pd.DataFrame(endog[rows_end:], columns=endog_names)
        mdl.fit(tr_df)
        pred_mat = mdl.forecast_rolling(te_df, tr_df).to_numpy(float)
        return pred_mat, mdl

    penalty = "lasso" if (lasso_alpha is not None and lasso_alpha > 0) else "none"
    alpha = float(lasso_alpha) if lasso_alpha is not None else 0.0
    mdl = VARXModel(p=p, q=max(1, q), penalty=penalty, alpha=alpha, trend="c", exog_lag_mode="past_only")
    tr_y = pd.DataFrame(endog[:rows_end], columns=endog_names)
    te_y = pd.DataFrame(endog[rows_end:], columns=endog_names)
    tr_x = pd.DataFrame(exog[:rows_end], columns=exog_names)
    te_x = pd.DataFrame(exog[rows_end:], columns=exog_names)
    mdl.fit(tr_y, tr_x)
    pred_mat = mdl.forecast_rolling(te_y, te_x, tr_y, tr_x).to_numpy(float)
    return pred_mat, mdl


def run_varx(df_levels: pd.DataFrame, target: str, exog_cols: list[str], d: int, train_n: int, p: int, q: int,
             lasso_alpha: float | None = None) -> dict:
    """Chạy VAR/VARX/VARX-L trên dữ liệu sai phân bậc d, trả về dự báo TEST ở mức giá gốc."""
    cols = [target] + list(exog_cols)
    data_d = difference(df_levels[cols], d)
    endog = data_d[[target]].to_numpy(float)
    exog = data_d[exog_cols].to_numpy(float) if exog_cols else None
    rows_end = train_n - d

    lvl = df_levels[target].to_numpy(float)
    gi = np.arange(rows_end, len(endog)) + d

    if exog is None or exog.shape[1] == 0:
        pred_mat, mdl = _fit_and_forecast_linear(endog, None, [target], [], rows_end, p, q, lasso_alpha)
        imp = None
        n_params = p + 1
    else:
        pred_mat, mdl = _fit_and_forecast_linear(endog, exog, [target], list(exog_cols), rows_end, p, q, lasso_alpha)
        imp = mdl.exog_importance()
        n_params = len(mdl.feature_names) if mdl.feature_names else (p + q * len(exog_cols) + 1)

    y_pred = reconstruct_level(lvl, gi, d, pred_mat[:, 0])
    return {
        "index": df_levels.index[gi],
        "y_true": lvl[gi],
        "y_pred": y_pred,
        "n_params": int(n_params),
        "model": mdl,
        "exog_importance": imp,
    }


def run_local_pkg(g: pd.DataFrame, date_col: str, target: str, feats: list[str], d: int, p: int, q: int,
                  lasso_alpha: float | None = None, ect_rank: int = 0, k_ar_diff: int = 2,
                  levels_input: bool = False, endog_cols: list[str] | None = None,
                  shrinkage: float = 1.0, **kwargs):
    """Chạy VAR/VARX/VARX-L (hoặc VECM-X khi ect_rank > 0) cho MỘT tỉnh từ gói dữ liệu của tỉnh đó."""
    if endog_cols is None and "endog_cols" in kwargs:
        endog_cols = kwargs["endog_cols"]
    from modules.stat_tests import johansen_beta

    g = g.sort_values(date_col).reset_index(drop=True)
    endog_names = [c for c in endog_cols if c in g.columns] if endog_cols else [target]
    if not endog_names:
        endog_names = [target]
    exog_names = [c for c in feats if c in g.columns and c not in endog_names]
    cols = endog_names + exog_names
    levels_input = levels_input or ect_rank > 0
    if levels_input:
        lev = g[cols].to_numpy(float)
        dd, off = difference(g[cols], d).to_numpy(float), d
    else:
        lev, dd, off = None, g[cols].to_numpy(float), 0

    split = g["split"].to_numpy()[off:]
    n, rows_end = len(dd), int((split == "train").sum())
    endog = dd[:, :len(endog_names)]
    exog = dd[:, len(endog_names):] if exog_names else None

    if ect_rank > 0:
        beta = johansen_beta(lev[:rows_end + off], ect_rank, k_ar_diff)
        ect = (lev @ beta)[off:]
        exog = ect if exog is None else np.column_stack([exog, ect])
        exog_names = exog_names + [f"ECT_{r+1}" for r in range(ect.shape[1])]

    lvl = g["y_level"].to_numpy(float)
    pred_mat, mdl = _fit_and_forecast_linear(endog, exog, endog_names, exog_names, rows_end, p, q, lasso_alpha)
    gi = np.arange(rows_end, n) + off
    y_pred = reconstruct_level(lvl, gi, d, pred_mat[:, 0])

    y_prev_arr = lvl[gi - 1]
    if shrinkage is not None and 0.0 < float(shrinkage) < 1.0:
        y_pred = y_prev_arr + float(shrinkage) * (y_pred - y_prev_arr)

    out = pd.DataFrame({
        "date": g[date_col].to_numpy()[gi],
        "y_true": lvl[gi],
        "y_pred": y_pred,
        "y_prev": y_prev_arr,
    })
    return out, mdl


def run_wide_pkg(pkg_df: pd.DataFrame, y_levels_df: pd.DataFrame, date_col: str, endog_cols: list[str],
                 feats: list[str], d: int, p: int, q: int, lasso_alpha: float | None = None,
                 ect_rank: int = 0, k_ar_diff: int = 2, levels_input: bool = False,
                 shrinkage: float = 1.0, endog_labels: dict[str, str] | None = None):
    """Chạy VAR/VARX/VARX-L đồng thời cho toàn bộ hệ biến nội sinh (giá từng tỉnh VN) ở chế độ Wide."""
    from modules.stat_tests import johansen_beta

    pkg_df = pkg_df.sort_values(date_col).reset_index(drop=True)
    y_lvl_sorted = y_levels_df.sort_values(date_col).reset_index(drop=True)
    endog_names = [c for c in endog_cols if c in pkg_df.columns]
    exog_names = [c for c in feats if c in pkg_df.columns and c not in endog_names]
    cols = endog_names + exog_names
    levels_input = levels_input or ect_rank > 0

    if levels_input:
        lev = pkg_df[cols].to_numpy(float)
        dd, off = difference(pkg_df[cols], d).to_numpy(float), d
        split = pkg_df["split"].to_numpy()[off:]
        eff_dates = pkg_df[date_col].to_numpy()[off:]
    else:
        lev, dd, off = None, pkg_df[cols].to_numpy(float), 0
        split = pkg_df["split"].to_numpy()
        eff_dates = pkg_df[date_col].to_numpy()

    rows_end = int((split == "train").sum())
    endog = dd[:, :len(endog_names)]
    exog = dd[:, len(endog_names):] if exog_names else None

    if ect_rank > 0:
        if lev is None:
            lev = y_lvl_sorted.set_index(date_col).loc[pd.to_datetime(pkg_df[date_col]), endog_names].to_numpy(float)
            if exog_names:
                lev = np.column_stack([lev, pkg_df[exog_names].to_numpy(float)])
        beta = johansen_beta(lev[:rows_end + off], ect_rank, k_ar_diff)
        ect = (lev @ beta)[off:]
        exog = ect if exog is None else np.column_stack([exog, ect])
        exog_names = exog_names + [f"ECT_{r+1}" for r in range(ect.shape[1])]

    pred_mat, mdl = _fit_and_forecast_linear(endog, exog, endog_names, exog_names, rows_end, p, q, lasso_alpha)

    date_to_idx = {pd.Timestamp(dt): idx for idx, dt in enumerate(y_lvl_sorted[date_col])}
    test_dates = eff_dates[rows_end:]
    gi = np.array([date_to_idx[pd.Timestamp(dt)] for dt in test_dates], dtype=int)
    lvl_mat = y_lvl_sorted[endog_names].to_numpy(float)
    lbls = endog_labels or {}

    frames = []
    for k, ecol in enumerate(endog_names):
        lvl_k = lvl_mat[:, k]
        y_pred_k = reconstruct_level(lvl_k, gi, d, pred_mat[:, k])
        y_prev_k = lvl_k[gi - 1]
        if shrinkage is not None and 0.0 < float(shrinkage) < 1.0:
            y_pred_k = y_prev_k + float(shrinkage) * (y_pred_k - y_prev_k)
        frames.append(pd.DataFrame({
            "date": test_dates,
            "province": lbls.get(ecol, ecol),
            "y_true": lvl_k[gi],
            "y_pred": y_pred_k,
            "y_prev": y_prev_k,
        }))
    return pd.concat(frames, ignore_index=True), mdl

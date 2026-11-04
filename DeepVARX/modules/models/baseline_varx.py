"""Baseline tuyến tính: VAR (không X), VARX (có X), VARX-L (VARX + phạt Lasso/Ridge).
Cài đặt kết nối trực tiếp với var_model.py và varx_model.py.
Dự báo 1 bước (rolling one-step-ahead) trên đoạn TEST, rồi quy đổi về mức giá gốc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_utils import difference, reconstruct_level
from modules.models.var_model import VARModel
from modules.models.varx_model import VARXModel


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
        mdl = VARModel(lags=p, trend="c")
        tr_df = pd.DataFrame(endog[:rows_end], columns=[target])
        te_df = pd.DataFrame(endog[rows_end:], columns=[target])
        mdl.fit(tr_df)
        pred_te = mdl.forecast_rolling(te_df, tr_df).values[:, 0]
        imp = None
        n_params = p + 1
    else:
        penalty = "lasso" if (lasso_alpha is not None and lasso_alpha > 0) else "none"
        alpha = float(lasso_alpha) if lasso_alpha is not None else 0.0
        mdl = VARXModel(p=p, q=max(1, q), penalty=penalty, alpha=alpha, trend="c", exog_lag_mode="past_only")
        tr_y = pd.DataFrame(endog[:rows_end], columns=[target])
        te_y = pd.DataFrame(endog[rows_end:], columns=[target])
        tr_x = pd.DataFrame(exog[:rows_end], columns=exog_cols)
        te_x = pd.DataFrame(exog[rows_end:], columns=exog_cols)
        mdl.fit(tr_y, tr_x)
        pred_te = mdl.forecast_rolling(te_y, te_x, tr_y, tr_x).values[:, 0]
        imp = mdl.exog_importance()
        n_params = len(mdl.feature_names) if mdl.feature_names else (p + q * len(exog_cols) + 1)

    y_pred = reconstruct_level(lvl, gi, d, pred_te)
    return {
        "index": df_levels.index[gi],
        "y_true": lvl[gi],
        "y_pred": y_pred,
        "n_params": int(n_params),
        "model": mdl,
        "exog_importance": imp
    }


def run_local_pkg(g: pd.DataFrame, date_col: str, target: str, feats: list[str], d: int, p: int, q: int,
                  lasso_alpha: float | None = None, ect_rank: int = 0, k_ar_diff: int = 2,
                  levels_input: bool = False):
    """Chạy VAR/VARX/VARX-L (hoặc VARX có số hạng hiệu chỉnh sai số - dạng VECM-X khi ect_rank > 0) cho MỘT tỉnh
    từ gói dữ liệu của tỉnh đó. g: các cột [date, split, y_level, target, feats].
    Sử dụng VARModel từ var_model.py và VARXModel từ varx_model.py.
    Trả về (DataFrame dự báo mức giá gốc VNĐ/kg, mô hình)."""
    from modules.stat_tests import johansen_beta

    g = g.sort_values(date_col).reset_index(drop=True)
    cols = [target] + list(feats)
    levels_input = levels_input or ect_rank > 0
    if levels_input:
        lev = g[cols].to_numpy(float)
        dd, off = difference(g[cols], d).to_numpy(float), d
    else:
        lev, dd, off = None, g[cols].to_numpy(float), 0

    split = g["split"].to_numpy()[off:]
    n, rows_end = len(dd), int((split == "train").sum())
    endog = dd[:, :1]
    exog = dd[:, 1:] if feats else None

    # Thêm số hạng ECT từ Johansen (nếu có yêu cầu VECM-X)
    if ect_rank > 0:
        beta = johansen_beta(lev[:rows_end + off], ect_rank, k_ar_diff)
        ect = (lev @ beta)[off:]
        exog = ect if exog is None else np.column_stack([exog, ect])

    lvl = g["y_level"].to_numpy(float)

    if exog is None or exog.shape[1] == 0:
        # Trường hợp VAR: Gọi trực tiếp VARModel từ modules/models/var_model.py
        mdl = VARModel(lags=p, trend="c")
        tr_df = pd.DataFrame(endog[:rows_end], columns=[target])
        te_df = pd.DataFrame(endog[rows_end:], columns=[target])
        mdl.fit(tr_df)
        pred_te = mdl.forecast_rolling(te_df, tr_df).values[:, 0]
        gi = np.arange(rows_end, n) + off
        y_pred = reconstruct_level(lvl, gi, d, pred_te)
    else:
        # Trường hợp VARX / VARX-L: Gọi trực tiếp VARXModel từ modules/models/varx_model.py
        penalty = "lasso" if (lasso_alpha is not None and lasso_alpha > 0) else "none"
        alpha = float(lasso_alpha) if lasso_alpha is not None else 0.0
        mdl = VARXModel(p=p, q=max(1, q), penalty=penalty, alpha=alpha, trend="c", exog_lag_mode="past_only")
        
        exog_names = [f"X_{k+1}" for k in range(exog.shape[1])]
        tr_y = pd.DataFrame(endog[:rows_end], columns=[target])
        te_y = pd.DataFrame(endog[rows_end:], columns=[target])
        tr_x = pd.DataFrame(exog[:rows_end], columns=exog_names)
        te_x = pd.DataFrame(exog[rows_end:], columns=exog_names)

        mdl.fit(tr_y, tr_x)
        pred_te = mdl.forecast_rolling(te_y, te_x, tr_y, tr_x).values[:, 0]
        gi = np.arange(rows_end, n) + off
        y_pred = reconstruct_level(lvl, gi, d, pred_te)

    out = pd.DataFrame({
        "date": g[date_col].to_numpy()[gi],
        "y_true": lvl[gi],
        "y_pred": y_pred,
        "y_prev": lvl[gi - 1]
    })
    return out, mdl

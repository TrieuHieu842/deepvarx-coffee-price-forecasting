"""Baseline tuyến tính: VAR (không X), VARX (có X), VARX-L (VARX + phạt Lasso).
Cài đặt bằng OLS/Lasso để chạy nhanh và minh bạch.
Dự báo 1 bước (rolling one-step-ahead) trên đoạn TEST, rồi quy đổi về mức giá gốc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_utils import difference, reconstruct_level


def _design(endog: np.ndarray, exog: np.ndarray | None, p: int, q: int, m: int):
    T = endog.shape[0]
    cols = [endog[m - k:T - k] for k in range(1, p + 1)]
    if exog is not None and q > 0:
        cols += [exog[m - k:T - k] for k in range(1, q + 1)]
    return np.hstack(cols), endog[m:]


class VARXModel:
    def __init__(self, p: int, q: int, lasso_alpha: float | None = None):
        self.p, self.q, self.lasso_alpha = p, q, lasso_alpha

    def fit(self, endog: np.ndarray, exog: np.ndarray | None, rows_end: int):
        """Khớp trên các dòng t ∈ [m, rows_end) (chỉ TRAIN)."""
        self.m = max(self.p, self.q if exog is not None else 0)
        X, Y = _design(endog[:rows_end], exog[:rows_end] if exog is not None else None, self.p, self.q, self.m)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Xs = (X - self.mu) / self.sd
        if self.lasso_alpha:
            from sklearn.linear_model import Lasso

            coefs, ints = [], []
            for j in range(Y.shape[1]):
                las = Lasso(alpha=self.lasso_alpha, max_iter=20000).fit(Xs, Y[:, j])
                coefs.append(las.coef_)
                ints.append(las.intercept_)
            self.coef, self.intercept = np.array(coefs).T, np.array(ints)
        else:
            Z = np.column_stack([np.ones(len(Xs)), Xs])
            beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
            self.intercept, self.coef = beta[0], beta[1:]
        self.n_endog = endog.shape[1]
        self.n_exog = 0 if exog is None else exog.shape[1]
        self.resid = Y - (Xs @ self.coef + self.intercept)
        return self

    def predict(self, endog: np.ndarray, exog: np.ndarray | None) -> np.ndarray:
        """Dự báo 1 bước cho mọi dòng t ∈ [m, T)."""
        X, _ = _design(endog, exog, self.p, self.q, self.m)
        return ((X - self.mu) / self.sd) @ self.coef + self.intercept

    def exog_importance(self) -> pd.Series | None:
        """Tổng |hệ số chuẩn hoá| của từng biến X trên mọi lag (đối với endog đầu tiên)."""
        if self.n_exog == 0:
            return None
        ne, nx = self.n_endog, self.n_exog
        c = np.abs(self.coef[:, 0])[ne * self.p:]
        return pd.Series(c.reshape(self.q, nx).sum(0))


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
    mdl = VARXModel(p, q if exog_cols else 0, lasso_alpha).fit(endog, exog, rows_end)
    pred_all = mdl.predict(endog, exog)[:, 0]
    rows = np.arange(mdl.m, len(endog))
    keep = rows >= rows_end
    t_idx = rows[keep] + d
    level = df_levels[target].to_numpy(float)
    y_pred = reconstruct_level(level, t_idx, d, pred_all[keep])
    return {"index": df_levels.index[t_idx], "y_true": level[t_idx], "y_pred": y_pred,
            "n_params": int(mdl.coef.size + 1), "model": mdl,
            "exog_importance": None if not exog_cols else pd.Series(mdl.exog_importance().to_numpy(), index=exog_cols)}


def run_local_pkg(g: pd.DataFrame, date_col: str, target: str, feats: list[str], d: int, p: int, q: int,
                  lasso_alpha: float | None = None, ect_rank: int = 0, k_ar_diff: int = 2,
                  levels_input: bool = False):
    """Chạy VAR/VARX/VARX-L (hoặc VARX có số hạng hiệu chỉnh sai số - dạng VECM-X khi ect_rank > 0) cho MỘT tỉnh
    từ gói dữ liệu của tỉnh đó. g: các cột [date, split, y_level, target, feats]. Gói dạng sai phân (ect_rank = 0)
    hoặc dạng mức gốc (levels_input=True; bắt buộc khi ect_rank > 0 để tính ECT từ Johansen). Trả về (DataFrame dự báo mức giá gốc, mô hình)."""
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
    if ect_rank > 0:
        beta = johansen_beta(lev[:rows_end + off], ect_rank, k_ar_diff)
        ect = (lev @ beta)[off:]
        exog = ect if exog is None else np.column_stack([exog, ect])
    mdl = VARXModel(p, q if exog is not None else 0, lasso_alpha).fit(endog, exog, rows_end)
    pred_all = mdl.predict(endog, exog)[:, 0]
    rows = np.arange(mdl.m, n)
    gi = rows[rows >= rows_end] + off
    lvl = g["y_level"].to_numpy(float)
    y_pred = reconstruct_level(lvl, gi, d, pred_all[rows >= rows_end])
    out = pd.DataFrame({"date": g[date_col].to_numpy()[gi], "y_true": lvl[gi], "y_pred": y_pred, "y_prev": lvl[gi - 1]})
    return out, mdl

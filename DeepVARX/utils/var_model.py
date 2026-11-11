# -*- coding: utf-8 -*-
"""
========================================================================================
MÔ HÌNH VECTOR AUTOREGRESSION (VAR) ĐA BIẾN DỰ BÁO GIÁ CÀ PHÊ ROBUSTA VIỆT NAM
(Đắk Lắk, Lâm Đồng, Gia Lai) - TÍCH HỢP HỆ THỐNG WEB VÀ THỰC NGHIỆM ĐỘC LẬP
========================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

TARGET_PROVINCES: List[str] = ["dak_lak", "lam_dong", "gia_lai"]


@dataclass
class CoffeeDataBundle:
    raw_df: pd.DataFrame
    price_df: pd.DataFrame
    ret_df: pd.DataFrame
    dates: pd.DatetimeIndex
    train_end: int
    val_end: int


def load_and_prepare_coffee_data(
    csv_path: str = "data_daily_panel_clean_v2.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> CoffeeDataBundle:
    raw_df = pd.read_csv(csv_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    price_raw = (
        raw_df[raw_df["province_slug"].isin(TARGET_PROVINCES)]
        .pivot(index="date", columns="province_slug", values="price_vnd_kg")[TARGET_PROVINCES]
        .dropna()
        .sort_index()
    )

    ret_df = np.log(price_raw).diff().dropna()
    dates = ret_df.index
    price_df = price_raw.loc[dates]

    n_total = len(ret_df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    return CoffeeDataBundle(
        raw_df=raw_df,
        price_df=price_df,
        ret_df=ret_df,
        dates=dates,
        train_end=train_end,
        val_end=val_end,
    )


def compute_metrics(
    true_price: np.ndarray,
    pred_price: np.ndarray,
    true_ret: np.ndarray,
    pred_ret: np.ndarray,
) -> Dict[str, float]:
    p_true = np.asarray(true_price, dtype=np.float64)
    p_pred = np.asarray(pred_price, dtype=np.float64)
    r_true = np.asarray(true_ret, dtype=np.float64)
    r_pred = np.asarray(pred_ret, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((p_true - p_pred) ** 2)))
    mae = float(np.mean(np.abs(p_true - p_pred)))
    mape = float(np.mean(np.abs((p_true - p_pred) / np.maximum(p_true, 1e-6)))) * 100.0

    ss_tot_p = float(np.sum((p_true - np.mean(p_true)) ** 2))
    ss_res_p = float(np.sum((p_true - p_pred) ** 2))
    r2_price = 1.0 - (ss_res_p / ss_tot_p) if ss_tot_p > 1e-12 else 0.0

    ss_tot_r = float(np.sum((r_true - np.mean(r_true)) ** 2))
    ss_res_r = float(np.sum((r_true - r_pred) ** 2))
    r2_ret = 1.0 - (ss_res_r / ss_tot_r) if ss_tot_r > 1e-12 else 0.0

    rmse_ret = float(np.sqrt(np.mean((r_true - r_pred) ** 2)))

    active = (r_true != 0.0) & (r_pred != 0.0)
    if np.sum(active) > 0:
        da = float(np.mean(np.sign(r_true[active]) == np.sign(r_pred[active]))) * 100.0
    else:
        da = 0.0

    return {
        "RMSE (VND)": rmse,
        "MAE (VND)": mae,
        "MAPE (%)": mape,
        "R2_price": r2_price,
        "R2_ret": r2_ret,
        "DA (%)": da,
        "RMSE_ret": rmse_ret,
    }


def diebold_mariano_from_loss_diff(
    d: np.ndarray,
    h: int = 1,
    max_lag: Optional[int] = None,
    alternative: str = "greater",
) -> Tuple[float, float]:
    d_arr = np.asarray(d, dtype=np.float64)
    n = len(d_arr)
    if n < 2:
        return 0.0, 1.0

    mean_d = float(np.mean(d_arr))
    dc = d_arr - mean_d
    gamma0 = float(np.mean(dc ** 2))

    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    max_lag = max(h - 1, min(max_lag, n - 1))

    lrv = gamma0
    for k in range(1, max_lag + 1):
        gamma_k = float(np.dot(dc[k:], dc[:-k])) / n
        weight = 1.0 - k / (max_lag + 1.0)
        lrv += 2.0 * weight * gamma_k

    if lrv <= 1e-14:
        se = float(np.sqrt(gamma0 / n)) if gamma0 > 1e-14 else 1e-7
    else:
        se = float(np.sqrt(lrv / n))

    dm_stat = mean_d / se
    hln_factor = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-6))
    dm_stat = float(dm_stat * hln_factor)
    df = n - 1

    if alternative == "greater":
        p_val = float(1.0 - stats.t.cdf(dm_stat, df=df))
    elif alternative == "two-sided":
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_stat), df=df)))
    else:
        p_val = float(stats.t.cdf(dm_stat, df=df))

    return dm_stat, p_val


class VARModel:
    """
    Mô hình Vector Autoregression (VAR) đa biến / đơn biến tối ưu hóa bằng NumPy slicing.
    Y_t = C + A1*Y_(t-1) + ... + Ap*Y_(t-p) + e_t
    """

    def __init__(self, p: int = 5, lags: Optional[int] = None, ic: Optional[str] = None, trend: str = "c", **kwargs):
        self.p = int(lags if lags is not None else p)
        self.lags = self.p
        self.ic = ic
        self.trend = str(trend).lower()
        fit_intercept = self.trend not in ("nc", "n", "none")
        self.reg = LinearRegression(fit_intercept=fit_intercept)
        self.is_fitted = False
        self.K = len(TARGET_PROVINCES)
        self.endog_names: List[str] = TARGET_PROVINCES.copy()
        self.coef_matrix: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def _build_lags(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(Y)
        X = np.hstack([Y[self.p - k : T - k] for k in range(1, self.p + 1)])
        target = Y[self.p :]
        return X, target

    def fit(self, train_df: Union[pd.DataFrame, pd.Series, np.ndarray]) -> "VARModel":
        if isinstance(train_df, pd.Series):
            df = train_df.to_frame()
            self.endog_names = list(df.columns)
            Y = df.values.astype(np.float64)
        elif isinstance(train_df, pd.DataFrame):
            self.endog_names = list(train_df.columns)
            Y = train_df.values.astype(np.float64)
        else:
            Y = np.asarray(train_df, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            self.endog_names = [f"Y{i+1}" for i in range(Y.shape[1])]

        self.K = Y.shape[1]
        X_tr, y_tr = self._build_lags(Y)
        self.reg.fit(X_tr, y_tr)
        self.is_fitted = True

        coef = self.reg.coef_
        self.coef_matrix = coef.T if coef.ndim == 2 else coef.reshape(-1, 1)
        self.intercept_ = np.atleast_1d(self.reg.intercept_) if self.reg.fit_intercept else np.zeros(self.K)

        self.feature_names = []
        if self.reg.fit_intercept:
            self.feature_names.append("const")
        for lag in range(1, self.p + 1):
            for col in self.endog_names:
                self.feature_names.append(f"{col}_lag{lag}")

        return self

    def forecast(self, history_df: Union[pd.DataFrame, pd.Series, np.ndarray], steps: int = 1) -> pd.DataFrame:
        """Dự báo đệ quy `steps` bước tiếp theo từ lịch sử gần nhất."""
        if not self.is_fitted:
            raise ValueError("Mô hình VAR chưa được huấn luyện.")
        if isinstance(history_df, (pd.Series, pd.DataFrame)):
            hist = history_df.values.astype(np.float64)
        else:
            hist = np.asarray(history_df, dtype=np.float64)
        if hist.ndim == 1:
            hist = hist.reshape(-1, 1)
        if len(hist) < self.p:
            raise ValueError(f"Dữ liệu lịch sử ({len(hist)}) nhỏ hơn bậc trễ p={self.p}.")

        curr = list(hist[-self.p:])
        preds = []
        for _ in range(steps):
            x_step = np.hstack([curr[-k] for k in range(1, self.p + 1)]).reshape(1, -1)
            pred = self.reg.predict(x_step)[0]
            preds.append(pred)
            curr.append(pred)
        return pd.DataFrame(preds, columns=self.endog_names)

    def forecast_rolling(
        self,
        test_df: Union[pd.DataFrame, pd.Series, np.ndarray],
        history_df: Union[pd.DataFrame, pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """Dự báo rolling 1 bước vector hóa (tối ưu hóa ma trận) tương thích API hệ thống."""
        if not self.is_fitted:
            raise ValueError("Mô hình VAR chưa được huấn luyện.")

        if isinstance(test_df, (pd.Series, pd.DataFrame)):
            test_idx = test_df.index
            te_arr = test_df.values.astype(np.float64)
            if te_arr.ndim == 1:
                te_arr = te_arr.reshape(-1, 1)
        else:
            te_arr = np.asarray(test_df, dtype=np.float64)
            if te_arr.ndim == 1:
                te_arr = te_arr.reshape(-1, 1)
            test_idx = pd.RangeIndex(len(te_arr))

        if isinstance(history_df, (pd.Series, pd.DataFrame)):
            hist_arr = history_df.values.astype(np.float64)
            if hist_arr.ndim == 1:
                hist_arr = hist_arr.reshape(-1, 1)
        else:
            hist_arr = np.asarray(history_df, dtype=np.float64)
            if hist_arr.ndim == 1:
                hist_arr = hist_arr.reshape(-1, 1)

        y_all = np.vstack([hist_arr, te_arr])
        n_hist, n_test = len(hist_arr), len(te_arr)

        if n_hist < self.p:
            raise ValueError(f"Dữ liệu lịch sử ({n_hist}) nhỏ hơn bậc trễ p={self.p}.")

        cols = [y_all[n_hist - k : n_hist + n_test - k, :] for k in range(1, self.p + 1)]
        Z_test = np.hstack(cols)
        preds = self.reg.predict(Z_test)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return pd.DataFrame(preds, columns=self.endog_names, index=test_idx)

    def predict_rolling(self, Y_full: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """Dự báo rolling 1 bước truyền thống nhận mảng chỉ số."""
        preds = []
        for t in test_indices:
            x_t = np.hstack([Y_full[t - k] for k in range(1, self.p + 1)]).reshape(1, -1)
            pred_t = self.reg.predict(x_t)[0]
            preds.append(pred_t)
        return np.array(preds)

    @property
    def selected_lag(self) -> int:
        return self.p

    @property
    def is_stable(self) -> bool:
        if not self.is_fitted:
            return False
        K, p = self.K, self.p
        coef = self.reg.coef_
        A = coef if coef.ndim == 2 else coef.reshape(1, -1)
        if A.shape != (K, K * p):
            return True
        companion = np.zeros((K * p, K * p), dtype=np.float64)
        companion[:K, :] = A
        if p > 1:
            companion[K:, : K * (p - 1)] = np.eye(K * (p - 1))
        eigvals = np.linalg.eigvals(companion)
        return bool(np.all(np.abs(eigvals) < 1.0))


# Alias tương thích hai chiều
CoffeeVAR = VARModel


def select_var_order_aic_bic(Y_train: np.ndarray, max_p: int = 10) -> pd.DataFrame:
    T, K = Y_train.shape
    rows = []
    for p in range(1, max_p + 1):
        X = np.hstack([Y_train[p - k : T - k] for k in range(1, p + 1)])
        y = Y_train[p:]
        N = len(y)
        reg = LinearRegression().fit(X, y)
        res = y - reg.predict(X)
        sigma = np.atleast_2d(np.cov(res.T, bias=True))
        log_det = np.log(np.maximum(np.linalg.det(sigma), 1e-300))
        k_params = K * (1 + K * p)
        aic = N * log_det + 2 * k_params
        bic = N * log_det + k_params * np.log(N)
        rows.append({"p": p, "k_params": k_params, "AIC": aic, "BIC": bic})
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)

# -*- coding: utf-8 -*-
"""
========================================================================================
MÔ HÌNH VECTOR AUTOREGRESSION WITH EXOGENOUS VARIABLES (VARX & VARX-L)
Dự báo đa biến giá thu mua cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
TÍCH HỢP HỆ THỐNG WEB VÀ THỰC NGHIỆM ĐỘC LẬP
========================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLasso, Ridge

TARGET_PROVINCES: List[str] = ["dak_lak", "lam_dong", "gia_lai"]

CORE_WEATHER_COLS: List[str] = ["temp_mean", "precip_nasa", "humidity_air"]
EXT_WEATHER_COLS: List[str] = ["d_soil_moisture", "solar_radiation", "wind_speed_2m"]
CORE_ECON_LEVEL_COLS: List[str] = [
    "london_price_usd_ton",
    "arabica_price_usd_lb",
    "brent_price_usd_barrel",
    "dxy_index",
    "usd_vnd_rate",
]
EXT_ECON_LEVEL_COLS: List[str] = ["usd_brl_rate"]
CALENDAR_FRICTION_COLS: List[str] = [
    "sin_day_of_year",
    "cos_day_of_year",
    "is_harvest_season_vn",
    "is_harvest_season_brazil",
    "exog_stale_days",
]


@dataclass
class CoffeeDataBundle:
    raw_df: pd.DataFrame
    price_df: pd.DataFrame
    ret_df: pd.DataFrame
    dates: pd.DatetimeIndex
    train_end: int
    val_end: int
    feature_sets: Dict[str, pd.DataFrame]


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

    all_econ_level_cols = CORE_ECON_LEVEL_COLS + EXT_ECON_LEVEL_COLS
    econ_level_df = raw_df.groupby("date")[all_econ_level_cols].mean().loc[dates]
    econ_ret_df = np.log(econ_level_df).diff().fillna(0.0)
    econ_ret_df.columns = [
        "ret_london",
        "ret_arabica",
        "ret_brent",
        "ret_dxy",
        "ret_usd_vnd",
        "ret_usd_brl",
    ]

    raw_weather_cols = [
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_nasa",
        "soil_moisture_top",
        "humidity_air",
        "solar_radiation",
        "wind_speed_2m",
    ]
    weather_mean_df = raw_df.groupby("date")[raw_weather_cols].mean().loc[dates]
    weather_mean_df["d_soil_moisture"] = weather_mean_df["soil_moisture_top"].diff().fillna(0.0)

    cal_df = raw_df.groupby("date")[CALENDAR_FRICTION_COLS].mean().loc[dates]

    x_v0 = ret_df.copy()
    x_v1 = pd.concat([ret_df, econ_ret_df[["ret_london", "ret_usd_vnd"]]], axis=1)
    x_v2 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS],
            econ_ret_df[["ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd"]],
        ],
        axis=1,
    )
    x_v3 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS + EXT_WEATHER_COLS],
            econ_ret_df,
            cal_df,
        ],
        axis=1,
    )

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
        feature_sets={
            "VAR (No X)": x_v0,
            "VARX 1": x_v1,
            "VARX 2": x_v2,
            "VARX 3": x_v3,
        },
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


class VARXModel:
    """
    Mô hình Vector Autoregression with Exogenous Variables (VARX & VARX-L) tối ưu bằng NumPy.
    Y_t = C + A1*Y_(t-1) + ... + Ap*Y_(t-p) + B1*X_(t-1) + ... + Bq*X_(t-q) + e_t
    """

    def __init__(
        self,
        p: int = 5,
        q: int = 2,
        penalty: str = "none",
        alpha: float = 0.1,
        trend: str = "c",
        exog_lag_mode: str = "past_only",
        **kwargs,
    ):
        self.p = int(p)
        self.q = int(q)
        self.penalty = str(penalty).lower()
        self.alpha = float(alpha)
        self.trend = str(trend).lower()
        self.exog_lag_mode = str(exog_lag_mode).lower()
        self.K = len(TARGET_PROVINCES)
        self.endog_columns: List[str] = TARGET_PROVINCES.copy()
        self.exog_columns: List[str] = []
        self.feature_names: List[str] = []
        self.fit_intercept = self.trend not in ("nc", "n", "none")
        self.is_fitted = False
        self.coef_matrix: Optional[np.ndarray] = None
        self.max_lag = self.p
        self.reg = None
        self.r2_: Dict[str, float] = {}
        self.mse_: Dict[str, float] = {}

    def _build_features(self, Y: np.ndarray, X: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
        T = len(Y)
        has_x = X is not None and X.shape[1] > 0
        max_lag = max(self.p, self.q) if has_x else self.p
        parts = [Y[max_lag - k : T - k] for k in range(1, self.p + 1)]
        if has_x:
            parts.extend([X[max_lag - k : T - k] for k in range(1, self.q + 1)])
        Z = np.hstack(parts)
        target = Y[max_lag:]
        return Z, target, max_lag

    def fit(
        self,
        Y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        X_train: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
    ) -> "VARXModel":
        if isinstance(Y_train, pd.Series):
            df_y = Y_train.to_frame()
            self.endog_columns = list(df_y.columns)
            Y = df_y.values.astype(np.float64)
        elif isinstance(Y_train, pd.DataFrame):
            self.endog_columns = list(Y_train.columns)
            Y = Y_train.values.astype(np.float64)
        else:
            Y = np.asarray(Y_train, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            self.endog_columns = [f"Y{i+1}" for i in range(Y.shape[1])]

        self.K = Y.shape[1]

        if X_train is not None:
            if isinstance(X_train, pd.Series):
                df_x = X_train.to_frame()
                self.exog_columns = list(df_x.columns)
                X = df_x.values.astype(np.float64)
            elif isinstance(X_train, pd.DataFrame):
                self.exog_columns = list(X_train.columns)
                X = X_train.values.astype(np.float64)
            else:
                X = np.asarray(X_train, dtype=np.float64)
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                self.exog_columns = [f"X{i+1}" for i in range(X.shape[1])]
        else:
            self.exog_columns = []
            X = None

        Z_tr, y_tr, self.max_lag = self._build_features(Y, X)

        feat_names = []
        if self.fit_intercept:
            feat_names.append("const")
        for lag in range(1, self.p + 1):
            for col in self.endog_columns:
                feat_names.append(f"{col}_lag{lag}")
        if X is not None and X.shape[1] > 0:
            for lag in range(1, self.q + 1):
                for col in self.exog_columns:
                    feat_names.append(f"{col}_lag{lag}")
        self.feature_names = feat_names

        if self.penalty in ("ridge", "l2"):
            reg_alpha = self.alpha if self.alpha > 0 else 0.1
            self.reg = Ridge(alpha=reg_alpha, fit_intercept=self.fit_intercept)
            self.reg.fit(Z_tr, y_tr)
        elif self.penalty in ("lasso", "l1"):
            reg_alpha = self.alpha if self.alpha > 0 else 0.001
            if y_tr.shape[1] == 1:
                self.reg = Lasso(alpha=reg_alpha, fit_intercept=self.fit_intercept, max_iter=10000)
                self.reg.fit(Z_tr, y_tr.ravel())
            else:
                self.reg = MultiTaskLasso(alpha=reg_alpha, fit_intercept=self.fit_intercept, max_iter=10000)
                self.reg.fit(Z_tr, y_tr)
        else:
            self.reg = LinearRegression(fit_intercept=self.fit_intercept)
            self.reg.fit(Z_tr, y_tr)

        coef = self.reg.coef_
        if coef.ndim == 1:
            self.coef_matrix = coef.reshape(-1, 1)
        else:
            self.coef_matrix = coef.T

        # Đánh giá R2 trên tập huấn luyện
        preds_tr = self._predict(Z_tr)
        res = y_tr - preds_tr
        ss_res = np.sum(res ** 2, axis=0)
        ss_tot = np.sum((y_tr - np.mean(y_tr, axis=0)) ** 2, axis=0)
        ss_tot = np.where(ss_tot == 0, 1e-8, ss_tot)
        r2_vals = 1.0 - (ss_res / ss_tot)
        mse_vals = np.mean(res ** 2, axis=0)
        for i, col in enumerate(self.endog_columns):
            self.r2_[col] = float(r2_vals[i])
            self.mse_[col] = float(mse_vals[i])

        self.is_fitted = True
        return self

    def _predict(self, Z: np.ndarray) -> np.ndarray:
        preds = self.reg.predict(Z)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def forecast(
        self,
        history_endog: Union[pd.DataFrame, pd.Series, np.ndarray],
        history_exog: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        exog_future: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        steps: int = 1,
    ) -> pd.DataFrame:
        """Dự báo `steps` bước tiếp theo từ lịch sử gần nhất."""
        if not self.is_fitted:
            raise ValueError("Mô hình VARX chưa được huấn luyện.")
        if isinstance(history_endog, (pd.Series, pd.DataFrame)):
            hist_y = history_endog.values.astype(np.float64)
        else:
            hist_y = np.asarray(history_endog, dtype=np.float64)
        if hist_y.ndim == 1:
            hist_y = hist_y.reshape(-1, 1)

        curr_y = list(hist_y[-self.p:])
        if history_exog is not None and len(self.exog_columns) > 0:
            if isinstance(history_exog, (pd.Series, pd.DataFrame)):
                hist_x = history_exog.values.astype(np.float64)
            else:
                hist_x = np.asarray(history_exog, dtype=np.float64)
            if hist_x.ndim == 1:
                hist_x = hist_x.reshape(-1, 1)

            if exog_future is not None:
                if isinstance(exog_future, (pd.Series, pd.DataFrame)):
                    fut_x = exog_future.values.astype(np.float64)
                else:
                    fut_x = np.asarray(exog_future, dtype=np.float64)
                if fut_x.ndim == 1:
                    fut_x = fut_x.reshape(-1, 1)
                x_pool = np.vstack([hist_x, fut_x])
            else:
                x_pool = hist_x
        else:
            x_pool = None

        preds = []
        t_start = len(hist_y)
        for h in range(steps):
            parts = [curr_y[-k] for k in range(1, self.p + 1)]
            if x_pool is not None and len(x_pool) > 0 and self.q > 0:
                for k in range(1, self.q + 1):
                    idx = max(0, min(len(x_pool) - 1, t_start + h - k))
                    parts.append(x_pool[idx])
            z = np.hstack(parts).reshape(1, -1)
            pred = self._predict(z)[0]
            preds.append(pred)
            curr_y.append(pred)

        return pd.DataFrame(preds, columns=self.endog_columns)

    def forecast_rolling(
        self,
        test_endog: Union[pd.DataFrame, pd.Series, np.ndarray],
        test_exog: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]],
        history_endog: Union[pd.DataFrame, pd.Series, np.ndarray],
        history_exog: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]],
    ) -> pd.DataFrame:
        """Dự báo rolling 1 bước vector hóa (tối ưu hóa ma trận) tương thích API hệ thống."""
        if not self.is_fitted:
            raise ValueError("Mô hình VARX chưa được huấn luyện.")

        if isinstance(test_endog, (pd.Series, pd.DataFrame)):
            test_idx = test_endog.index
            te_y = test_endog.values.astype(np.float64)
            if te_y.ndim == 1:
                te_y = te_y.reshape(-1, 1)
        else:
            te_y = np.asarray(test_endog, dtype=np.float64)
            if te_y.ndim == 1:
                te_y = te_y.reshape(-1, 1)
            test_idx = pd.RangeIndex(len(te_y))

        if isinstance(history_endog, (pd.Series, pd.DataFrame)):
            hist_y = history_endog.values.astype(np.float64)
            if hist_y.ndim == 1:
                hist_y = hist_y.reshape(-1, 1)
        else:
            hist_y = np.asarray(history_endog, dtype=np.float64)
            if hist_y.ndim == 1:
                hist_y = hist_y.reshape(-1, 1)

        y_all = np.vstack([hist_y, te_y])
        n_hist, n_test = len(hist_y), len(te_y)

        parts = [y_all[n_hist - k : n_hist + n_test - k, :] for k in range(1, self.p + 1)]

        if test_exog is not None and history_exog is not None and len(self.exog_columns) > 0:
            if isinstance(test_exog, (pd.Series, pd.DataFrame)):
                te_x = test_exog.values.astype(np.float64)
            else:
                te_x = np.asarray(test_exog, dtype=np.float64)
            if te_x.ndim == 1:
                te_x = te_x.reshape(-1, 1)

            if isinstance(history_exog, (pd.Series, pd.DataFrame)):
                hist_x = history_exog.values.astype(np.float64)
            else:
                hist_x = np.asarray(history_exog, dtype=np.float64)
            if hist_x.ndim == 1:
                hist_x = hist_x.reshape(-1, 1)

            x_all = np.vstack([hist_x, te_x])
            parts.extend([x_all[n_hist - k : n_hist + n_test - k, :] for k in range(1, self.q + 1)])

        Z_test = np.hstack(parts)
        preds = self._predict(Z_test)
        return pd.DataFrame(preds, columns=self.endog_columns, index=test_idx)

    def predict_rolling(
        self, Y_full: np.ndarray, X_full: Optional[np.ndarray], test_indices: np.ndarray
    ) -> np.ndarray:
        """Dự báo rolling 1 bước truyền thống nhận mảng chỉ số."""
        preds = []
        for t in test_indices:
            parts = [Y_full[t - k] for k in range(1, self.p + 1)]
            if X_full is not None and X_full.shape[1] > 0:
                parts.extend([X_full[t - k] for k in range(1, self.q + 1)])
            z_t = np.hstack(parts).reshape(1, -1)
            pred_t = self._predict(z_t)[0]
            preds.append(pred_t)
        return np.array(preds)

    def exog_importance(self) -> Optional[pd.Series]:
        """Tính tổng độ lớn tuyệt đối hệ số của từng biến ngoại sinh qua mọi bước trễ."""
        if not self.exog_columns or not self.is_fitted or self.coef_matrix is None:
            return None
        c0 = np.abs(self.coef_matrix[:, 0])
        pure_feats = [f for f in self.feature_names if f != "const"]
        scores = {}
        for col in self.exog_columns:
            tot = 0.0
            for idx, feat in enumerate(pure_feats):
                if feat.startswith(f"{col}_lag"):
                    tot += float(c0[idx])
            scores[col] = tot
        return pd.Series(scores)

    def summary(self) -> Dict[str, Union[str, int, float, dict]]:
        """Trả về tóm tắt thông số cấu hình mô hình."""
        return {
            "model": "VARX",
            "p_lag": self.p,
            "q_lag": self.q,
            "penalty": self.penalty,
            "alpha": self.alpha if self.penalty in ("ridge", "lasso") else 0.0,
            "trend": self.trend,
            "exog_lag_mode": self.exog_lag_mode,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "r2": self.r2_,
            "mse": self.mse_,
        }

    @property
    def selected_lag(self) -> int:
        return self.p


# Alias tương thích hai chiều
CoffeeVARX = VARXModel


def compute_varx_permutation_importance(
    varx_mdl: VARXModel,
    Y_full: np.ndarray,
    X_full: np.ndarray,
    exog_names: List[str],
    te_idx: np.ndarray,
    base_pred_ret: np.ndarray,
    true_ret: np.ndarray,
    p_prev: np.ndarray,
    p_true: np.ndarray,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    base_m = compute_metrics(p_true, p_prev * np.exp(base_pred_ret), true_ret, base_pred_ret)
    base_rmse = base_m["RMSE (VND)"]

    rows = []
    for col_idx, col_name in enumerate(exog_names):
        rmse_increases = []
        for _ in range(n_repeats):
            X_perm = X_full.copy()
            perm_order = rng.permutation(len(X_perm))
            X_perm[:, col_idx] = X_perm[perm_order, col_idx]
            pred_perm_ret = varx_mdl.predict_rolling(Y_full, X_perm, te_idx)
            pred_perm_p = p_prev * np.exp(pred_perm_ret)
            m_perm = compute_metrics(p_true, pred_perm_p, true_ret, pred_perm_ret)
            rmse_increases.append(m_perm["RMSE (VND)"] - base_rmse)

        mean_inc = float(np.mean(rmse_increases))
        std_inc = float(np.std(rmse_increases))
        rows.append({
            "Feature": col_name,
            "Mean_RMSE_Increase": mean_inc,
            "Std_RMSE_Increase": std_inc,
        })

    imp_df = pd.DataFrame(rows).sort_values("Mean_RMSE_Increase", ascending=False).reset_index(drop=True)
    tot = float(np.sum(np.maximum(imp_df["Mean_RMSE_Increase"], 0.0)))
    imp_df["Importance_Share (%)"] = [
        round(max(v, 0.0) / tot * 100.0, 2) if tot > 0 else 0.0 for v in imp_df["Mean_RMSE_Increase"]
    ]
    return imp_df

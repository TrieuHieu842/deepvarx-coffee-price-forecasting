"""Các kiểm định thống kê dùng cho Luồng 1 (tuyến tính) và Luồng 2 (phi tuyến).
Phần dùng numpy/scipy thuần: CCF, Granger-HC1, Ljung-Box, Variance Ratio.
Phần cần statsmodels (import trễ): ADF, KPSS, Johansen, BDS."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------- AR & tiền trắng hoá
def fit_ar(x: np.ndarray, max_p: int = 10) -> tuple[np.ndarray, int]:
    """Khớp AR(p) bằng OLS, chọn p theo AIC trên mẫu chung. Trả về (phi, p)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    max_p = int(min(max_p, max(1, n // 5)))
    best = (np.inf, np.array([]), 0)
    for p in range(0, max_p + 1):
        y = x[max_p:]
        if p == 0:
            rss, phi = float(np.sum(y ** 2)), np.array([])
        else:
            X = np.column_stack([x[max_p - k:n - k] for k in range(1, p + 1)])
            phi, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(np.sum((y - X @ phi) ** 2))
        aic = len(y) * np.log(rss / len(y) + 1e-300) + 2 * (p + 1)
        if aic < best[0]:
            best = (aic, phi, p)
    return best[1], best[2]


def apply_ar_filter(x: np.ndarray, phi: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    p = len(phi)
    if p == 0:
        return x
    n = len(x)
    return x[p:] - sum(phi[k - 1] * x[p - k:n - k] for k in range(1, p + 1))


def ar_residuals(x: np.ndarray, max_p: int = 10) -> tuple[np.ndarray, int]:
    phi, p = fit_ar(x, max_p)
    return apply_ar_filter(x, phi), p


# ----------------------------------------------------------------------------- CCF tiền trắng hoá
def ccf_prewhitened(x: np.ndarray, y: np.ndarray, max_lag: int = 20, alpha: float = 0.05,
                    ar_max: int = 10) -> tuple[pd.DataFrame, int]:
    """CCF giữa X (đã tiền trắng hoá) và Y (lọc bằng CHÍNH bộ lọc AR của X).
    lag k > 0: X_{t-k} dẫn trước Y_t. Ngưỡng hiệu chuẩn Bonferroni cho các lag k = 1..max_lag."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    phi, p = fit_ar(x, ar_max)
    xw, yw = apply_ar_filter(x, phi), apply_ar_filter(y, phi)
    n = len(xw)
    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = xw[:n - k] if k else xw, yw[k:]
        else:
            a, b = xw[-k:], yw[:n + k]
        r = np.corrcoef(a, b)[0, 1] if len(a) > 3 and a.std() > 0 and b.std() > 0 else np.nan
        rows.append((k, r))
    out = pd.DataFrame(rows, columns=["lag", "ccf"])
    m = max(1, max_lag)
    out["threshold"] = stats.norm.ppf(1 - alpha / (2 * m)) / np.sqrt(n)
    out["significant"] = out["ccf"].abs() > out["threshold"]
    return out, p


def summarize_ccf(tab: pd.DataFrame) -> dict:
    lead = tab[tab["lag"] >= 1]
    if lead.empty or lead["ccf"].isna().all():
        return {"best_lag": np.nan, "ccf": np.nan, "threshold": np.nan, "n_sig": 0, "pass": False}
    i = lead["ccf"].abs().idxmax()
    sig = lead[lead["significant"]]
    return {"best_lag": int(lead.loc[i, "lag"]), "ccf": float(lead.loc[i, "ccf"]),
            "threshold": float(lead.loc[i, "threshold"]), "n_sig": int(len(sig)), "pass": bool(len(sig) > 0)}


# ----------------------------------------------------------------------------- Granger (Wald, HC1, Bonferroni)
def _lag_mat(a: np.ndarray, p: int) -> np.ndarray:
    n = len(a)
    return np.column_stack([a[p - k:n - k] for k in range(1, p + 1)])


def _hc1_wald(Y: np.ndarray, X: np.ndarray, test_idx: list[int]) -> tuple[float, float]:
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ Y
    e = Y - X @ beta
    xe = X * e[:, None]
    cov = n / (n - k) * xtx_inv @ (xe.T @ xe) @ xtx_inv
    b = beta[test_idx]
    vb = cov[np.ix_(test_idx, test_idx)]
    chi2 = float(b @ np.linalg.pinv(vb) @ b)
    q = len(test_idx)
    F = chi2 / q
    return F, float(stats.f.sf(F, q, n - k))


def granger_hc1(y: np.ndarray, x: np.ndarray, max_lag: int = 10) -> pd.DataFrame:
    """H0: X không giúp dự báo Y. Wald F-test với sai số chuẩn HC1, p-value hiệu chỉnh Bonferroni theo số lag."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    rows = []
    for p in range(1, max_lag + 1):
        Y = y[p:]
        Xf = np.column_stack([np.ones(len(Y)), _lag_mat(y, p), _lag_mat(x, p)])
        F, pv = _hc1_wald(Y, Xf, list(range(1 + p, 1 + 2 * p)))
        rows.append((p, F, pv))
    out = pd.DataFrame(rows, columns=["lag", "F", "p_raw"])
    out["p_bonf"] = np.minimum(1.0, out["p_raw"] * max_lag)
    return out


def summarize_granger(tab: pd.DataFrame, alpha: float = 0.05) -> dict:
    i = tab["p_bonf"].idxmin()
    return {"best_lag": int(tab.loc[i, "lag"]), "F": float(tab.loc[i, "F"]),
            "p_raw": float(tab.loc[i, "p_raw"]), "p_bonf": float(tab.loc[i, "p_bonf"]),
            "pass": bool(tab.loc[i, "p_bonf"] < alpha)}


# ----------------------------------------------------------------------------- ADF / KPSS / Johansen / BDS (statsmodels)
def adf_kpss(s, alpha: float = 0.05) -> dict:
    from statsmodels.tsa.stattools import adfuller, kpss

    s = pd.Series(s).dropna().astype(float)
    if len(s) < 30 or s.std() == 0:
        return {"adf_p": np.nan, "kpss_p": np.nan, "stationary": False, "verdict": "Không đủ dữ liệu/hằng số"}
    adf_p = float(adfuller(s, autolag="AIC")[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_p = float(kpss(s, regression="c", nlags="auto")[1])
    adf_ok, kpss_ok = adf_p < alpha, kpss_p >= alpha
    if adf_ok and kpss_ok:
        verdict = "Dừng"
    elif not adf_ok and not kpss_ok:
        verdict = "Không dừng"
    else:
        verdict = "Mâu thuẫn (ADF/KPSS)"
    return {"adf_p": adf_p, "kpss_p": kpss_p, "stationary": bool(adf_ok and kpss_ok), "verdict": verdict}


def integration_order(s, max_d: int = 2, alpha: float = 0.05) -> int:
    s = pd.Series(s).dropna().astype(float)
    fallback = None
    x = s.copy()
    for d in range(max_d + 1):
        if d > 0:
            x = x.diff().dropna()
        r = adf_kpss(x, alpha)
        if r["stationary"]:
            return d
        if fallback is None and r["adf_p"] == r["adf_p"] and r["adf_p"] < alpha:
            fallback = d
    return fallback if fallback is not None else max_d


def johansen_rank(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 2) -> dict:
    """Johansen (Trace & Max-eig). Hạng r = số giả thuyết trace bị bác bỏ liên tiếp ở mức 95%."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    res = coint_johansen(np.asarray(data, float), det_order, k_ar_diff)
    r = 0
    for i in range(len(res.lr1)):
        if res.lr1[i] > res.cvt[i, 1]:
            r += 1
        else:
            break
    rm = 0
    for i in range(len(res.lr2)):
        if res.lr2[i] > res.cvm[i, 1]:
            rm += 1
        else:
            break
    return {"rank_trace": r, "rank_maxeig": rm, "trace": res.lr1.tolist(), "trace_crit95": res.cvt[:, 1].tolist(),
            "maxeig": res.lr2.tolist(), "maxeig_crit95": res.cvm[:, 1].tolist()}


def bds_residuals(resid: np.ndarray, max_dim: int = 5) -> pd.DataFrame:
    from statsmodels.tsa.stattools import bds

    stat, pv = bds(np.asarray(resid, float), max_dim=max_dim)
    return pd.DataFrame({"Chiều nhúng m": list(range(2, max_dim + 1)), "Thống kê BDS": np.atleast_1d(stat),
                         "p-value": np.atleast_1d(pv)})


# ----------------------------------------------------------------------------- Ljung-Box & Variance Ratio
def ljung_box(x: np.ndarray, lags: list[int]) -> pd.DataFrame:
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = float(np.sum(x ** 2))
    rho = np.array([np.sum(x[k:] * x[:-k]) / denom for k in range(1, max(lags) + 1)])
    rows = []
    for h in lags:
        q = n * (n + 2) * np.sum(rho[:h] ** 2 / (n - np.arange(1, h + 1)))
        rows.append((h, float(q), float(stats.chi2.sf(q, h))))
    return pd.DataFrame(rows, columns=["Lag", "Q (Ljung-Box)", "p-value"])


def variance_ratio(returns: np.ndarray, qs: list[int]) -> pd.DataFrame:
    """Lo-MacKinlay (1988): VR(q) và thống kê z* bền với phương sai thay đổi. H0: bước ngẫu nhiên."""
    r = np.asarray(returns, float)
    N = len(r)
    mu = r.mean()
    a = (r - mu) ** 2
    var1 = a.sum() / (N - 1)
    rows = []
    for q in qs:
        if q < 2 or q >= N // 2:
            continue
        rq = np.convolve(r, np.ones(q), mode="valid")
        m = q * (N - q + 1) * (1 - q / N)
        varq = np.sum((rq - q * mu) ** 2) / m
        vr = varq / var1
        theta = 0.0
        for j in range(1, q):
            delta = N * np.sum(a[j:] * a[:-j]) / (a.sum() ** 2)
            theta += (2 * (q - j) / q) ** 2 * delta
        z = np.sqrt(N) * (vr - 1) / np.sqrt(theta)
        rows.append((q, float(vr), float(z), float(2 * stats.norm.sf(abs(z)))))
    return pd.DataFrame(rows, columns=["q", "VR(q)", "z*", "p-value"])


def bds_with_exog(y: np.ndarray, x: np.ndarray, p: int = 3, max_dim: int = 4) -> float:
    """p-value BDS (nhỏ nhất qua các chiều nhúng, hiệu chỉnh Bonferroni) trên phần dư của mô hình TUYẾN TÍNH
    y_t ~ const + y_{t-1..p} + x_{t-1..p}. Bác bỏ i.i.d. ⇒ còn phụ thuộc phi tuyến mà quan hệ tuyến tính (kể cả X) chưa giải thích."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    Y = y[p:]
    Z = np.column_stack([np.ones(len(Y)), _lag_mat(y, p), _lag_mat(x, p)])
    beta, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    res = Y - Z @ beta
    t = bds_residuals(res, max_dim)
    return float(min(1.0, t["p-value"].min() * len(t)))


def johansen_beta(levels: np.ndarray, r: int, k_ar_diff: int = 2, det_order: int = 0) -> np.ndarray:
    """Vector đồng liên kết β (chuẩn hoá hệ số của biến đầu tiên = 1), shape (n_vars, r)."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    res = coint_johansen(np.asarray(levels, float), det_order, k_ar_diff)
    b = res.evec[:, :r]
    return b / b[0:1, :]


def dm_from_loss_diff(d: np.ndarray, h: int = 1) -> dict:
    """DM (hiệu chỉnh Harvey-Leybourne-Newbold, phân phối t, hai phía) trên chuỗi chênh lệch tổn thất
    d_t = L(B) - L(A). mean_diff > 0 ⇒ A tốt hơn B."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    T = len(d)
    dbar = float(d.mean()) if T else np.nan
    dc = d - dbar
    lrv = float(np.sum(dc ** 2) / max(T, 1))
    for k in range(1, h):
        lrv += 2 * float(np.sum(dc[k:] * dc[:-k]) / T)
    if lrv <= 0 or T < 10:
        return {"stat": np.nan, "p": np.nan, "mean_diff": dbar, "n": T}
    dm = dbar / np.sqrt(lrv / T)
    stat = float(dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    return {"stat": stat, "p": float(2 * stats.t.sf(abs(stat), T - 1)), "mean_diff": dbar, "n": T}


def diebold_mariano(e_a: np.ndarray, e_b: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """DM giữa hai chuỗi sai số dự báo (A so với B)."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    return dm_from_loss_diff(np.abs(e_b) ** power - np.abs(e_a) ** power, h)

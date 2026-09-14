"""
Các hàm tiền xử lý dữ liệu: chuẩn hóa, xử lý missing value, outlier,
và chia train/val/test theo nguyên tắc fit-on-train-only (chống rò rỉ dữ liệu).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# ---------------------------------------------------------------------------
# 1. Các phương pháp chuẩn hóa / biến đổi
# ---------------------------------------------------------------------------

SCALER_OPTIONS = {
    "Min-Max (0-1)": "minmax",
    "Z-score (Standardization)": "zscore",
    "Log-transform": "log",
    "Log-transform + Sai phân bậc 1": "log_diff",
    "Robust Scaler (chịu outlier)": "robust",
    "Không chuẩn hóa": "none",
}


def fit_transform_column(train_series: pd.Series, method: str):
    """
    Fit scaler CHỈ trên dữ liệu train, trả về (series đã transform, scaler object).
    scaler object dùng lại để transform val/test -> chống data leakage.
    """
    values = train_series.values.reshape(-1, 1)

    if method == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(values)
        return pd.Series(scaler.transform(values).ravel(), index=train_series.index), scaler

    if method == "zscore":
        scaler = StandardScaler()
        scaler.fit(values)
        return pd.Series(scaler.transform(values).ravel(), index=train_series.index), scaler

    if method == "robust":
        scaler = RobustScaler()
        scaler.fit(values)
        return pd.Series(scaler.transform(values).ravel(), index=train_series.index), scaler

    if method == "log":
        shift = 0.0
        if (train_series <= 0).any():
            shift = abs(train_series.min()) + 1.0
        transformed = np.log(train_series + shift)
        return transformed, {"type": "log", "shift": shift}

    if method == "log_diff":
        shift = 0.0
        if (train_series <= 0).any():
            shift = abs(train_series.min()) + 1.0
        logged = np.log(train_series + shift)
        diffed = logged.diff()
        return diffed, {"type": "log_diff", "shift": shift, "last_log_value": logged.iloc[0]}

    # "none"
    return train_series.copy(), {"type": "none"}


def transform_with_fitted(series: pd.Series, method: str, scaler):
    """Áp dụng scaler đã fit từ tập train cho val/test (không fit lại)."""
    values = series.values.reshape(-1, 1)

    if method in ("minmax", "zscore", "robust"):
        return pd.Series(scaler.transform(values).ravel(), index=series.index)

    if method == "log":
        shift = scaler["shift"]
        return np.log(series + shift)

    if method == "log_diff":
        shift = scaler["shift"]
        logged = np.log(series + shift)
        return logged.diff()

    return series.copy()


def inverse_transform(values: np.ndarray, method: str, scaler):
    """Biến đổi ngược để đưa dự báo về thang đo gốc (phục vụ trang Kết quả)."""
    values = np.asarray(values).reshape(-1, 1)

    if method in ("minmax", "zscore", "robust"):
        return scaler.inverse_transform(values).ravel()

    if method == "log":
        shift = scaler["shift"]
        return np.exp(values.ravel()) - shift

    if method == "log_diff":
        # cần cộng dồn (cumsum) lại từ last_log_value rồi exp - xử lý ở tầng gọi
        return values.ravel()

    return values.ravel()


# ---------------------------------------------------------------------------
# 2. Xử lý missing value
# ---------------------------------------------------------------------------

MISSING_OPTIONS = {
    "Nội suy tuyến tính (interpolate)": "interpolate",
    "Forward-fill": "ffill",
    "Backward-fill": "bfill",
    "Điền bằng trung vị nhóm (theo tỉnh)": "median_group",
    "Xóa dòng có missing": "drop",
}


def handle_missing(df: pd.DataFrame, cols: list, method: str, group_col: str = None) -> pd.DataFrame:
    df = df.copy()
    if method == "interpolate":
        df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
    elif method == "ffill":
        df[cols] = df[cols].ffill().bfill()
    elif method == "bfill":
        df[cols] = df[cols].bfill().ffill()
    elif method == "median_group" and group_col is not None:
        df[cols] = df.groupby(group_col)[cols].transform(lambda s: s.fillna(s.median()))
    elif method == "drop":
        df = df.dropna(subset=cols)
    return df


# ---------------------------------------------------------------------------
# 3. Xử lý outlier
# ---------------------------------------------------------------------------

def winsorize_iqr(df: pd.DataFrame, cols: list, k: float = 1.5) -> pd.DataFrame:
    """Kẹp (clip) giá trị ngoài khoảng [Q1 - k*IQR, Q3 + k*IQR]."""
    df = df.copy()
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower, upper)
    return df


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


# ---------------------------------------------------------------------------
# 4. Chia train / validation / test theo thời gian (không shuffle -> chống leakage)
# ---------------------------------------------------------------------------

def time_based_split(df: pd.DataFrame, date_col: str, train_ratio: float, val_ratio: float):
    """
    Chia theo mốc thời gian (không random shuffle vì là time series).
    Trả về 3 DataFrame train/val/test và các mốc ngày cắt.
    """
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df_sorted.iloc[:n_train]
    val = df_sorted.iloc[n_train:n_train + n_val]
    test = df_sorted.iloc[n_train + n_val:]

    cutoffs = {
        "train_end": train[date_col].max() if len(train) else None,
        "val_end": val[date_col].max() if len(val) else None,
    }
    return train, val, test, cutoffs

"""
Các kiểm định thống kê dùng trong trang Phân tích khám phá (EDA):
ADF (tính dừng), VIF (đa cộng tuyến), tương quan chéo giữa các tỉnh.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_adf_test(series: pd.Series) -> dict:
    """Kiểm định Augmented Dickey-Fuller cho tính dừng của một chuỗi."""
    s = series.dropna()
    if len(s) < 10:
        return {"error": "Chuỗi quá ngắn để kiểm định (cần >= 10 quan sát)."}
    result = adfuller(s, autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags_used": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Tính Variance Inflation Factor cho từng biến trong danh sách cols."""
    X = df[cols].dropna()
    X = X.assign(_const=1.0)
    vif_data = []
    for i, col in enumerate(cols):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({"Biến": col, "VIF": vif})
    result = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    return result


def cross_group_correlation(df: pd.DataFrame, date_col: str, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Tính ma trận tương quan giữa các nhóm (tỉnh) trên cùng một biến giá trị,
    dùng để phát hiện dư thừa panel (near-perfect correlation) như đã nêu trong luận văn.
    """
    pivot = df.pivot_table(index=date_col, columns=group_col, values=value_col)
    return pivot.corr()


def adf_summary_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Chạy ADF cho nhiều cột cùng lúc, trả về bảng tóm tắt để hiển thị."""
    rows = []
    for c in cols:
        res = run_adf_test(df[c])
        if "error" in res:
            rows.append({"Biến": c, "ADF stat": None, "p-value": None, "Kết luận": res["error"]})
        else:
            rows.append({
                "Biến": c,
                "ADF stat": round(res["adf_statistic"], 4),
                "p-value": round(res["p_value"], 4),
                "Kết luận": "Dừng I(0)" if res["is_stationary"] else "Không dừng (cần sai phân) I(1)+",
            })
    return pd.DataFrame(rows)

"""Chọn biến: VIF, lọc đa cộng tuyến, Mutual Information, tầm quan trọng dạng cây (XGBoost/RandomForest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_utils import difference


# ----------------------------------------------------------------------------- VIF & đa cộng tuyến
def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j) khi hồi quy biến j lên các biến còn lại (có hằng số)."""
    cols = list(df.columns)
    X = df.to_numpy(float)
    out = []
    for j, c in enumerate(cols):
        if len(cols) == 1:
            out.append((c, 1.0))
            continue
        y = X[:, j]
        Z = np.column_stack([np.ones(len(y)), np.delete(X, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        ss_res = float(np.sum((y - Z @ beta) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append((c, np.inf if r2 > 1 - 1e-12 else 1 / (1 - r2)))
    t = pd.DataFrame(out, columns=["Biến", "VIF"])
    t["Mức"] = np.where(t["VIF"] >= 10, "Mạnh (≥10)", np.where(t["VIF"] >= 5, "Cảnh báo (≥5)", "OK"))
    return t.sort_values("VIF", ascending=False).reset_index(drop=True)


def high_corr_pairs(df: pd.DataFrame, thr: float = 0.8, method: str = "pearson") -> pd.DataFrame:
    corr = df.corr(method=method)
    rows = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= thr:
                rows.append((cols[i], cols[j], round(float(r), 4)))
    return pd.DataFrame(rows, columns=["Biến A", "Biến B", "ρ"])


def suggest_multicollinearity_drops(dtrain: pd.DataFrame, target: str, xcols: list[str], rho_thr: float = 0.8,
                                    vif_thr: float = 10.0, method: str = "pearson") -> pd.DataFrame:
    """Giữ 1 đại diện cho mỗi cụm |ρ|≥ngưỡng (ưu tiên biến có |corr với Y| lớn hơn), rồi loại dần theo VIF≥ngưỡng.
    Toàn bộ tính trên sai phân TRAIN."""
    drops: list[tuple[str, str, str]] = []
    corr = dtrain[[target] + xcols].corr(method=method)
    order = corr.loc[xcols, target].abs().sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    for v in order:
        partner = next((k for k in kept if abs(corr.loc[v, k]) >= rho_thr), None)
        if partner is None:
            kept.append(v)
        else:
            drops.append((v, f"|ρ|={abs(corr.loc[v, partner]):.2f} ≥ {rho_thr} với {partner}", partner))
    while len(kept) > 1:
        vt = vif_table(dtrain[kept])
        worst = vt.iloc[0]
        if worst["VIF"] < vif_thr:
            break
        kept.remove(worst["Biến"])
        drops.append((worst["Biến"], f"VIF={worst['VIF']:.1f} ≥ {vif_thr}", ""))
    return pd.DataFrame(drops, columns=["Biến", "Lý do", "Đại diện giữ lại"])


# ----------------------------------------------------------------------------- đặc trưng trễ
def build_lagged(df: pd.DataFrame, target: str, xcols: list[str], n_lags: int, include_y_lags: bool = True):
    """Ma trận đặc trưng: y_t ~ [X_{t-1..t-L}, (Y_{t-1..t-L})]. Trả về (X, y, tên cột, chỉ số dòng)."""
    feats, names = {}, []
    for c in xcols:
        for k in range(1, n_lags + 1):
            feats[f"{c}__L{k}"] = df[c].shift(k)
    if include_y_lags:
        for k in range(1, n_lags + 1):
            feats[f"{target}__L{k}"] = df[target].shift(k)
    F = pd.DataFrame(feats)
    full = pd.concat([df[target].rename("__y__"), F], axis=1).dropna()
    return full.drop(columns="__y__"), full["__y__"], list(F.columns), full.index


# ----------------------------------------------------------------------------- XGBoost (cố định)
def xgb_importance(frames: list[pd.DataFrame], target: str, xcols: list[str], n_lags: int = 5,
                   n_estimators: int = 300, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Tầm quan trọng biến X bằng XGBoost (gain). `frames`: danh sách DataFrame đã sai phân (mỗi tỉnh một khung, chỉ TRAIN).
    y_t ~ [X_{t-1..t-L}, Y_{t-1..t-L}] ; 80% đầu mỗi khung để học, 20% cuối để đo R² kiểm chứng (đúng thứ tự thời gian)."""
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor  # bắt buộc cài xgboost

    Xtr, ytr, Xva, yva = [], [], [], []
    for f in frames:
        X, y, _, _ = build_lagged(f, target, xcols, n_lags)
        k = int(len(X) * 0.8)
        Xtr.append(X.iloc[:k]); ytr.append(y.iloc[:k]); Xva.append(X.iloc[k:]); yva.append(y.iloc[k:])
    Xtr, ytr, Xva, yva = pd.concat(Xtr), pd.concat(ytr), pd.concat(Xva), pd.concat(yva)
    mdl = XGBRegressor(n_estimators=n_estimators, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, importance_type="gain", tree_method="hist", random_state=seed, n_jobs=-1)
    mdl.fit(Xtr, ytr)
    r2 = float(r2_score(yva, mdl.predict(Xva)))
    imp = pd.Series(mdl.feature_importances_, index=Xtr.columns)
    rows = []
    for c in xcols:
        cols_c = [f"{c}__L{k}" for k in range(1, n_lags + 1)]
        rows.append({"Biến": c, "Importance (gain)": float(imp[cols_c].sum()),
                     "Lag mạnh nhất": int(imp[cols_c].idxmax().split("__L")[1])})
    out = pd.DataFrame(rows).sort_values("Importance (gain)", ascending=False).reset_index(drop=True)
    out["Hạng"] = np.arange(1, len(out) + 1)
    return out, {"r2_val": r2, "n_train": len(Xtr), "y_lag_share": float(imp[[c for c in imp.index if c.startswith(target)]].sum())}

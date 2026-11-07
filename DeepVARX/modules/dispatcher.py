"""Bộ định tuyến dữ liệu (Data Dispatcher): cắt df_clean_panel thành 3 gói vật lý A, B, C."""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_utils import DATA_PROC

PKG_FILES = {"A": "train_test_A_base.csv", "B": "train_test_B_expanded_diff.csv", "C": "train_test_C_expanded_scaled.csv"}


def _diff(df: pd.DataFrame, ent: pd.Series, cols: list[str], d: int) -> pd.DataFrame:
    x = df[cols]
    for _ in range(d):
        x = x.groupby(ent, observed=True).diff()
    return x


def build_packages(panel: pd.DataFrame, date_col: str, entity_col: str, target: str, base: list[str],
                   ext: list[str], d: int, train_end, base_model: str = "VAR", write: bool = True,
                   endog_cols: list[str] | None = None, **kwargs) -> tuple[dict, dict]:
    """Gói A: Tập Cơ Sở (sai phân bậc d; hoặc mức gốc nếu base_model='VECM'). Dùng cho VAR/VECM, VARX.
    Gói B: Tập Mở Rộng sai phân bậc d. Dùng cho VARX-L.
    Gói C: Tập Mở Rộng mức gốc (d=0), Standard Scaler THEO TỪNG TỈNH (khớp trên TRAIN) + province_id. Dùng cho mô hình Global.
    Mọi gói có cột `split` (train/test theo mốc ngày chung) và `y_level` (giá gốc VNĐ/kg để khôi phục đơn vị).
    Hỗ trợ đầy đủ bộ biến nội sinh đa biến endog_cols (chuỗi giá cà phê) và ngoại sinh exog_cols."""
    if endog_cols is None and "endog_cols" in kwargs:
        endog_cols = kwargs["endog_cols"]
    all_endog = [c for c in endog_cols if c in panel.columns] if endog_cols else [target]
    if not all_endog:
        all_endog = [target]
    primary_target = all_endog[0]

    m = panel.sort_values([entity_col, date_col]).reset_index(drop=True)
    m["split"] = np.where(m[date_col] <= pd.Timestamp(train_end), "train", "test")
    m["y_level"] = m[primary_target]
    cats = list(m[entity_col].cat.categories)
    keep = [date_col, entity_col, "split", "y_level"]
    ent = m[entity_col]

    def diff_pkg(cols):
        all_c = all_endog + [c for c in cols if c not in all_endog]
        x = _diff(m, ent, all_c, d)
        return pd.concat([m[keep], x], axis=1).dropna(subset=all_c).reset_index(drop=True)

    def level_pkg(cols):
        all_c = all_endog + [c for c in cols if c not in all_endog]
        return pd.concat([m[keep], m[all_c]], axis=1).reset_index(drop=True)

    cols_all = all_endog + [c for c in ext if c not in all_endog]
    tr = m["split"] == "train"
    gb = m[cols_all].where(np.broadcast_to(tr.to_numpy()[:, None], (len(m), len(cols_all)))).groupby(ent, observed=True)
    mu, sd = gb.transform("mean"), gb.transform("std").replace(0, 1.0).fillna(1.0)
    C = pd.concat([m[keep], (m[cols_all] - mu) / sd], axis=1)
    C["province_id"] = pd.Categorical(ent, categories=cats).codes
    scalers = {str(e): (float(mu.loc[ent == e, primary_target].iloc[0]), float(sd.loc[ent == e, primary_target].iloc[0])) for e in cats}

    pk = {"A": level_pkg(base) if base_model == "VECM" else diff_pkg(base), "B": diff_pkg(ext), "C": C.reset_index(drop=True)}
    meta = {"scalers": scalers, "entities": [str(c) for c in cats], "d": int(d), "base_model": base_model,
            "A_mode": "levels" if base_model == "VECM" else "diff", "train_end": pd.Timestamp(train_end),
            "target": primary_target, "endog_cols": list(all_endog), "exog_cols": list(ext),
            "base": list(base), "ext": list(ext), "date_col": date_col, "entity_col": entity_col,
            "paths": {}}
    if write:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        for k, f in PKG_FILES.items():
            pk[k].to_csv(DATA_PROC / f, index=False)
            meta["paths"][k] = str(DATA_PROC / f)
    return pk, meta

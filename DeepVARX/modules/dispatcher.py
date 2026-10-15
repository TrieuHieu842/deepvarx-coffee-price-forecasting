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
                   ext: list[str], d: int, train_end, base_model: str = "VAR", write: bool = True) -> tuple[dict, dict]:
    """Gói A: Tập Cơ Sở (sai phân bậc d; hoặc mức gốc nếu base_model='VECM'). Dùng cho VAR/VECM, VARX.
    Gói B: Tập Mở Rộng sai phân bậc d. Dùng cho VARX-L.
    Gói C: Tập Mở Rộng mức gốc (d=0), Standard Scaler THEO TỪNG TỈNH (khớp trên TRAIN) + province_id. Dùng cho mô hình Global.
    Mọi gói có cột `split` (train/test theo mốc ngày chung) và `y_level` (giá gốc VNĐ/kg để khôi phục đơn vị)."""
    m = panel.sort_values([entity_col, date_col]).reset_index(drop=True)
    m["split"] = np.where(m[date_col] <= pd.Timestamp(train_end), "train", "test")
    m["y_level"] = m[target]
    cats = list(m[entity_col].cat.categories)
    keep = [date_col, entity_col, "split", "y_level"]
    ent = m[entity_col]

    def diff_pkg(cols):
        x = _diff(m, ent, [target] + cols, d)
        return pd.concat([m[keep], x], axis=1).dropna(subset=[target] + cols).reset_index(drop=True)

    def level_pkg(cols):
        return pd.concat([m[keep], m[[target] + cols]], axis=1).reset_index(drop=True)

    cols_x = [target] + ext
    tr = m["split"] == "train"
    gb = m[cols_x].where(np.broadcast_to(tr.to_numpy()[:, None], (len(m), len(cols_x)))).groupby(ent, observed=True)
    mu, sd = gb.transform("mean"), gb.transform("std").replace(0, 1.0).fillna(1.0)
    C = pd.concat([m[keep], (m[cols_x] - mu) / sd], axis=1)
    C["province_id"] = pd.Categorical(ent, categories=cats).codes
    scalers = {str(e): (float(mu.loc[ent == e, target].iloc[0]), float(sd.loc[ent == e, target].iloc[0])) for e in cats}

    pk = {"A": level_pkg(base) if base_model == "VECM" else diff_pkg(base), "B": diff_pkg(ext), "C": C.reset_index(drop=True)}
    meta = {"scalers": scalers, "entities": [str(c) for c in cats], "d": int(d), "base_model": base_model,
            "A_mode": "levels" if base_model == "VECM" else "diff", "train_end": pd.Timestamp(train_end),
            "base": list(base), "ext": list(ext), "date_col": date_col, "entity_col": entity_col, "target": target,
            "paths": {}}
    if write:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        for k, f in PKG_FILES.items():
            pk[k].to_csv(DATA_PROC / f, index=False)
            meta["paths"][k] = str(DATA_PROC / f)
    return pk, meta

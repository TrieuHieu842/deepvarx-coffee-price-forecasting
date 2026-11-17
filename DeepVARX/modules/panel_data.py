"""Hệ sinh thái dữ liệu bảng (Panel Data) và dữ liệu rộng đa tỉnh (Wide Multivariate):
đọc → nhận diện cấu trúc → kiểm tra → chuẩn hoá → passport → làm sạch / trích chuỗi.

Nguyên tắc:
- Nếu file là Panel (có cột thực thể như `province`): giữ nguyên từng dòng, ép kiểu ngày và sắp xếp [thực thể → ngày].
- Nếu file là dạng Bảng rộng (không có cột thực thể, như `data_daily_clean_v2.csv`): tự động nhận diện
  các cột giá từng tỉnh ở Việt Nam làm biến nội sinh (Y) và giữ nguyên các cột ngoại sinh (X).
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata

import numpy as np
import pandas as pd

from modules.data_utils import (DATA_PROC, DATA_RAW, ROOT, gap_report, interpolate_train_safe,
                                make_demo_data, numeric_columns, zero_diff_ratio)

ENTITY_HINTS = ["province", "provinces", "tinh", "tỉnh", "region", "vung", "vùng", "dia_phuong", "địa phương", "city",
                "market", "thị trường"]
DATE_HINTS = ["date", "ngay", "ngày", "datetime", "timestamp", "time", "month", "thang", "tháng"]
DEFAULT_NAMES = ["data_daily_panel_clean_v2.csv", "data_daily_clean_v2.csv", "coffee_price_cleaned.csv", "coffee-data.xlsx"]

VN_PROVINCES: dict[str, str] = {
    "dak_lak": "Đắk Lắk", "daklak": "Đắk Lắk",
    "lam_dong": "Lâm Đồng", "lamdong": "Lâm Đồng",
    "gia_lai": "Gia Lai", "gialai": "Gia Lai",
    "dak_nong": "Đắk Nông", "daknong": "Đắk Nông",
    "kon_tum": "Kon Tum", "kontum": "Kon Tum",
    "binh_phuoc": "Bình Phước", "binhphuoc": "Bình Phước",
    "dong_nai": "Đồng Nai", "dongnai": "Đồng Nai",
    "ba_ria_vung_tau": "Bà Rịa - Vũng Tàu", "vung_tau": "Bà Rịa - Vũng Tàu",
    "son_la": "Sơn La", "sonla": "Sơn La",
    "dien_bien": "Điện Biên", "dienbien": "Điện Biên",
    "quang_tri": "Quảng Trị", "quangtri": "Quảng Trị",
    "nghe_an": "Nghệ An", "nghean": "Nghệ An",
    "tay_nguyen": "Tây Nguyên", "taynguyen": "Tây Nguyên",
    "an_giang": "An Giang", "bac_giang": "Bắc Giang", "bac_kan": "Bắc Kạn",
    "bac_lieu": "Bạc Liêu", "bac_ninh": "Bắc Ninh", "ben_tre": "Bến Tre",
    "binh_dinh": "Bình Định", "binh_duong": "Bình Dương", "binh_thuan": "Bình Thuận",
    "ca_mau": "Cà Mau", "can_tho": "Cần Thơ", "cao_bang": "Cao Bằng",
    "da_nang": "Đà Nẵng", "dong_thap": "Đồng Tháp", "ha_giang": "Hà Giang",
    "ha_nam": "Hà Nam", "ha_noi": "Hà Nội", "ha_tinh": "Hà Tĩnh",
    "hai_duong": "Hải Dương", "hai_phong": "Hải Phòng", "hau_giang": "Hậu Giang",
    "hoa_binh": "Hòa Bình", "hung_yen": "Hưng Yên", "khanh_hoa": "Khánh Hòa",
    "kien_giang": "Kiên Giang", "lai_chau": "Lai Châu", "lang_son": "Lạng Sơn",
    "lao_cai": "Lào Cai", "long_an": "Long An", "nam_dinh": "Nam Định",
    "ninh_binh": "Ninh Bình", "ninh_thuan": "Ninh Thuận", "phu_tho": "Phú Thọ",
    "phu_yen": "Phú Yên", "quang_binh": "Quảng Bình", "quang_nam": "Quảng Nam",
    "quang_ngai": "Quảng Ngãi", "quang_ninh": "Quảng Ninh", "soc_trang": "Sóc Trăng",
    "tay_ninh": "Tây Ninh", "thai_binh": "Thái Bình", "thai_nguyen": "Thái Nguyên",
    "thanh_hoa": "Thanh Hóa", "thua_thien_hue": "Thừa Thiên Huế", "tien_giang": "Tiền Giang",
    "tp_hcm": "TP. Hồ Chí Minh", "ho_chi_minh": "TP. Hồ Chí Minh", "tra_vinh": "Trà Vinh",
    "tuyen_quang": "Tuyên Quang", "vinh_long": "Vĩnh Long", "vinh_phuc": "Vĩnh Phúc",
    "yen_bai": "Yên Bái",
}
_VN_SLUGS_SORTED = sorted(VN_PROVINCES, key=len, reverse=True)
_FOREIGN_KEYWORDS = {"london", "arabica", "brent", "dxy", "usd", "brl", "world", "brazil", "global", "diesel"}
_PRICE_TOKENS = {"price", "gia", "vnd", "coffee"}


def _slugify_vn(text: str) -> str:
    s = str(text).strip().replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")


def _match_vn_province(slug: str) -> str | None:
    padded = f"_{slug}_"
    for p_slug in _VN_SLUGS_SORTED:
        if f"_{p_slug}_" in padded:
            return p_slug
    return None


def is_vn_province_price_col(col: str) -> tuple[bool, str | None]:
    """Nhận diện cột giá thu mua cà phê của từng tỉnh ở Việt Nam (không nhầm giá sàn/ngoại sinh)."""
    s = _slugify_vn(col)
    if not s:
        return False, None
    if s.startswith(("ret_", "d_", "diff_", "pct_", "ecm_", "log_")) or "return" in s or re.search(r"_lag\d+$", s):
        return False, None
    tokens = set(s.split("_"))
    if tokens & _FOREIGN_KEYWORDS:
        return False, None
    p_slug = _match_vn_province(s)
    if p_slug is None:
        return False, None
    has_price_hint = bool(tokens & _PRICE_TOKENS) or "ca_phe" in s or s == p_slug
    if not has_price_hint:
        return False, None
    return True, VN_PROVINCES[p_slug]


def is_endog_derived_col(col: str, endog_cols: list[str]) -> bool:
    """Nhận diện cột phái sinh từ giá nội sinh các tỉnh (vd: ret_dak_lak, ret_lam_dồng, price_dak_lak_lag1)."""
    s = _slugify_vn(col)
    if not s:
        return False
    endog_prov_slugs = {_match_vn_province(_slugify_vn(c)) for c in endog_cols}
    endog_prov_slugs.discard(None)
    p_slug = _match_vn_province(s)
    if p_slug and p_slug in endog_prov_slugs:
        if s.startswith(("ret_", "d_", "diff_", "pct_", "log_")) or "return" in s or re.search(r"_lag\d+$", s):
            return True
    endog_slugs = {_slugify_vn(c) for c in endog_cols}
    return any(s.startswith(f"{es}_lag") for es in endog_slugs)


def province_display_labels(endog_cols: list[str]) -> dict[str, str]:
    """Tạo nhãn hiển thị duy nhất cho mỗi cột nội sinh (ưu tiên tên tỉnh tiếng Việt)."""
    raw_labels = {}
    for c in endog_cols:
        ok, name = is_vn_province_price_col(c)
        raw_labels[c] = name if ok and name else str(c)
    counts = pd.Series(list(raw_labels.values())).value_counts() if raw_labels else pd.Series(dtype=int)
    return {c: (f"{lbl} ({c})" if counts.get(lbl, 0) > 1 else lbl) for c, lbl in raw_labels.items()}


# ----------------------------------------------------------------------------- đọc file
def fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def find_default_file():
    for folder in (ROOT / "data", DATA_PROC, DATA_RAW):
        for n in DEFAULT_NAMES:
            if (folder / n).exists():
                return folder / n
    return None


def read_table(raw: bytes, filename: str) -> tuple[pd.DataFrame, dict]:
    """Đọc CSV/Excel; tự nhận encoding và dấu phân cách. Trả về (df, meta)."""
    meta = {"filename": filename, "bytes": len(raw), "sha256": fingerprint(raw)}
    if filename.lower().endswith((".xlsx", ".xls")):
        meta.update(encoding="Excel", delimiter="Excel")
        return pd.read_excel(io.BytesIO(raw)), meta
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            meta["encoding"] = enc
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    if df.shape[1] == 1:  # Sniffer đoán sai → thử các dấu còn lại, lấy cái nhiều cột nhất
        best = (1, sep, df)
        for s in (",", ";", "\t", "|"):
            try:
                t = pd.read_csv(io.StringIO(text), sep=s)
            except Exception:  # noqa: BLE001
                continue
            if t.shape[1] > best[0]:
                best = (t.shape[1], s, t)
        sep, df = best[1], best[2]
    meta["delimiter"] = {"\t": "TAB"}.get(sep, sep)
    return df, meta


# ----------------------------------------------------------------------------- nhận diện cột
def detect_columns(df: pd.DataFrame) -> dict:
    low = {str(c).strip().lower(): c for c in df.columns}
    date = next((low[h] for h in DATE_HINTS if h in low), None)
    if date is None:
        for c in df.columns:
            if df[c].dtype == object:
                if pd.to_datetime(df[c].head(50), errors="coerce").notna().mean() > 0.8:
                    date = c
                    break
    entity = [low[h] for h in ENTITY_HINTS if h in low][:1]
    nums = numeric_columns(df, exclude=[date] + entity if date else entity)

    if entity:
        target = None
        for c in nums:
            if "coffee_price" in str(c).lower():
                target = c
                break
        if target is None:
            target = next((c for c in nums if any(k in str(c).lower() for k in ("price", "gia", "giá"))), None)
        endog = [target] if target else []
        return {
            "mode": "panel", "date": date, "entity": entity, "endog": endog, "target": target,
            "derived_endog": [], "confident": bool(date and entity and target),
        }

    # Không có cột thực thể → kiểm tra dạng bảng rộng (các cột giá tỉnh Việt Nam là nội sinh)
    vn_prices = [c for c in nums if is_vn_province_price_col(c)[0]]
    derived = [c for c in nums if c not in vn_prices and is_endog_derived_col(c, vn_prices)] if vn_prices else []
    target = None
    if vn_prices:
        target = next((c for c in vn_prices if "dak_lak" in _slugify_vn(c)), vn_prices[0])
    else:
        target = next((c for c in nums if any(k in str(c).lower() for k in ("price", "gia", "giá"))), None)
    return {
        "mode": "wide", "date": date, "entity": [], "endog": vn_prices, "target": target,
        "derived_endog": derived, "confident": bool(date and len(vn_prices) >= 1),
    }


def validate_panel(df: pd.DataFrame, date_col, entity_cols: list[str], target) -> list[tuple[str, str]]:
    """Trả về danh sách (mức, thông điệp). Mức 'error' chặn khởi tạo."""
    out = []
    if date_col is None or date_col not in df.columns:
        out.append(("error", "Thiếu cột thời gian (date/datetime…)."))
    if not entity_cols:
        out.append(("error", "Thiếu cột không gian (province/region). Không thể chạy mô hình đa không gian."))
    if target is None or target not in df.columns:
        out.append(("error", "Chưa chọn cột giá mục tiêu Y."))
    elif not pd.api.types.is_numeric_dtype(df[target]):
        out.append(("error", f"Cột mục tiêu `{target}` không phải kiểu số."))
    else:
        excl = [c for c in [date_col] + list(entity_cols) + [target] if c in df.columns]
        if not numeric_columns(df, exclude=excl):
            out.append(("warning", "Không có cột số nào làm biến ngoại sinh X."))
    return out


def validate_wide(df: pd.DataFrame, date_col, endog_cols: list[str], target,
                  derived_endog: list[str] | None = None) -> list[tuple[str, str]]:
    """Kiểm tra dữ liệu bảng rộng đa biến (các tỉnh là các cột nội sinh)."""
    out = []
    if date_col is None or date_col not in df.columns:
        out.append(("error", "Thiếu cột thời gian (date/datetime…)."))
    if not endog_cols:
        out.append(("error", "Chưa chọn cột giá tỉnh nào làm biến nội sinh Y."))
    else:
        non_num = [c for c in endog_cols if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            out.append(("error", f"Các cột nội sinh không hợp lệ hoặc không phải số: {', '.join(non_num)}."))
        elif target is None or target not in endog_cols:
            out.append(("error", "Biến mục tiêu trọng tâm Y₁ phải nằm trong danh sách biến nội sinh."))
    if not out:
        excl = [date_col] + list(endog_cols) + list(derived_endog or [])
        if not numeric_columns(df, exclude=excl):
            out.append(("warning", "Không có cột số nào làm biến ngoại sinh X."))
    return out


# ----------------------------------------------------------------------------- chuẩn hoá (không gộp)
def normalize_panel(df: pd.DataFrame, date_col: str, entity_cols: list[str], dayfirst: bool = False,
                    dup_policy: str = "mean") -> tuple[pd.DataFrame, dict, str]:
    """Ép ngày, chuẩn hoá nhãn thực thể, xử lý khoá trùng (nếu có), sắp xếp [thực thể → ngày].
    dup_policy: 'mean' | 'first' | 'keep'. TUYỆT ĐỐI không gộp theo ngày."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=dayfirst)
    bad_dates = int(d[date_col].isna().sum())
    d = d.dropna(subset=[date_col])
    for c in entity_cols:
        d[c] = d[c].astype(str).str.strip()
    if len(entity_cols) == 1:
        entity_col = entity_cols[0]
    else:
        entity_col = "entity_id"
        d[entity_col] = d[entity_cols].agg(" | ".join, axis=1)
    key = [entity_col, date_col]
    n_dup = int(d.duplicated(key).sum())
    if n_dup and dup_policy == "first":
        d = d.drop_duplicates(key, keep="first")
    elif n_dup and dup_policy == "mean":
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(d[c]) else "first") for c in d.columns if c not in key}
        d = d.groupby(key, as_index=False, sort=False).agg(agg)
    d = d.sort_values(key, kind="mergesort").reset_index(drop=True)
    d[entity_col] = d[entity_col].astype("category")
    info = {"bad_dates": bad_dates, "duplicate_keys": n_dup, "dup_policy": dup_policy if n_dup else "không có trùng"}
    return d, info, entity_col


def normalize_wide(df: pd.DataFrame, date_col: str, dayfirst: bool = False,
                   dup_policy: str = "mean") -> tuple[pd.DataFrame, dict]:
    """Chuẩn hoá dữ liệu bảng rộng đa biến: ép kiểu ngày, xử lý ngày trùng (nếu có), sắp xếp tăng dần theo ngày."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=dayfirst)
    bad_dates = int(d[date_col].isna().sum())
    d = d.dropna(subset=[date_col])
    n_dup = int(d.duplicated([date_col]).sum())
    if n_dup and dup_policy == "first":
        d = d.drop_duplicates([date_col], keep="first")
    elif n_dup and dup_policy == "mean":
        agg = {c: ("mean" if pd.api.types.is_numeric_dtype(d[c]) else "first") for c in d.columns if c != date_col}
        d = d.groupby(date_col, as_index=False, sort=False).agg(agg)
    d = d.sort_values(date_col, kind="mergesort").reset_index(drop=True)
    info = {"bad_dates": bad_dates, "duplicate_keys": n_dup, "dup_policy": dup_policy if n_dup else "không có trùng"}
    return d, info


# ----------------------------------------------------------------------------- báo cáo chất lượng
def infer_freq(dates: pd.Series) -> str:
    u = pd.DatetimeIndex(np.sort(dates.dropna().unique()))
    if len(u) < 3:
        return "D"
    step = int(np.median(np.diff(u.values).astype("timedelta64[D]").astype(int)))
    if step <= 1:
        return "B" if (u.dayofweek >= 5).mean() < 0.02 else "D"
    return f"{step}D"


def entity_report(df: pd.DataFrame, date_col: str, entity_col: str) -> pd.DataFrame:
    freq = infer_freq(df[date_col])
    rows = []
    for e, g in df.groupby(entity_col, observed=True):
        lo, hi = g[date_col].min(), g[date_col].max()
        exp = len(pd.date_range(lo, hi, freq=freq))
        rows.append({"Thực thể": e, "Số dòng": len(g), "Từ ngày": lo.date(), "Đến ngày": hi.date(),
                     "Số dòng kỳ vọng": exp, "Thiếu (dòng)": max(exp - len(g), 0),
                     "% thiếu": round(100 * max(exp - len(g), 0) / max(exp, 1), 2)})
    return pd.DataFrame(rows)


def variable_scope(df: pd.DataFrame, date_col: str, num_cols: list[str]) -> pd.Series:
    """Phân loại biến: 'Toàn quốc' nếu giá trị giống nhau giữa các tỉnh trong cùng ngày, ngược lại 'Theo tỉnh'.
    Biến toàn quốc chỉ có T quan sát độc lập (không phải N×T) — quan trọng khi tính kiểm định."""
    cnt = df.groupby(date_col, observed=True)[num_cols[0]].transform("size") if num_cols else None
    multi = df[cnt >= 2] if cnt is not None else df
    out = {}
    for c in num_cols:
        nun = multi.groupby(date_col, observed=True)[c].nunique(dropna=True)
        out[c] = "Toàn quốc" if len(nun) and (nun <= 1).mean() >= 0.99 else "Theo tỉnh"
    return pd.Series(out)


def build_passport(master: pd.DataFrame, meta: dict, info: dict, date_col: str, entity_cols: list[str],
                   entity_col: str, target: str) -> dict:
    excl = [date_col, entity_col] + list(entity_cols)
    num_cols = numeric_columns(master, exclude=excl)
    rep = entity_report(master, date_col, entity_col)
    scope = variable_scope(master, date_col, num_cols)
    cols = pd.DataFrame({"Kiểu": master[num_cols].dtypes.astype(str),
                         "% khuyết": (master[num_cols].isna().mean() * 100).round(2),
                         "Phạm vi": scope}).reset_index().rename(columns={"index": "Biến"})
    rows_per = rep.set_index("Thực thể")["Số dòng"].to_dict()
    return {
        **meta, **info, "mode": "panel", "date_col": date_col, "entity_cols": list(entity_cols), "entity_col": entity_col,
        "target": target, "num_cols": num_cols, "rows": len(master), "cols": master.shape[1],
        "n_entities": len(rows_per), "min_date": master[date_col].min(), "max_date": master[date_col].max(),
        "freq": infer_freq(master[date_col]), "rows_per_entity": rows_per,
        "rows_per_entity_mean": float(np.mean(list(rows_per.values()))),
        "balanced": len(set(rows_per.values())) == 1, "entity_table": rep, "column_table": cols,
    }


def wide_endog_report(master: pd.DataFrame, date_col: str, endog_cols: list[str],
                      endog_labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for c in endog_cols:
        s = master[c]
        valid = s.notna()
        rows.append({
            "Tỉnh / Vùng": endog_labels.get(c, c),
            "Cột dữ liệu": c,
            "Số dòng hợp lệ": int(valid.sum()),
            "Từ ngày": master.loc[valid, date_col].min().date() if valid.any() else None,
            "Đến ngày": master.loc[valid, date_col].max().date() if valid.any() else None,
            "% khuyết": round(float(s.isna().mean() * 100), 2),
            "Giá TB": round(float(s.mean()), 1) if valid.any() else np.nan,
            "Min": round(float(s.min()), 1) if valid.any() else np.nan,
            "Max": round(float(s.max()), 1) if valid.any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_wide_passport(master: pd.DataFrame, meta: dict, info: dict, date_col: str, endog_cols: list[str],
                        target: str, derived_endog: list[str], endog_labels: dict[str, str]) -> dict:
    num_cols = numeric_columns(master, exclude=[date_col])
    exog_cols = [c for c in num_cols if c not in endog_cols and c not in derived_endog]
    endog_set = set(endog_cols)
    derived_set = set(derived_endog)
    roles = []
    for c in num_cols:
        if c == target:
            roles.append("Nội sinh (Y₁ trọng tâm)")
        elif c in endog_set:
            roles.append("Nội sinh (Y tỉnh VN)")
        elif c in derived_set:
            roles.append("Lợi suất nội sinh (tách khỏi X)")
        else:
            roles.append("Ngoại sinh (X)")
    cols_tbl = pd.DataFrame({
        "Biến": num_cols,
        "Kiểu": master[num_cols].dtypes.astype(str).to_numpy(),
        "% khuyết": (master[num_cols].isna().mean() * 100).round(2).to_numpy(),
        "Vai trò": roles,
    })
    rep = wide_endog_report(master, date_col, endog_cols, endog_labels)
    return {
        **meta, **info, "mode": "wide", "date_col": date_col, "entity_cols": [], "entity_col": None,
        "target": target, "endog_cols": list(endog_cols), "endog_labels": dict(endog_labels),
        "exog_cols": exog_cols, "derived_endog": list(derived_endog), "num_cols": num_cols,
        "rows": len(master), "cols": master.shape[1], "n_entities": len(endog_cols),
        "min_date": master[date_col].min(), "max_date": master[date_col].max(),
        "freq": infer_freq(master[date_col]), "rows_per_entity": {endog_labels.get(c, c): len(master) for c in endog_cols},
        "rows_per_entity_mean": float(len(master)), "balanced": True,
        "entity_table": rep, "column_table": cols_tbl,
    }


# ----------------------------------------------------------------------------- trích chuỗi cho pipeline
def extract_series(master: pd.DataFrame, date_col: str, entity_col: str, num_cols: list[str], mode: str,
                   entity: str | None = None, min_coverage: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """mode='province': chuỗi của một thực thể (Local). mode='mean': trung bình các thực thể theo ngày
    (chỉ giữ ngày có ≥ min_coverage thực thể để tránh nhảy cấu phần)."""
    if mode == "province":
        out = master.loc[master[entity_col] == entity, [date_col] + num_cols].set_index(date_col)
        out = out.groupby(level=0).mean().sort_index()
        note = {"mode": "province", "entity": entity, "rows": len(out)}
    else:
        n_ent = master[entity_col].nunique()
        cnt = master.groupby(date_col, observed=True)[entity_col].nunique()
        keep = cnt[cnt >= min_coverage * n_ent].index
        sub = master[master[date_col].isin(keep)]
        out = sub.groupby(date_col, observed=True)[num_cols].mean().sort_index()
        note = {"mode": "mean", "entity": f"trung bình {n_ent} thực thể", "rows": len(out),
                "dropped_dates": int(len(cnt) - len(keep))}
    out.index.name = "date"
    return out, note


# ----------------------------------------------------------------------------- dữ liệu demo dạng panel
def make_demo_panel(n_prov: int = 9, n: int = 900, seed: int = 3) -> pd.DataFrame:
    base = make_demo_data(n=n, seed=seed).drop(columns=["gappy_index"])
    rng = np.random.default_rng(seed)
    names = ["Đắk Lắk", "Lâm Đồng", "Gia Lai", "Đắk Nông", "Kon Tum", "Bình Phước", "Đồng Nai", "Sơn La", "Quảng Trị"]
    frames = []
    for i in range(n_prov):
        f = base.copy()
        f["province"] = names[i % len(names)]
        f["coffee_price_vnd_kg"] = f["coffee_price_vnd_kg"] * (1 + 0.02 * i) + np.cumsum(rng.normal(0, 40, n))
        f["rainfall_mm"] = np.clip(f["rainfall_mm"] + rng.normal(0, 15, n), 0, None)
        f["temperature_c"] = f["temperature_c"] + rng.normal(0, 0.3, n)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ----------------------------------------------------------------------------- làm sạch TOÀN panel & dạng rộng
def train_end_from_ratio(master: pd.DataFrame, date_col: str, ratio: float) -> pd.Timestamp:
    """Mốc ngày cuối của TRAIN, CHUNG cho mọi tỉnh (tránh rò rỉ chéo tỉnh)."""
    u = np.sort(master[date_col].unique())
    return pd.Timestamp(u[max(int(len(u) * ratio) - 1, 0)])


def _max_gap(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def panel_gap_report(master: pd.DataFrame, entity_col: str, cols: list[str], date_col: str,
                     train_end: pd.Timestamp) -> pd.DataFrame:
    """Khoảng trống đo TRONG từng tỉnh (không bao giờ qua ranh giới giữa hai tỉnh)."""
    groups = [g for _, g in master.groupby(entity_col, observed=True, sort=False)]
    tr = master[date_col] <= train_end
    rows = []
    for c in cols:
        gaps = [_max_gap(g[c].isna().to_numpy()) for g in groups]
        rows.append({"Biến": c, "% khuyết (toàn bộ)": round(100 * master[c].isna().mean(), 2),
                     "% khuyết (TRAIN)": round(100 * master.loc[tr, c].isna().mean(), 2),
                     "Số tỉnh có khuyết": int(sum(g > 0 for g in gaps)), "Max gap (dòng, trong 1 tỉnh)": int(max(gaps))})
    return pd.DataFrame(rows)


def wide_gap_report(master: pd.DataFrame, date_col: str, cols: list[str], train_end: pd.Timestamp) -> pd.DataFrame:
    """Báo cáo khoảng trống trên dữ liệu bảng rộng (theo từng cột, sắp xếp theo ngày)."""
    g = master.sort_values(date_col)
    gi = g.set_index(date_col)[cols]
    n_tr = int((gi.index <= train_end).sum())
    rep = gap_report(gi, n_tr)
    return rep.rename(columns={"Max gap (số dòng)": "Max gap (dòng)"})


def panel_interpolate(master: pd.DataFrame, date_col: str, entity_col: str, cols: list[str],
                      train_end: pd.Timestamp, method: str) -> pd.DataFrame:
    """Nội suy chống look-ahead cho TỪNG tỉnh với cùng một mốc train_end."""
    parts = []
    for e, g in master.groupby(entity_col, observed=True, sort=False):
        gi = g.set_index(date_col)[cols]
        n_tr = int((gi.index <= train_end).sum())
        if n_tr == 0:
            continue
        out, _ = interpolate_train_safe(gi, n_tr, method)
        if len(out) == 0:
            continue
        out = out.reset_index()
        out.insert(1, entity_col, e)
        parts.append(out)
    res = pd.concat(parts, ignore_index=True).dropna(subset=cols)
    res[entity_col] = pd.Categorical(res[entity_col], categories=master[entity_col].cat.categories)
    return res.rename(columns={res.columns[0]: date_col})


def wide_interpolate(master: pd.DataFrame, date_col: str, cols: list[str],
                     train_end: pd.Timestamp, method: str) -> pd.DataFrame:
    """Nội suy chống look-ahead cho dữ liệu bảng rộng."""
    gi = master.sort_values(date_col).set_index(date_col)[cols]
    n_tr = int((gi.index <= train_end).sum())
    if n_tr == 0:
        return master.iloc[0:0][[date_col] + cols].copy()
    out, _ = interpolate_train_safe(gi, n_tr, method)
    if len(out) == 0:
        return master.iloc[0:0][[date_col] + cols].copy()
    res = out.reset_index().dropna(subset=cols)
    return res.rename(columns={res.columns[0]: date_col})


def panel_zero_diff(master: pd.DataFrame, date_col: str, entity_col: str, cols: list[str],
                    train_end: pd.Timestamp) -> pd.Series:
    """Tỷ lệ sai phân = 0 tính trong từng tỉnh (TRAIN), gộp lại."""
    d = master.groupby(entity_col, observed=True)[cols].diff()
    tr = (master[date_col] <= train_end).to_numpy()
    valid = d.notna().to_numpy() & tr[:, None]
    zero = (d.to_numpy() == 0) & valid
    return pd.Series(zero.sum(0) / np.maximum(valid.sum(0), 1), index=cols)


def wide_zero_diff(master: pd.DataFrame, date_col: str, cols: list[str],
                   train_end: pd.Timestamp) -> pd.Series:
    """Tỷ lệ sai phân = 0 trên đoạn TRAIN cho dữ liệu bảng rộng."""
    tr = master[master[date_col] <= train_end].sort_values(date_col)
    return pd.Series({c: zero_diff_ratio(tr[c]) for c in cols}, index=cols)


def panel_diff_train_z(df, date_col, entity_col, cols, train_end):
    """Δ bậc 1 trong từng tỉnh (đoạn TRAIN), chuẩn hoá z-score trong tỉnh rồi xếp chồng."""
    parts = []
    for _, g in df.groupby(entity_col, sort=False):
        g = g[g[date_col] <= train_end].sort_values(date_col)
        d = g[cols].diff().dropna()
        if len(d) > 1:
            parts.append(((d - d.mean()) / d.std(ddof=0).replace(0, np.nan)).fillna(0))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def wide_diff_train_z(df: pd.DataFrame, date_col: str, cols: list[str], train_end) -> pd.DataFrame:
    """Δ bậc 1 đoạn TRAIN chuẩn hoá z-score cho dữ liệu bảng rộng."""
    g = df[df[date_col] <= train_end].sort_values(date_col)
    d = g[cols].diff().dropna()
    if len(d) <= 1:
        return pd.DataFrame(columns=cols)
    return ((d - d.mean()) / d.std(ddof=0).replace(0, np.nan)).fillna(0)


def panel_diff_train(df: pd.DataFrame, date_col: str, entity_col: str, cols: list[str], train_end, d: int = 1) -> pd.DataFrame:
    """Sai phân bậc d trong từng tỉnh (đoạn TRAIN) rồi xếp chồng để tính VIF."""
    parts = []
    for _, g in df.groupby(entity_col, observed=True, sort=False):
        g = g[g[date_col] <= train_end].sort_values(date_col)
        sub = g[cols].astype(float)
        for _ in range(max(1, int(d))):
            sub = sub.diff()
        sub = sub.dropna()
        if len(sub) > 0:
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


balanced_panel_diff_train = panel_diff_train


def entity_frame(master: pd.DataFrame, date_col: str, entity_col: str, entity: str,
                 cols: list[str] | None = None) -> pd.DataFrame:
    g = master[master[entity_col] == entity].set_index(date_col).sort_index()
    return g[cols] if cols else g


def proxy_table(df, date_col, entity_col, target, train_end, min_cov=0.8):
    tr = df[df[date_col] <= train_end]
    wd = tr.pivot_table(index=date_col, columns=entity_col, values=target, aggfunc="mean").sort_index().diff()
    wd.columns = wd.columns.astype(str)
    n = tr[entity_col].astype(str).value_counts()
    t = pd.DataFrame({"Số dòng TRAIN": n})
    t["Độ phủ"] = t["Số dòng TRAIN"] / n.max()
    t["ρ (Δ) với TB các tỉnh còn lại"] = [wd[e].corr(wd.drop(columns=e).mean(axis=1)) for e in t.index]
    t["Đủ điều kiện"] = t["Độ phủ"] >= min_cov
    return t.sort_values(["Đủ điều kiện", "ρ (Δ) với TB các tỉnh còn lại"], ascending=False)

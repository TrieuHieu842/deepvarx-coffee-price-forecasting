import hashlib
import json
import os
import re
import warnings
from datetime import datetime
from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá thu mua cà phê Việt Nam"
PASSPORT_VERSION = "2.2"

PROVINCE_COL = "province"
COFFEE_TYPE_COL = "coffee_type"
DATE_CANDIDATES = {"date", "datetime", "timestamp"}

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "coffee_price_cleaned.csv"
)

# Tham số ứng dụng (không phải chuẩn học thuật)
REPEAT_FLAG_SHARE = 50.0                      # % giá trị lặp liên tiếp để cảnh báo biến "đứng yên"
MIN_GAPS_FOR_RANKING = 30                     # số đoạn ẩn tối thiểu để tin xếp hạng ở B.6

st.set_page_config(page_title="DeepVARX - Data Passport", page_icon="☕", layout="wide")


def _streamlit_version():
    out = []
    for p in st.__version__.split(".")[:2]:
        out.append(int(re.sub(r"\D", "", p) or 0))
    return tuple(out)


# use_container_width đã bị deprecate; dùng width="stretch" khi Streamlit đủ mới
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "raw_data": None,
    "date_col": "date",
    "data_file_name": None,
    "data_file_hash": None,
    "data_loaded_at": None,
    "data_frequency": None,
    "data_date_min": None,
    "data_date_max": None,
    "missing_masks": None,
    "imputation_result": None,
    "data_passport": {},
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ============================================================
# HELPERS — ĐỌC FILE
# ============================================================
def calculate_sha256(file_bytes):
    """SHA-256 của bytes file đầu vào."""
    return hashlib.sha256(file_bytes).hexdigest()


def detect_separator(raw_bytes):
    """Tự phát hiện CSV phân tách bằng , ; hoặc tab."""
    try:
        first_line = raw_bytes.decode("utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return ","
    counts = {sep: first_line.count(sep) for sep in [",", "\t", ";"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _read_csv_bytes(raw_bytes):
    return pd.read_csv(BytesIO(raw_bytes), sep=detect_separator(raw_bytes))


def read_uploaded_file(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        df = _read_csv_bytes(raw_bytes)
    else:
        df = pd.read_excel(BytesIO(raw_bytes))
    return df, raw_bytes


@st.cache_data(show_spinner=False)
def load_default_file(path, mtime):
    """Cache theo (path, mtime) để không đọc lại file mỗi lần rerun."""
    with open(path, "rb") as f:
        raw_bytes = f.read()
    return _read_csv_bytes(raw_bytes), raw_bytes


_DMY = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")


def detect_date_order(series):
    """
    Phát hiện thứ tự ngày/tháng của chuỗi dạng d/m/y hoặc m/d/y.
    Trả về: None (không phải dạng d/m/y, ví dụ ISO) | 'dmy' | 'mdy' | 'mixed' | 'ambiguous'.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return None
    m = s.str.extract(_DMY)
    if m.isna().all().all():
        return None
    a = pd.to_numeric(m[0], errors="coerce")
    b = pd.to_numeric(m[1], errors="coerce")
    a_big, b_big = bool((a > 12).any()), bool((b > 12).any())
    if a_big and b_big:
        return "mixed"
    if a_big:
        return "dmy"
    if b_big:
        return "mdy"
    return "ambiguous"


def parse_date_column(df):
    """
    Tự phát hiện cột ngày, KHÔNG để pandas đoán mù thứ tự ngày/tháng.
    Trả về (df, date_col, số giá trị không parse được, ghi chú về thứ tự ngày/tháng).
    """
    df = df.copy()
    date_col = next((c for c in df.columns if str(c).lower() in DATE_CANDIDATES), None)
    if date_col is None:
        return df, None, 0, None
    n_bad, note = 0, None
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        before = int(df[date_col].notna().sum())
        order = detect_date_order(df[date_col])
        dayfirst = order in ("dmy", "ambiguous", "mixed")
        if order == "ambiguous":
            note = ("Chuỗi ngày dạng dd/mm/yyyy hoặc mm/dd/yyyy nhưng mọi giá trị đều ≤ 12 nên không phân biệt được; "
                    "đã mặc định dd/mm/yyyy (chuẩn Việt Nam). Hãy kiểm tra Date min/max bên dưới.")
        elif order == "mixed":
            note = ("Phát hiện cả giá trị ngày > 12 ở vị trí thứ nhất lẫn thứ hai: định dạng ngày không nhất quán. "
                    "Đã thử dd/mm/yyyy; các giá trị lỗi sẽ thành NaT.")
        elif order == "dmy":
            note = "Phát hiện định dạng dd/mm/yyyy; đã parse với dayfirst=True."
        elif order == "mdy":
            note = "Phát hiện định dạng mm/dd/yyyy; đã parse với dayfirst=False."
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
        n_bad = before - int(df[date_col].notna().sum())
    return df, date_col, n_bad, note


# ============================================================
# HELPERS — TÊN BIẾN / PANEL
# ============================================================
def name_matches(name, kw):
    """So khớp theo TOKEN (tránh 'rain' khớp 'train', 'gia' khớp 'original')."""
    norm = re.sub(r"[^0-9a-zà-ỹ]+", " ", str(name).lower()).strip()
    tokens = set(norm.split())
    if tokens & set(kw.get("exact", ())):
        return True
    if any(t.startswith(p) for t in tokens for p in kw.get("prefix", ())):
        return True
    return any(ph in norm for ph in kw.get("phrases", ()))


KW_PRICE = {"exact": {"price", "gia", "giá", "target"}, "prefix": ("price",)}
KW_HUMID = {"exact": {"rh"}, "prefix": ("humid",), "phrases": ("do am", "độ ẩm")}
KW_RAIN = {"exact": {"mua", "mưa"}, "prefix": ("rain", "precip")}
KW_TEMP = {"exact": {"nhiet", "nhiệt", "t2m", "tmax", "tmin"}, "prefix": ("temp",),
           "phrases": ("nhiet do", "nhiệt độ")}



def get_group_cols(df):
    return [c for c in (PROVINCE_COL, COFFEE_TYPE_COL) if c in df.columns]


def sort_panel(df, date_col):
    """Sắp xếp theo (tỉnh, loại, ngày) — bắt buộc trước khi tính gap / sai phân."""
    cols = get_group_cols(df) + ([date_col] if date_col else [])
    return df.sort_values(cols, kind="mergesort") if cols else df


def iter_series(df, col, date_col):
    """Duyệt từng chuỗi (tỉnh × loại) đã sắp theo ngày."""
    d = sort_panel(df, date_col)
    gcols = get_group_cols(d)
    if gcols:
        for _, g in d.groupby(gcols, sort=False, dropna=False):
            yield g[col].reset_index(drop=True)
    else:
        yield d[col].reset_index(drop=True)


def choose_price_columns(df):
    return [c for c in df.columns if name_matches(c, KW_PRICE) and pd.api.types.is_numeric_dtype(df[c])]


# ============================================================
# HELPERS — TẦN SUẤT / NGÀY
# ============================================================
CAL_AUTO = "Tự suy từ dữ liệu"
CAL_ALL = "Tất cả các ngày"
CAL_BUSINESS = "Ngày làm việc (T2–T6)"


def get_unique_dates(df, date_col):
    if date_col is None or date_col not in df.columns:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(df[date_col].dropna().drop_duplicates().sort_values())


def infer_frequency(dates):
    """Tần suất = mode của chênh lệch ngày liên tiếp."""
    if len(dates) < 2:
        return None, None
    deltas = pd.Series(dates).diff().dropna()
    mode_delta = deltas.mode().iloc[0]
    days = mode_delta.total_seconds() / 86400
    if days == 1:
        freq = "Daily"
    elif 6 <= days <= 8:
        freq = "Weekly"
    elif 27 <= days <= 32:
        freq = "Monthly"
    elif 85 <= days <= 95:
        freq = "Quarterly"
    elif 360 <= days <= 370:
        freq = "Yearly"
    else:
        freq = f"Không chuẩn hóa ({days:.2f} ngày)"
    return freq, mode_delta


def build_expected_index(dates, frequency, mode_delta, calendar=CAL_ALL):
    """
    Lưới ngày kỳ vọng.
    - Tháng: lịch MS/ME (Δt không cố định).
    - Ngày: tùy chọn lịch (mọi ngày / T2–T6 / tự suy các thứ có trong dữ liệu) để không coi
      cuối tuần, ngày không công bố giá là "ngày thiếu".
    - Còn lại: Δt = mode.
    """
    if len(dates) < 2 or mode_delta is None:
        return None
    start, end = dates.min(), dates.max()
    if frequency == "Monthly":
        alias = "MS" if (dates.day == 1).all() else ("ME" if dates.is_month_end.all() else None)
        if alias:
            try:
                return pd.date_range(start, end, freq=alias)
            except ValueError:
                return pd.date_range(start, end, freq="M")
    if frequency == "Daily":
        full = pd.date_range(start, end, freq="D")
        if calendar == CAL_BUSINESS:
            return full[full.dayofweek < 5]
        if calendar == CAL_AUTO:
            share = pd.Series(dates.dayofweek).value_counts(normalize=True)
            keep = set(share[share >= 0.05].index)
            return full[full.dayofweek.isin(keep)]
        return full
    return pd.date_range(start, end, freq=mode_delta)


def missing_segments(expected, missing):
    if len(missing) == 0:
        return pd.DataFrame(columns=["Từ ngày", "Đến ngày", "Số kỳ thiếu"])
    pos = expected.get_indexer(missing)
    chunks = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
    return pd.DataFrame([{
        "Từ ngày": expected[c[0]].date(),
        "Đến ngày": expected[c[-1]].date(),
        "Số kỳ thiếu": len(c),
    } for c in chunks])


def abnormal_delta_pairs(dates, frequency, mode_delta, expected=None):
    """
    Cặp ngày liên tiếp bất thường.
    Tháng: ngoài 28–31 ngày. Ngày: nhảy bước trên lưới lịch kỳ vọng (T6→T2 là bình thường nếu lịch bỏ cuối tuần).
    """
    if len(dates) < 2 or mode_delta is None:
        return pd.DataFrame(columns=["Ngày trước", "Ngày sau", "Δt (ngày)"])
    s = pd.Series(dates)
    delta = s.diff()
    days = delta.dt.total_seconds() / 86400
    if frequency == "Monthly":
        bad = ~days.between(28, 31)
    elif frequency == "Daily" and expected is not None:
        pos = pd.Series(expected.get_indexer(dates))
        bad = (pos.diff() != 1) & (pos >= 0) & (pos.shift() >= 0)
    else:
        bad = delta != mode_delta
    bad = bad & delta.notna()
    return pd.DataFrame({
        "Ngày trước": s.shift(1)[bad].dt.date.values,
        "Ngày sau": s[bad].dt.date.values,
        "Δt (ngày)": days[bad].values,
    })


# ============================================================
# HELPERS — THỐNG KÊ MÔ TẢ
# ============================================================
def get_dtype_summary(df):
    result = {"Float": 0, "Integer": 0, "Datetime": 0, "Categorical": 0}
    for dtype in df.dtypes:
        if pd.api.types.is_float_dtype(dtype):
            result["Float"] += 1
        elif pd.api.types.is_integer_dtype(dtype):
            result["Integer"] += 1
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            result["Datetime"] += 1
        else:
            result["Categorical"] += 1
    return result


def moment_stats(s):
    """g1 = m3/m2^(3/2);  g2 = m4/m2² − 3 (moment trung tâm, đúng đặc tả)."""
    x = s.to_numpy(dtype=float)
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    if m2 == 0:
        return np.nan, np.nan
    return m3 / m2 ** 1.5, m4 / m2 ** 2 - 3


def calculate_cv(s):
    mean = s.mean()
    return np.nan if pd.isna(mean) or mean == 0 else s.std() / mean


def calculate_descriptive_statistics(df, numeric_cols):
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        g1, g2 = moment_stats(s)
        rows.append({
            "Variable": col, "Mean": s.mean(), "Median": s.median(), "Std": s.std(),
            "Min": s.min(), "Max": s.max(), "Skewness": g1, "Kurtosis (excess)": g2,
            "CV": calculate_cv(s),
        })
    return pd.DataFrame(rows)


def hist_kde_chart(s, title, height=220):
    s = s.dropna()
    n_bins = int(min(60, max(10, np.sqrt(len(s)))))
    counts, edges = np.histogram(s, bins=n_bins, density=True)
    hist_df = pd.DataFrame({"x0": edges[:-1], "x1": edges[1:], "Density": counts})
    layers = [alt.Chart(hist_df).mark_bar(opacity=0.55).encode(
        x=alt.X("x0:Q", title=title), x2="x1:Q", y=alt.Y("Density:Q"))]
    if SCIPY_AVAILABLE and s.nunique() > 1:
        try:
            sample = s.sample(min(len(s), 5000), random_state=0).to_numpy()
            kde = stats.gaussian_kde(sample)
            xs = np.linspace(s.min(), s.max(), 200)
            layers.append(alt.Chart(pd.DataFrame({"x": xs, "Density": kde(xs)}))
                          .mark_line(color="#d62728").encode(x="x:Q", y="Density:Q"))
        except Exception:
            pass
    return alt.layer(*layers).properties(height=height)


# ============================================================
# HELPERS — MISSING
# ============================================================
def calculate_gap_lengths(series):
    """Run-length của các đoạn khuyết liên tiếp."""
    is_missing = series.isna()
    if not is_missing.any():
        return []
    groups = is_missing.ne(is_missing.shift()).cumsum()
    return is_missing[is_missing].groupby(groups[is_missing]).size().tolist()


@st.cache_data(show_spinner=False)
def compute_missing_report(df, date_col):
    """Gap tính TRONG từng chuỗi (tỉnh × loại) đã sắp theo ngày, không vắt qua ranh giới tỉnh."""
    n = len(df)
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        lengths = []
        for s in iter_series(df, col, date_col):
            lengths.extend(calculate_gap_lengths(s))
        rows.append({
            "Variable": col,
            "Missing": n_miss,
            "Missing %": 100 * n_miss / n if n else 0.0,
            "Consecutive gaps": len(lengths),
            "Max gap": int(max(lengths)) if lengths else 0,
            "Median gap": float(np.median(lengths)) if lengths else 0.0,
        })
    return pd.DataFrame(rows).sort_values("Missing", ascending=False).reset_index(drop=True)


MISSING_BY_OPTIONS = ("Tháng", "Năm", "Ngày trong tuần")


def chi_square_missing(df, col, date_col, by):
    """
    χ² độc lập giữa chỉ báo khuyết và nhóm thời gian.
    Để giảm phụ thuộc chéo trong panel (cùng một ngày lặp qua nhiều tỉnh/loại), chỉ báo được gộp
    về MỨC NGÀY: ngày đó có ≥ 1 chuỗi bị khuyết hay không. Kèm Cramér's V để đo độ lớn hiệu ứng.
    """
    if not SCIPY_AVAILABLE or date_col is None:
        return None
    d = pd.DataFrame({"d": df[date_col], "m": df[col].isna()}).dropna(subset=["d"])
    daily = d.groupby("d")["m"].max().astype(int)
    idx = daily.index
    period = {"Tháng": idx.month, "Năm": idx.year, "Ngày trong tuần": idx.dayofweek}[by]
    ct = pd.crosstab(pd.Series(period, index=idx), daily)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else np.nan
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": v,
            "pct_low_expected": float((exp < 5).mean() * 100), "n_days": int(n)}


@st.cache_data(show_spinner=False)
def compute_missing_patterns(df, date_col):
    """Kết quả χ² (thăm dò) cho từng biến -> đưa vào Data Passport. Hiệu chỉnh Bonferroni theo số phép thử."""
    out = {}
    for col in df.columns:
        if col == date_col:
            continue
        if not df[col].isna().any():
            out[col] = "Không khuyết"
            continue
        res = [(by, chi_square_missing(df, col, date_col, by)) for by in MISSING_BY_OPTIONS]
        res = [(by, r) for by, r in res if r is not None]
        if not res:
            out[col] = "Không kiểm định được"
            continue
        alpha = 0.05 / len(res)
        sig = [(by, r) for by, r in res if r["p"] < alpha]
        if sig:
            by, r = min(sig, key=lambda x: x[1]["p"])
            out[col] = f"Có dấu hiệu theo {by.lower()} (p={r['p']:.3g}, V={r['cramers_v']:.2f}; thăm dò)"
        else:
            out[col] = "Chưa có bằng chứng về quy luật"
    return out


# ---- Kiểm chứng phương pháp điền (B.6) ----
IMPUTE_METHODS = {
    "Forward-fill": "Nhân quả",
    "Rolling mean quá khứ": "Nhân quả",
    "Linear interpolation": "Không nhân quả",
    "Backward-fill": "Không nhân quả",
    "Centered window": "Không nhân quả",
    "Spline": "Không nhân quả",
}


def impute_series(s, method, window):
    nan_series = pd.Series(np.nan, index=s.index)
    if method == "Forward-fill":
        return s.ffill()
    if method == "Rolling mean quá khứ":
        return s.fillna(s.shift(1).rolling(window, min_periods=1).mean())
    if method == "Linear interpolation":
        return s.interpolate(method="linear", limit_area="inside")
    if method == "Backward-fill":
        return s.bfill()
    if method == "Centered window":
        return s.fillna(s.rolling(window, center=True, min_periods=1).mean())
    if method == "Spline":
        if not SCIPY_AVAILABLE:
            return nan_series
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return s.interpolate(method="spline", order=3, limit_area="inside")
        except Exception:
            return nan_series
    return nan_series


def gap_bucket(length):
    if length == 1:
        return "1"
    if length <= 3:
        return "2–3"
    if length <= 7:
        return "4–7"
    return "> 7"


GAP_ORDER = {"1": 0, "2–3": 1, "4–7": 2, "> 7": 3}


def run_imputation_experiment(df, value_col, date_col, train_frac, n_trials, window, seed):
    """
    Ẩn ngẫu nhiên các đoạn giá trị THẬT, độ dài lấy từ phân phối gap quan sát được
    (trong đoạn TRAIN), điền lại bằng từng phương pháp rồi so với giá trị thật.
    Báo cáo MAE, RMSE, MAPE, sai số chuẩn của MAE (giữa các đoạn ẩn) và độ phủ.
    Giả định: vị trí khuyết ngẫu nhiên hoàn toàn (MCAR).
    """
    rng = np.random.default_rng(seed)
    series_list = []
    for s in iter_series(df, value_col, date_col):
        s = pd.to_numeric(s, errors="coerce")
        series_list.append(s.iloc[: int(len(s) * train_frac)].reset_index(drop=True))

    observed_gaps = [L for s in series_list for L in calculate_gap_lengths(s)]
    used_fallback = len(observed_gaps) == 0
    gap_pool = [1, 2, 3, 5, 8] if used_fallback else observed_gaps

    recs, done, attempts = [], 0, 0
    while done < n_trials and attempts < n_trials * 40:
        attempts += 1
        s = series_list[int(rng.integers(len(series_list)))]
        L = int(gap_pool[int(rng.integers(len(gap_pool)))])
        n = len(s)
        if n < L + 4:
            continue
        start = int(rng.integers(1, n - L))
        if not (s.iloc[start - 1:start + L + 1].notna().all()):
            continue
        ctx = max(window * 2, L * 3, 20)
        lo, hi = max(0, start - ctx), min(n, start + L + ctx)
        local = s.iloc[lo:hi].reset_index(drop=True)
        a = start - lo
        truth = local.iloc[a:a + L].to_numpy()
        masked = local.copy()
        masked.iloc[a:a + L] = np.nan
        for m in IMPUTE_METHODS:
            pred = impute_series(masked, m, window).iloc[a:a + L].to_numpy()
            ok = ~np.isnan(pred)
            err = pred[ok] - truth[ok]
            denom = np.abs(truth[ok])
            pos = denom > 0
            n_pts = int(ok.sum())
            recs.append({
                "Gap": gap_bucket(L), "Method": m, "n_total": L, "n_pts": n_pts,
                "abs": float(np.abs(err).sum()), "sq": float((err ** 2).sum()),
                "ape": float((np.abs(err)[pos] / denom[pos]).sum()), "n_ape": int(pos.sum()),
                "gap_mae": float(np.abs(err).sum() / n_pts) if n_pts > 0 else np.nan,
            })
        done += 1

    info = {"done": done, "fallback": used_fallback, "observed_gaps": len(observed_gaps)}
    if not recs:
        return pd.DataFrame(), info
    r = pd.DataFrame(recs)
    agg = r.groupby(["Gap", "Method"]).agg(
        n_pts=("n_pts", "sum"), n_total=("n_total", "sum"),
        abs_sum=("abs", "sum"), sq_sum=("sq", "sum"),
        ape_sum=("ape", "sum"), n_ape=("n_ape", "sum"),
        n_gaps=("n_total", "size")).reset_index()
    se = (r.groupby(["Gap", "Method"])["gap_mae"]
          .agg(lambda x: x.std(ddof=1) / np.sqrt(x.notna().sum()) if x.notna().sum() > 1 else np.nan)
          .rename("SE_MAE").reset_index())
    agg = agg.merge(se, on=["Gap", "Method"], how="left")
    pts = agg.n_pts.replace(0, np.nan)
    agg["MAE_imp"] = agg.abs_sum / pts
    agg["RMSE_imp"] = np.sqrt(agg.sq_sum / pts)
    agg["MAPE %"] = 100 * agg.ape_sum / agg.n_ape.replace(0, np.nan)
    agg["Coverage %"] = 100 * agg.n_pts / agg.n_total
    agg["Loại"] = agg["Method"].map(IMPUTE_METHODS)
    agg["Ghi chú"] = np.where(agg.n_gaps < MIN_GAPS_FOR_RANKING, f"⚠ < {MIN_GAPS_FOR_RANKING} đoạn: xếp hạng kém tin cậy", "")
    agg = agg.sort_values(["Gap", "MAE_imp"], key=lambda c: c.map(GAP_ORDER) if c.name == "Gap" else c)
    out = agg[["Gap", "Method", "Loại", "n_gaps", "MAE_imp", "SE_MAE", "RMSE_imp",
               "MAPE %", "Coverage %", "Ghi chú"]].reset_index(drop=True)
    return out, info


# ============================================================
# HELPERS — QUALITY
# ============================================================
def calculate_invalid_domain_flags(df):
    """Gắn cờ, KHÔNG xóa dữ liệu."""
    rows, masks = [], {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        checks = []
        if name_matches(col, KW_PRICE):
            checks.append(("Giá <= 0", s <= 0))
        if name_matches(col, KW_HUMID):
            checks.append(("Độ ẩm ngoài [0,100]", (s < 0) | (s > 100)))
        if name_matches(col, KW_RAIN):
            checks.append(("Lượng mưa < 0", s < 0))
        if name_matches(col, KW_TEMP):
            checks.append(("Nhiệt độ ngoài khoảng vật lý [-90,70]", (s < -90) | (s > 70)))
        for rule, mask in checks:
            if mask.any():
                rows.append({"Variable": col, "Rule": rule, "Count": int(mask.sum())})
                masks[(col, rule)] = mask
    return pd.DataFrame(rows), masks


SCALE_GLOBAL = "Toàn chuỗi"
SCALE_ROLLING = "Cục bộ (rolling)"


def robust_score(x, scale_mode=SCALE_GLOBAL, window=60):
    """
    Điểm Z hiệu chỉnh của Iglewicz & Hoaglin (1993).
      M_i = 0.6745 (x_i − median) / MAD            nếu MAD > 0
      M_i = (x_i − median) / (1.253314 · MeanAD)   nếu MAD = 0 (khuyến nghị gốc của I&H)
    Giá thu mua thường đứng yên nhiều kỳ nên sai phân = 0 chiếm đa số -> MAD = 0 rất hay xảy ra;
    nhánh dự phòng tránh trả về toàn NaN (im lặng không phát hiện được gì).
    Trả về (score, n_dùng_MAD, n_dùng_MeanAD).
    """
    x = pd.to_numeric(x, errors="coerce")
    score = pd.Series(np.nan, index=x.index)
    if x.notna().sum() < 3:
        return score, 0, 0
    if scale_mode == SCALE_ROLLING:
        mp = max(10, window // 2)
        med = x.rolling(window, min_periods=mp, center=True).median()
        dev = (x - med).abs()
        mad = dev.rolling(window, min_periods=mp, center=True).median()
        mean_ad = dev.rolling(window, min_periods=mp, center=True).mean()
    else:
        med = pd.Series(x.median(), index=x.index)
        dev = (x - med).abs()
        mad = pd.Series(float(np.median(dev.dropna())), index=x.index)
        mean_ad = pd.Series(float(dev.dropna().mean()), index=x.index)
    use_mad = (mad > 0) & x.notna()
    use_mean = (~(mad > 0)) & (mean_ad > 0) & x.notna()
    score[use_mad] = 0.6745 * (x - med)[use_mad] / mad[use_mad]
    score[use_mean] = (x - med)[use_mean] / (1.253314 * mean_ad[use_mean])
    return score, int(use_mad.sum()), int(use_mean.sum())


def outlier_scores(df, col, date_col, method, scale_mode=SCALE_GLOBAL, window=60):
    """Sai phân / log-return TRONG từng chuỗi (tỉnh × loại), rồi điểm M_i theo từng chuỗi."""
    d = sort_panel(df, date_col)
    gcols = get_group_cols(d)
    s = pd.to_numeric(d[col], errors="coerce")
    if method == "Log-return":
        s = np.log(s.where(s > 0))
    keys = [d[c] for c in gcols]
    diff = s.groupby(keys, sort=False, dropna=False).diff() if keys else s.diff()
    score = pd.Series(np.nan, index=d.index)
    n_mad = n_mean = 0
    groups = diff.groupby(keys, sort=False, dropna=False) if keys else [(None, diff)]
    for _, dg in groups:
        sc, a, b = robust_score(dg, scale_mode, window)
        score.loc[dg.index] = sc
        n_mad += a
        n_mean += b
    return d, diff, score, {"n_mad": n_mad, "n_meanad": n_mean, "n_valid": int(diff.notna().sum())}


@st.cache_data(show_spinner=False)
def compute_repeat_stats(df, date_col, delta_days):
    """
    Mức độ 'đứng yên' của từng biến số trong từng chuỗi: tỷ lệ giá trị lặp liên tiếp và chuỗi lặp dài nhất.
    Dùng để (1) phát hiện biến tần suất thấp bị lặp xuống tần suất cao (tháng -> ngày),
    (2) đo thời gian tối đa một biến không được cập nhật.
    """
    rows = []
    for col in df.columns:
        if col == date_col or col in get_group_cols(df) or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_pairs = n_same = 0
        max_run = 1
        for s in iter_series(df, col, date_col):
            s = pd.to_numeric(s, errors="coerce").dropna().reset_index(drop=True)
            if len(s) < 2:
                continue
            same = s.eq(s.shift())
            n_pairs += len(s) - 1
            n_same += int(same.iloc[1:].sum())
            max_run = max(max_run, int((~same).cumsum().value_counts().max()))
        share = 100 * n_same / n_pairs if n_pairs else np.nan
        rows.append({
            "Variable": col,
            "Tỷ lệ lặp liên tiếp (%)": share,
            "Chuỗi lặp dài nhất (kỳ)": max_run,
            "Chuỗi lặp dài nhất (ngày)": max_run * delta_days if delta_days else np.nan,
        })
    return pd.DataFrame(rows, columns=["Variable", "Tỷ lệ lặp liên tiếp (%)",
                                       "Chuỗi lặp dài nhất (kỳ)", "Chuỗi lặp dài nhất (ngày)"])


def df_to_records(df):
    return json.loads(df.to_json(orient="records", date_format="iso", force_ascii=False))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    if st.session_state.get("raw_data") is not None:
        _d = st.session_state["raw_data"]
        st.success(f"Dữ liệu: {_d.shape[0]:,} × {_d.shape[1]}")
    if st.session_state.get("data_file_hash"):
        st.caption("SHA-256: " + st.session_state["data_file_hash"][:16] + "...")
    st.divider()
    if st.button("← Về Home", key="sb_home", **STRETCH):
        st.switch_page("Home.py")


# ============================================================
# HEADER
# ============================================================
st.caption(APP_HEADER)
st.title("Trang 2 — Dữ liệu & Data Passport")
st.markdown(
    """
Trang này tính **Data Passport** của bộ dữ liệu đầu vào: mọi con số đều được tính trực tiếp từ tệp, trang không biến đổi dữ liệu và không yêu cầu khai báo thủ công.

Mục tiêu là trả lời 3 câu hỏi:

1. Dữ liệu có cấu trúc như thế nào?
2. Dữ liệu có thiếu hoặc có vấn đề gì không?
3. Dữ liệu có vấn đề về toàn vẹn (trùng lặp, giá trị không hợp lệ, ngoại lai, ngày thiếu) không?
"""
)


# ============================================================
# 0. NẠP DỮ LIỆU
# ============================================================
st.divider()
st.subheader("0. Nạp dữ liệu")

load_mode = st.radio("Nguồn dữ liệu", ["Dùng file mặc định", "Upload file khác"], horizontal=True)

df_raw, file_bytes, file_name = None, None, None

if load_mode == "Dùng file mặc định":
    if os.path.exists(DEFAULT_PATH):
        try:
            df_raw, file_bytes = load_default_file(DEFAULT_PATH, os.path.getmtime(DEFAULT_PATH))
            file_name = os.path.basename(DEFAULT_PATH)
            st.success(f"Đã tải `{file_name}`")
        except Exception as e:
            st.error(f"Lỗi khi đọc file mặc định: {e}")
    else:
        st.error(f"Không tìm thấy file: `{DEFAULT_PATH}`")
else:
    uploaded = st.file_uploader("Chọn CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df_raw, file_bytes = read_uploaded_file(uploaded)
            file_name = uploaded.name
            st.success(f"Đã tải `{file_name}`")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

if df_raw is None:
    if st.session_state.get("raw_data") is not None:
        df_raw = st.session_state["raw_data"]
    else:
        st.info("Chưa có dữ liệu. Vui lòng chọn nguồn dữ liệu.")
        st.stop()

# Metadata file: chỉ cập nhật "ngày tải" khi dữ liệu THỰC SỰ đổi (hash khác)
if file_bytes is not None:
    new_hash = calculate_sha256(file_bytes)
    if st.session_state.get("data_file_hash") != new_hash:
        st.session_state["data_loaded_at"] = datetime.now()
    st.session_state["data_file_name"] = file_name
    st.session_state["data_file_hash"] = new_hash

df_raw, date_col, n_bad_dates, date_note = parse_date_column(df_raw)
st.session_state["raw_data"] = df_raw
st.session_state["date_col"] = date_col if date_col is not None else "date"

if date_col is None:
    st.warning("Không tìm thấy cột ngày (date/datetime/timestamp): các kiểm tra theo thời gian bị vô hiệu.")
if date_note:
    st.info(f"Cột `{date_col}`: {date_note}")
if n_bad_dates:
    st.warning(f"{n_bad_dates:,} giá trị ở cột `{date_col}` không parse được thành ngày (NaT).")

# ---- Các đại lượng dùng chung giữa các tab (đọc từ tệp, không nhập tay) ----
dates_unique = get_unique_dates(df_raw, date_col)
date_min = dates_unique.min() if len(dates_unique) else None
date_max = dates_unique.max() if len(dates_unique) else None
frequency, mode_delta = infer_frequency(dates_unique)

cal_mode = CAL_ALL
if frequency == "Daily":
    cal_mode = st.radio(
        "Lịch kỳ vọng cho dữ liệu ngày", [CAL_AUTO, CAL_ALL, CAL_BUSINESS], horizontal=True,
        help="Giá thu mua thường không công bố cuối tuần/lễ. Chọn 'Tất cả các ngày' sẽ coi các ngày đó là 'thiếu'.")
    if cal_mode == CAL_AUTO:
        _share = pd.Series(dates_unique.dayofweek).value_counts(normalize=True)
        _keep = sorted(int(i) for i in _share[_share >= 0.05].index)
        st.caption("Các thứ có trong dữ liệu (0 = Thứ 2 … 6 = CN): " + ", ".join(map(str, _keep)) +
                   ". Ngày ngoài tập này không bị tính là thiếu.")

expected_idx = build_expected_index(dates_unique, frequency, mode_delta, cal_mode)
n_exp = len(expected_idx) if expected_idx is not None else None
n_provinces = int(df_raw[PROVINCE_COL].nunique()) if PROVINCE_COL in df_raw.columns else 0
n_coffee_types = int(df_raw[COFFEE_TYPE_COL].nunique()) if COFFEE_TYPE_COL in df_raw.columns else 0
panel_balanced = None
quality_summary = {}
missing_df = pd.DataFrame()
loaded_at = st.session_state.get("data_loaded_at")
numeric_cols = [c for c in df_raw.select_dtypes(include=np.number).columns if c not in get_group_cols(df_raw)]

st.session_state["data_frequency"] = frequency
st.session_state["data_date_min"] = date_min
st.session_state["data_date_max"] = date_max

missing_patterns = compute_missing_patterns(df_raw, date_col)
st.session_state["missing_masks"] = {"original_missing": df_raw.isna()}

# Tần suất thực của từng biến (giá trị lặp liên tiếp)
if frequency == "Monthly":
    _delta_days = 30.4375
elif mode_delta is not None:
    _delta_days = mode_delta.total_seconds() / 86400
else:
    _delta_days = None
repeat_df = compute_repeat_stats(df_raw, date_col, _delta_days)


# ============================================================
# TABS
# ============================================================
tab_a, tab_b, tab_c = st.tabs([
    "A — Dataset Overview", "B — Missing Data", "C — Data Quality",
])


# ============================================================
# TAB A
# ============================================================
with tab_a:
    st.header("A. Dataset Overview")
    st.markdown("Tổng quan cấu trúc, quy mô, tần suất và phân bố của dữ liệu.")

    st.subheader("A.1 Quy mô dữ liệu")
    rows, columns = df_raw.shape
    memory_mb = df_raw.memory_usage(deep=True).sum() / 1024 ** 2

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{columns:,}")
    c3.metric("Memory", f"{memory_mb:.2f} MB")
    c4.metric("Date min", date_min.strftime("%d/%m/%Y") if date_min is not None else "N/A")
    c5.metric("Date max", date_max.strftime("%d/%m/%Y") if date_max is not None else "N/A")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frequency", frequency or "Không xác định")
    c2.metric("Số tỉnh/khu vực", f"{n_provinces:,}")
    c3.metric("Số loại cà phê", f"{n_coffee_types:,}")
    c4.metric("Expected observations", f"{n_exp:,}" if n_exp is not None else "N/A")
    st.caption("N_exp = số ngày trên lưới kỳ vọng (theo lịch đã chọn ở mục 0); dữ liệu tháng dùng lưới lịch tháng "
               "vì Δt của tháng không cố định. Khoảng ngày được đọc trực tiếp từ tệp.")

    st.subheader("A.2 Phân bố kiểu dữ liệu")
    dtype_summary = get_dtype_summary(df_raw)
    st.dataframe(pd.DataFrame({"Data type": list(dtype_summary.keys()),
                               "Số biến": list(dtype_summary.values())}),
                 hide_index=True, **STRETCH)

    st.subheader("A.3 Cấu trúc panel")
    if PROVINCE_COL in df_raw.columns:
        by_prov = (df_raw.groupby(PROVINCE_COL).size().reset_index(name="Số quan sát")
                   .sort_values("Số quan sát", ascending=False))
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Số quan sát theo tỉnh**")
            st.dataframe(by_prov, hide_index=True, **STRETCH)
        balanced_prov = by_prov["Số quan sát"].nunique() == 1
        balanced_type = True
        if COFFEE_TYPE_COL in df_raw.columns:
            by_pt = (df_raw.groupby([PROVINCE_COL, COFFEE_TYPE_COL]).size().reset_index(name="Số quan sát"))
            by_type = df_raw.groupby(COFFEE_TYPE_COL).size().reset_index(name="Số quan sát")
            with cB:
                st.markdown("**Số quan sát theo loại cà phê**")
                st.dataframe(by_type, hide_index=True, **STRETCH)
            st.markdown("**Số quan sát theo (tỉnh, loại cà phê)**")
            st.dataframe(by_pt, hide_index=True, **STRETCH)
            balanced_type = by_pt["Số quan sát"].nunique() == 1
        panel_balanced = bool(balanced_prov and balanced_type)
        if panel_balanced:
            st.success("Panel cân bằng: mọi n_i bằng nhau (theo tỉnh" +
                       (" và theo tỉnh × loại)." if COFFEE_TYPE_COL in df_raw.columns else ")."))
        else:
            st.warning("Panel không cân bằng: n_i khác nhau" +
                       ("" if balanced_prov else " giữa các tỉnh") +
                       ("" if balanced_type else " / giữa các cặp (tỉnh, loại)") + ".")
    else:
        st.info(f"Không tìm thấy cột `{PROVINCE_COL}`, không thể đánh giá cấu trúc panel.")

    st.subheader("A.4 Thống kê mô tả")
    if numeric_cols:
        desc_df = calculate_descriptive_statistics(df_raw, numeric_cols)
        st.dataframe(
            desc_df.style.format({k: "{:.4f}" for k in
                                  ["Mean", "Median", "Std", "Min", "Max", "Skewness", "Kurtosis (excess)", "CV"]}),
            hide_index=True, **STRETCH)
        st.caption("Thống kê gộp (pooled) trên toàn panel nên Std/CV bao gồm cả chênh lệch giữa các tỉnh/loại. "
                   "Skewness g1 = m3/m2^(3/2); excess kurtosis g2 = m4/m2² − 3 (moment trung tâm). "
                   "Bảng dùng để mô tả và phát hiện chênh lệch thang đo; kết luận dạng phân phối dựa vào "
                   "kiểm định Jarque–Bera (Trang 16.4), không kết luận bằng mắt.")

        sel_var = st.selectbox("Chọn biến để xem histogram + KDE", numeric_cols)
        s_sel = pd.to_numeric(df_raw[sel_var], errors="coerce").dropna()
        if len(s_sel) > 1:
            st.altair_chart(hist_kde_chart(s_sel, sel_var))
        if st.checkbox("Hiển thị histogram + KDE thu nhỏ cho tất cả biến số"):
            cols3 = st.columns(3)
            for i, col in enumerate(numeric_cols[:24]):
                ss = pd.to_numeric(df_raw[col], errors="coerce").dropna()
                if len(ss) > 1:
                    with cols3[i % 3]:
                        st.caption(col)
                        st.altair_chart(hist_kde_chart(ss, col, height=120))
            if len(numeric_cols) > 24:
                st.caption("Chỉ hiển thị 24 biến đầu tiên.")
    else:
        st.info("Không tìm thấy biến số.")

    st.subheader("A.5 Dấu vân tay dữ liệu")
    fp1, fp2 = st.columns(2)
    fp1.text_input("Tên file", value=st.session_state.get("data_file_name") or "", disabled=True)
    fp2.text_input("SHA-256", value=st.session_state.get("data_file_hash") or "Chưa xác định", disabled=True)
    fp3, fp4 = st.columns(2)
    fp3.text_input("Ngày/giờ tải dữ liệu",
                   value=loaded_at.strftime("%d/%m/%Y %H:%M:%S") if loaded_at else "", disabled=True)
    fp4.text_input("Phiên bản Data Passport", value=PASSPORT_VERSION, disabled=True)
    st.caption("SHA-256 tính trực tiếp trên bytes của file đầu vào. Cùng SHA-256 nghĩa là cùng dữ liệu đầu vào; "
               "dùng cho tái lập ở Trang 17. Ngày tải chỉ đổi khi hash đổi.")


# ============================================================
# TAB B
# ============================================================
with tab_b:
    st.header("B. Missing Data")
    st.markdown("Phân tích mức độ thiếu, khoảng trống liên tiếp và nguy cơ look-ahead khi điền dữ liệu.")

    st.subheader("B.1 Bảng khuyết")
    missing_df = compute_missing_report(df_raw, date_col)
    st.dataframe(missing_df.style.format({"Missing %": "{:.3f}%", "Median gap": "{:.2f}"}),
                 hide_index=True, **STRETCH)
    st.caption("Gap = số quan sát liên tiếp bị khuyết (run-length), tính TRONG từng chuỗi (tỉnh × loại) "
               "đã sắp theo ngày, không vắt qua ranh giới giữa các tỉnh.")
    if int(missing_df["Missing"].sum()) == 0:
        st.info("Dữ liệu không có ô khuyết nào. Các mục B.3–B.4 sẽ trống; B.6 chỉ là thí nghiệm giả định "
                "(không có gap thật để mô phỏng).")

    st.subheader("B.2 Ngưỡng cảnh báo gap")
    horizon_warning = st.number_input(
        "Horizon dùng để cảnh báo gap", min_value=1, value=1, step=1,
        help="Đây là tham số của ứng dụng, không phải ngưỡng học thuật.")
    large_gap = missing_df[missing_df["Max gap"] > horizon_warning]
    if len(large_gap) > 0:
        st.warning(f"Có {len(large_gap)} biến có gap lớn hơn horizon = {horizon_warning}: "
                   + ", ".join(map(str, large_gap["Variable"])))
    st.caption(f"Ngưỡng cảnh báo hiện tại: gap > {horizon_warning}. Đây là tham số ứng dụng, "
               "không phải chuẩn học thuật.")

    st.subheader("B.3 Heatmap khuyết")
    if date_col:
        miss_cols = [c for c in df_raw.columns if c != date_col and df_raw[c].isna().any()]
        if miss_cols:
            mm = df_raw[miss_cols].isna().astype(float)
            mm["Period"] = df_raw[date_col].dt.to_period("M").dt.to_timestamp()
            long_df = (mm.groupby("Period")[miss_cols].mean().reset_index()
                       .melt(id_vars="Period", var_name="Variable", value_name="Missing rate"))
            heat = alt.Chart(long_df).mark_rect().encode(
                x=alt.X("yearmonth(Period):T", title="Tháng"),
                y=alt.Y("Variable:N", title=None),
                color=alt.Color("Missing rate:Q", scale=alt.Scale(domain=[0, 1], scheme="reds"),
                                title="Tỷ lệ khuyết"),
                tooltip=["Variable:N", alt.Tooltip("yearmonth(Period):T", title="Tháng"),
                         alt.Tooltip("Missing rate:Q", format=".1%")],
            ).properties(height=max(120, 28 * len(miss_cols)))
            st.altair_chart(heat)
            st.caption("Màu đậm = tỷ lệ khuyết cao trong tháng đó (biến × thời gian).")

            cells = df_raw.drop(columns=[date_col]).isna().sum(axis=1)
            by_month = cells.groupby(df_raw[date_col].dt.to_period("M")).sum()
            by_month.index = by_month.index.to_timestamp()
            st.markdown("**Tổng ô khuyết theo tháng**")
            st.bar_chart(by_month.rename("Missing cells"))

            wd_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"]
            by_wd = cells.groupby(df_raw[date_col].dt.dayofweek).sum().reindex(range(7), fill_value=0)
            by_wd.index = wd_names
            st.markdown("**Tổng ô khuyết theo ngày trong tuần**")
            st.bar_chart(by_wd.rename("Missing cells"))
        else:
            st.success("Không có biến nào bị khuyết.")
    else:
        st.info("Cần cột ngày để vẽ heatmap khuyết.")

    st.subheader("B.4 Kiểm định χ² về quy luật missing (thăm dò)")
    if SCIPY_AVAILABLE and date_col:
        chi_var = st.selectbox("Chọn biến", [c for c in df_raw.columns if c != date_col], key="chi_missing_var")
        by = st.radio("Nhóm thời gian", list(MISSING_BY_OPTIONS), horizontal=True)
        res = chi_square_missing(df_raw, chi_var, date_col, by)
        if res is None:
            st.info("Biến này không có khuyết (hoặc khuyết toàn bộ), hoặc nhóm thời gian chỉ có 1 giá trị "
                    "nên không lập được bảng chéo 2×2 trở lên.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("χ²", f"{res['chi2']:.4f}")
            c2.metric("p-value", f"{res['p']:.6f}")
            c3.metric("df", f"{res['dof']}")
            c4.metric("Cramér's V", f"{res['cramers_v']:.3f}")
            if res["pct_low_expected"] > 20:
                st.caption(f"⚠ {res['pct_low_expected']:.0f}% ô có tần số kỳ vọng < 5: xấp xỉ χ² kém tin cậy.")
            if res["p"] < 0.05:
                st.warning("p < 0.05 (chưa hiệu chỉnh đa kiểm định): khuyết có dấu hiệu không ngẫu nhiên hoàn toàn. "
                           "Hãy xem Cramér's V để biết hiệu ứng có đáng kể không.")
            else:
                st.info("p ≥ 0.05: chưa có bằng chứng về quy luật khuyết theo nhóm thời gian đã kiểm tra.")
        st.caption("Chỉ báo khuyết được gộp về MỨC NGÀY (ngày có ≥ 1 chuỗi khuyết) để tránh các quan sát cùng ngày "
                   "trong panel bị đếm như độc lập. Đây là kiểm tra thăm dò, không thay cho kiểm định MCAR/MAR chính thức; "
                   "kết quả đưa vào Data Passport dùng ngưỡng Bonferroni theo số phép thử.")
        with st.expander("Kết quả χ² cho mọi biến (đưa vào Data Passport)"):
            st.dataframe(pd.DataFrame({"Variable": list(missing_patterns.keys()),
                                       "Missing pattern (χ²)": list(missing_patterns.values())}),
                         hide_index=True, **STRETCH)
    else:
        st.info("Cần scipy và cột ngày để thực hiện kiểm định χ².")

    st.subheader("B.5 Cảnh báo rò rỉ khi điền missing")
    st.dataframe(pd.DataFrame([
        {"Phương pháp": "Forward-fill", "Loại": "Nhân quả", "Look-ahead": "Thấp",
         "Quy tắc": "Dùng được nếu thông tin tại thời điểm t đã biết."},
        {"Phương pháp": "Rolling mean quá khứ", "Loại": "Nhân quả", "Look-ahead": "Thấp",
         "Quy tắc": "Chỉ sử dụng dữ liệu quá khứ."},
        {"Phương pháp": "Linear interpolation", "Loại": "Không nhân quả", "Look-ahead": "Trung bình/Cao",
         "Quy tắc": "Chỉ dùng bên trong TRAIN (nội suy tại t dùng giá trị sau t)."},
        {"Phương pháp": "Backward-fill", "Loại": "Không nhân quả", "Look-ahead": "Cao",
         "Quy tắc": "Không được vắt qua ranh giới train/val/test."},
        {"Phương pháp": "Centered window", "Loại": "Không nhân quả", "Look-ahead": "Cao",
         "Quy tắc": "Không dùng cho dự báo tại thời điểm thực."},
        {"Phương pháp": "Spline", "Loại": "Không nhân quả", "Look-ahead": "Cao",
         "Quy tắc": "Chỉ dùng trong TRAIN nếu được phép."},
    ]), hide_index=True, **STRETCH)
    st.warning("Quy tắc: phương pháp không nhân quả chỉ dùng bên trong đoạn TRAIN, cấm vắt qua ranh giới "
               "TRAIN/VAL/TEST. Mask các ô khuyết gốc đã được lưu vào session_state['missing_masks'] "
               "để Trang 13 kiểm tra; trang này không tự điền dữ liệu.")

    st.subheader("B.6 Kiểm chứng phương pháp điền")
    st.markdown(
        "Thí nghiệm **khuyết nhân tạo**: ẩn ngẫu nhiên các đoạn giá trị thật theo đúng phân phối độ dài gap "
        "quan sát được, điền lại bằng từng phương pháp rồi so với giá trị thật. "
        "MAE_imp = (1/m) Σ|x̂ᵢ − xᵢ|, RMSE_imp tương tự; thêm MAPE (so sánh giữa các biến) và SE_MAE "
        "(sai số chuẩn giữa các đoạn ẩn) để biết chênh lệch giữa phương pháp có vượt nhiễu không. "
        "Báo cáo theo nhóm gap 1, 2–3, 4–7, > 7."
    )
    exp_cols = numeric_cols
    if exp_cols:
        price_cols_all = choose_price_columns(df_raw)
        default_idx = exp_cols.index(price_cols_all[0]) if price_cols_all and price_cols_all[0] in exp_cols else 0
        e1, e2, e3, e4 = st.columns(4)
        exp_var = e1.selectbox("Biến", exp_cols, index=default_idx, key="imp_var")
        train_frac = e2.slider("Tỷ lệ đoạn đầu dùng làm TRAIN", 0.5, 0.9, 0.7, 0.05,
                               help="Tham số ứng dụng: thí nghiệm chỉ chạy trên đoạn đầu của mỗi chuỗi.")
        n_trials = e3.number_input("Số lần ẩn (trials)", 50, 3000, 300, 50)
        win = e4.number_input("Cửa sổ trung bình trượt", 3, 60, 5, 1)
        _pat = missing_patterns.get(exp_var, "")
        if _pat.startswith("Có dấu hiệu"):
            st.warning(f"B.4 thấy dấu hiệu khuyết có quy luật ở `{exp_var}` ({_pat}). Thí nghiệm này ẩn "
                       "NGẪU NHIÊN HOÀN TOÀN (MCAR) nên có thể không phản ánh đúng cơ chế khuyết thật.")
        if st.button("Chạy thí nghiệm khuyết nhân tạo", type="primary"):
            with st.spinner("Đang chạy thí nghiệm..."):
                res_df, info = run_imputation_experiment(
                    df_raw, exp_var, date_col, train_frac, int(n_trials), int(win), seed=42)
            st.session_state["imputation_result"] = {"var": exp_var, "table": res_df, "info": info}
        r = st.session_state.get("imputation_result")
        if r and not r["table"].empty:
            st.markdown(f"**Kết quả cho `{r['var']}`** ({r['info']['done']} lần ẩn)")
            if r["info"]["fallback"]:
                st.warning("Biến này không có gap quan sát nên dùng phân phối độ dài gap mặc định "
                           "{1,2,3,5,8}; đây là thí nghiệm GIẢ ĐỊNH, chưa phải phân phối thực của dữ liệu.")
            st.dataframe(r["table"].style.format({"MAE_imp": "{:.4f}", "SE_MAE": "{:.4f}", "RMSE_imp": "{:.4f}",
                                                  "MAPE %": "{:.2f}", "Coverage %": "{:.0f}"}),
                         hide_index=True, **STRETCH)
            best_rows = []
            tbl = r["table"].dropna(subset=["MAE_imp"])
            for g, sub in tbl.groupby("Gap", sort=False):
                sub = sub.sort_values("MAE_imp")
                b = sub.iloc[0]
                if len(sub) > 1:
                    s2 = sub.iloc[1]
                    se = np.hypot(b["SE_MAE"], s2["SE_MAE"])
                    if pd.isna(se):
                        sig = "Không đủ mẫu"
                    else:
                        sig = "Có" if (s2["MAE_imp"] - b["MAE_imp"]) > 2 * se else "Không (trong sai số)"
                else:
                    sig = ""
                bc = sub[sub["Loại"] == "Nhân quả"]
                best_rows.append({"Nhóm gap": g, "Tốt nhất (MAE)": b["Method"], "Loại": b["Loại"],
                                  "Hơn hạng 2 có ý nghĩa?": sig,
                                  "Tốt nhất nhân quả": bc.iloc[0]["Method"] if not bc.empty else ""})
            st.markdown("**Phương pháp có sai số nhỏ nhất theo từng nhóm gap**")
            st.dataframe(pd.DataFrame(best_rows), hide_index=True, **STRETCH)
            st.caption("'Có ý nghĩa' nghĩa là MAE hơn hạng 2 vượt 2 lần sai số chuẩn kết hợp (quy tắc thô). "
                       "Nếu phương pháp tốt nhất là không nhân quả thì chỉ được dùng bên trong TRAIN. "
                       "Coverage < 100% nghĩa là phương pháp không điền được hết các ô bị ẩn. "
                       "Biểu đồ KDE trước/sau chỉ là kiểm tra phụ: hai phân phối giống nhau vẫn có thể điền sai giá trị.")
        elif r is not None:
            st.info("Không đủ đoạn hợp lệ để chạy thí nghiệm (chuỗi quá ngắn hoặc khuyết quá nhiều).")
    else:
        st.info("Không có biến số để chạy thí nghiệm.")


# ============================================================
# TAB C
# ============================================================
with tab_c:
    st.header("C. Data Quality")
    st.markdown("Tab này chỉ kiểm tra **toàn vẹn dữ liệu**. ADF/KPSS thuộc Trang 5 và VIF thuộc Trang 4.7 "
                "để tránh hai kết quả lệch nhau (toàn mẫu so với train).")

    # ---- C.1 ----
    st.subheader("C.1 Trùng lặp")
    dup_rows = int(df_raw.duplicated().sum())
    st.metric("Duplicate rows", f"{dup_rows:,}")
    if dup_rows == 0:
        st.success("Không phát hiện duplicate rows.")
    else:
        st.warning(f"Phát hiện {dup_rows:,} duplicate rows.")
        st.dataframe(df_raw[df_raw.duplicated(keep=False)], **STRETCH)

    key_cols = get_group_cols(df_raw) + ([date_col] if date_col else [])
    dup_key = 0
    if key_cols and date_col:
        n_unique_keys = len(df_raw.drop_duplicates(subset=key_cols))
        dup_key = len(df_raw) - n_unique_keys  # n_dup = N − số khóa duy nhất
        st.metric(f"Duplicate ({', '.join(key_cols)})", f"{dup_key:,}")
        if dup_key > 0:
            st.warning("Các dòng có khóa trùng:")
            st.dataframe(df_raw[df_raw.duplicated(subset=key_cols, keep=False)].sort_values(key_cols), **STRETCH)
        else:
            st.success("Không có khóa (tỉnh, loại, ngày) trùng.")
    quality_summary.update({"duplicate_rows": dup_rows, "duplicate_keys": int(dup_key)})

    # ---- C.2 ----
    st.subheader("C.2 Giá trị không hợp lệ")
    invalid_df, invalid_masks = calculate_invalid_domain_flags(df_raw)
    if invalid_df.empty:
        st.success("Không phát hiện giá trị vi phạm các domain rule tự động.")
    else:
        st.warning("Các giá trị dưới đây chỉ được gắn cờ, không tự động xóa; người dùng quyết định.")
        st.dataframe(invalid_df, hide_index=True, **STRETCH)
        opts = [f"{v} — {r}" for (v, r) in invalid_masks]
        pick = st.selectbox("Xem các dòng bị gắn cờ", opts)
        st.dataframe(df_raw[list(invalid_masks.values())[opts.index(pick)]], **STRETCH)
    st.caption("Domain rule nhiệt độ [-90,70] chỉ là khoảng vật lý rộng của ứng dụng; khi dùng trong luận văn "
               "cần ghi rõ nguồn/tiêu chuẩn (ví dụ cơ quan khí tượng).")
    quality_summary["invalid_domain_flags"] = df_to_records(invalid_df) if not invalid_df.empty else []

    # ---- C.3 ----
    st.subheader("C.3 Ngoại lai ứng viên")
    price_cols = choose_price_columns(df_raw)
    quality_summary["outlier_candidates"] = None
    if price_cols:
        o1, o2, o3 = st.columns(3)
        sel_price = o1.selectbox("Chọn biến giá", price_cols, key="price_outlier_var")
        o_method = o2.radio("Biến đổi", ["Sai phân bậc 1", "Log-return"], horizontal=True)
        o_scale = o3.radio("Thang đo độ phân tán", [SCALE_GLOBAL, SCALE_ROLLING], horizontal=True,
                           help="Giá cà phê có biến động theo cụm; thang cục bộ tránh gắn cờ hàng loạt ở giai đoạn biến động cao.")
        p1, p2 = st.columns(2)
        o_thr = p1.number_input("Ngưỡng |M_i|", 2.0, 10.0, 3.5, 0.5,
                                help="Tham số ứng dụng; 3.5 là gợi ý của Iglewicz & Hoaglin (1993).")
        o_win = p2.number_input("Cửa sổ rolling (số kỳ)", 20, 500, 60, 10, disabled=(o_scale == SCALE_GLOBAL))

        d_sorted, diff_s, score_s, o_info = outlier_scores(
            df_raw, sel_price, date_col, o_method, o_scale, int(o_win))
        flagged = score_s.abs() > o_thr
        n_flag = int(flagged.sum())

        zero_share = float((diff_s.dropna() == 0).mean() * 100) if diff_s.notna().any() else np.nan
        n_valid = max(o_info["n_valid"], 1)
        k1, k2, k3 = st.columns(3)
        k1.metric("Ứng viên ngoại lai", f"{n_flag:,}")
        k2.metric("Sai phân = 0", f"{zero_share:.1f}%" if pd.notna(zero_share) else "N/A")
        k3.metric("Điểm tính được", f"{100 * (o_info['n_mad'] + o_info['n_meanad']) / n_valid:.1f}%")
        if o_info["n_meanad"] > 0:
            st.info(f"{100 * o_info['n_meanad'] / n_valid:.1f}% điểm có MAD = 0 (giá đứng yên nhiều kỳ) nên được tính "
                    "bằng công thức dự phòng M = (x − median)/(1.253314·MeanAD) theo Iglewicz & Hoaglin.")
        if o_info["n_mad"] + o_info["n_meanad"] < o_info["n_valid"]:
            st.warning("Có điểm không tính được (chuỗi quá ngắn hoặc toàn sai phân bằng nhau): "
                       "kết quả 0 ứng viên KHÔNG có nghĩa là dữ liệu sạch.")
        quality_summary["outlier_candidates"] = {
            "variable": sel_price, "method": o_method, "scale": o_scale, "threshold": float(o_thr),
            "count": n_flag, "zero_diff_share_pct": zero_share,
            "points_scored_pct": 100 * (o_info["n_mad"] + o_info["n_meanad"]) / n_valid}

        if n_flag > 0:
            show_cols = [c for c in [date_col] + get_group_cols(d_sorted) + [sel_price] if c]
            out_tbl = d_sorted.loc[flagged, show_cols].copy()
            out_tbl[o_method] = diff_s[flagged]
            out_tbl["M_i"] = score_s[flagged]
            out_tbl = out_tbl.reindex(out_tbl["M_i"].abs().sort_values(ascending=False).index).head(500)
            st.dataframe(out_tbl, hide_index=True, **STRETCH)

        if date_col:
            gcols = get_group_cols(d_sorted)
            label = (d_sorted[gcols].astype(str).agg(" | ".join, axis=1) if gcols
                     else pd.Series("Toàn bộ", index=d_sorted.index))
            pick_g = st.selectbox("Chuỗi hiển thị biểu đồ", sorted(label.unique()), key="outlier_chart_group")
            sub = d_sorted[label == pick_g]
            plot = pd.DataFrame({"Date": sub[date_col], "Value": pd.to_numeric(sub[sel_price], errors="coerce"),
                                 "Flag": flagged.loc[sub.index]})
            line = alt.Chart(plot).mark_line().encode(x=alt.X("Date:T", title="Ngày"),
                                                      y=alt.Y("Value:Q", title=sel_price, scale=alt.Scale(zero=False)))
            pts = alt.Chart(plot[plot["Flag"]]).mark_point(color="red", filled=True, size=70).encode(
                x="Date:T", y="Value:Q", tooltip=["Date:T", "Value:Q"])
            st.altair_chart(line + pts)
        st.caption("M_i tính trên sai phân bậc 1 (hoặc log-return) TRONG từng chuỗi (tỉnh × loại), không tính trên "
                   "mức giá: M_i = 0.6745 × (x_i − median)/MAD, hoặc công thức MeanAD khi MAD = 0 "
                   "(Iglewicz & Hoaglin, 1993). Log-return nên dùng khi giá thay đổi mạnh theo thời gian.")
        st.warning("Nhãn là 'ứng viên ngoại lai', không phải 'lỗi': cú sốc giá cà phê có thể là sự kiện thật. "
                   "Chỉ loại bỏ khi đối chiếu được nguồn thứ hai.")
    else:
        st.info("Không tự động tìm thấy biến số có tên liên quan đến giá.")

    # ---- C.4 ----
    st.subheader("C.4 Khoảng trống ngày và tần suất")
    if expected_idx is not None:
        missing_dates = expected_idx.difference(dates_unique)
        n_obs = len(dates_unique)
        n_missing_dates = len(missing_dates)
        off_grid = n_obs - (n_exp - n_missing_dates)

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected dates", f"{n_exp:,}")
        c2.metric("Observed dates", f"{n_obs:,}")
        c3.metric("Missing dates", f"{n_missing_dates:,}")
        if frequency == "Daily":
            st.caption(f"Lịch kỳ vọng đang dùng: **{cal_mode}** (đổi ở mục 0).")
        if off_grid > 0:
            st.caption(f"Có {off_grid} ngày quan sát nằm ngoài lưới kỳ vọng (lệch lịch).")

        if n_missing_dates > 0:
            st.warning(f"Missing dates ⚠ {n_missing_dates}")
            st.markdown("**Các đoạn thiếu**")
            st.dataframe(missing_segments(expected_idx, missing_dates), hide_index=True, **STRETCH)
            with st.expander(f"Xem chính xác {n_missing_dates} ngày bị thiếu"):
                st.dataframe(pd.DataFrame({"Missing date": missing_dates.date}), hide_index=True, **STRETCH)
        else:
            st.success("Không phát hiện ngày thiếu trong khoảng thời gian kỳ vọng.")

        abn = abnormal_delta_pairs(dates_unique, frequency, mode_delta, expected_idx)
        st.metric("Cặp ngày có Δt bất thường", f"{len(abn):,}")
        if len(abn) > 0:
            with st.expander("Xem các cặp ngày có Δt bất thường"):
                st.dataframe(abn, hide_index=True, **STRETCH)
        if frequency == "Monthly":
            st.caption("Dữ liệu tháng: Δt hợp lệ là 28–31 ngày nên được coi là nhất quán.")

        gcols = get_group_cols(df_raw)
        if gcols and date_col:
            rows_g = []
            for k, g in df_raw.groupby(gcols, dropna=False):
                gd = pd.DatetimeIndex(g[date_col].dropna().unique())
                rows_g.append({**dict(zip(gcols, k if isinstance(k, tuple) else (k,))),
                               "Observed dates": len(gd),
                               "Missing so với lưới toàn cục": len(expected_idx.difference(gd))})
            with st.expander("Ngày thiếu theo từng chuỗi (tỉnh × loại)"):
                st.dataframe(pd.DataFrame(rows_g), hide_index=True, **STRETCH)
        quality_summary.update({"expected_dates": int(n_exp), "observed_dates": int(n_obs),
                                "missing_dates": int(n_missing_dates), "abnormal_delta_pairs": int(len(abn)),
                                "calendar": cal_mode if frequency == "Daily" else None,
                                "unparsed_dates": int(n_bad_dates)})
    else:
        st.info("Không thể xác định khoảng trống ngày (thiếu cột ngày hoặc < 2 ngày hợp lệ).")

    # ---- C.5 ----
    st.subheader("C.5 Độ lớn các biến giá (kiểm tra nhất quán đơn vị)")
    st.markdown("Bảng dưới cho thấy độ lớn thực của từng biến giá để đối chiếu bằng mắt. Hai biến cùng loại "
                "nhưng median lệch nhau hàng nghìn lần thường là khác đơn vị (ví dụ USD/tấn so với VND/kg).")
    unit_rows = []
    for col in numeric_cols:
        if name_matches(col, KW_PRICE):
            s_ = pd.to_numeric(df_raw[col], errors="coerce")
            unit_rows.append({"Variable": col, "Median": s_.median(), "Min": s_.min(), "Max": s_.max(),
                              "Max/Min": (s_.max() / s_.min()) if s_.min() and s_.min() > 0 else np.nan})
    if unit_rows:
        unit_df = pd.DataFrame(unit_rows)
        st.dataframe(unit_df.style.format({"Median": "{:,.2f}", "Min": "{:,.2f}", "Max": "{:,.2f}",
                                           "Max/Min": "{:,.2f}"}), hide_index=True, **STRETCH)
        quality_summary["price_magnitude"] = df_to_records(unit_df)
    else:
        st.info("Không tìm thấy biến giá để so sánh độ lớn.")

    # ---- C.6 ----
    st.subheader("C.6 Tần suất thực của từng biến (giá trị lặp / đứng yên)")
    st.markdown("Tần suất của bảng có thể là hàng ngày nhưng một số biến thực chất chỉ cập nhật theo tuần/tháng "
                "(bị lặp giá trị xuống). Với biến ngoại sinh X, đây là dấu hiệu của độ trễ công bố.")
    if not repeat_df.empty:
        rep_show = repeat_df.copy()
        rep_show["Cảnh báo"] = np.where(rep_show["Tỷ lệ lặp liên tiếp (%)"] >= REPEAT_FLAG_SHARE,
                                        "⚠ nhiều khả năng tần suất thấp hơn bảng / giá đứng yên", "")
        st.dataframe(rep_show.style.format({"Tỷ lệ lặp liên tiếp (%)": "{:.1f}",
                                            "Chuỗi lặp dài nhất (ngày)": "{:,.0f}"}),
                     hide_index=True, **STRETCH)
        st.caption(f"Ngưỡng cảnh báo {REPEAT_FLAG_SHARE:.0f}% là tham số ứng dụng. Giá nội địa có thể đứng yên thật "
                   "(không phải lỗi); biến thời tiết/vĩ mô lặp nhiều thường là dữ liệu tháng bị nội suy xuống ngày. "
                   "Chuỗi lặp dài nhất cho biết thời gian một biến có thể không được cập nhật.")
        quality_summary["repeat_values"] = df_to_records(repeat_df)
    else:
        st.info("Không có biến số để kiểm tra.")


# ============================================================
# DATA PASSPORT (chỉ gồm các đại lượng tính từ dữ liệu)
# ============================================================
st.divider()

imp = st.session_state.get("imputation_result")

passport = {
    "version": PASSPORT_VERSION,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "overview": {
        "file_name": st.session_state.get("data_file_name"),
        "sha256": st.session_state.get("data_file_hash"),
        "loaded_at": loaded_at.isoformat(timespec="seconds") if loaded_at else None,
        "rows": int(df_raw.shape[0]),
        "columns": int(df_raw.shape[1]),
        "date_min": str(date_min.date()) if date_min is not None else None,
        "date_max": str(date_max.date()) if date_max is not None else None,
        "frequency": frequency,
        "calendar": cal_mode if frequency == "Daily" else None,
        "expected_observations": int(n_exp) if n_exp is not None else None,
        "n_provinces": n_provinces,
        "n_coffee_types": n_coffee_types,
        "panel_balanced": panel_balanced,
    },
    "missing_report": df_to_records(missing_df) if len(missing_df) else [],
    "missing_patterns": missing_patterns,
    "imputation_experiment": ({"variable": imp["var"], "info": imp["info"], "table": df_to_records(imp["table"])}
                              if imp and not imp["table"].empty else None),
    "quality": quality_summary,
}
st.session_state["data_passport"] = passport

st.subheader("Data Passport")
st.success("Data Passport đã được cập nhật trong session state.")
st.download_button(
    "⬇ Tải Data Passport (JSON)",
    data=json.dumps(passport, ensure_ascii=False, indent=2, default=str),
    file_name="data_passport.json", mime="application/json",
)


# ============================================================
# NAVIGATION
# ============================================================
st.divider()
nav1, nav2 = st.columns(2)
with nav1:
    if st.button("← Về Home", key="nav_home", **STRETCH):
        st.switch_page("Home.py")
with nav2:
    if st.button("Tiếp theo: Phân tích tương quan →", key="nav_next", **STRETCH):
        st.switch_page("pages/2_Phan_tich_tuong_quan.py")
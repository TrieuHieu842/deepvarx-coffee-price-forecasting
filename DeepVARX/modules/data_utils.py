"""Tiện ích dữ liệu: đọc/ghi, dữ liệu mẫu, khoảng trống, nội suy an toàn, sai phân, metric."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
DATE_CANDIDATES = ["date", "ngay", "ngày", "datetime", "timestamp", "time", "month", "thang", "tháng"]


# ----------------------------------------------------------------------------- nạp dữ liệu
def list_data_files() -> list[Path]:
    files: list[Path] = []
    for folder in (DATA_PROC, DATA_RAW, ROOT / "data"):
        if folder.exists():
            files += sorted(folder.glob("*.csv"))
    seen, uniq = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def load_csv(src) -> pd.DataFrame:
    """Đọc CSV từ đường dẫn hoặc file upload (thử utf-8 rồi utf-8-sig/latin-1)."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_csv(src, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Không đọc được file CSV (thử các encoding utf-8/utf-8-sig/latin-1 đều lỗi).")


def guess_date_col(df: pd.DataFrame) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    for c in df.columns:
        if df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c].head(30), errors="coerce")
                if parsed.notna().mean() > 0.8:
                    return c
            except Exception:  # noqa: BLE001
                pass
    return df.columns[0]


def numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = exclude or []
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def to_series_frame(df: pd.DataFrame, date_col: str, group_col: str | None = None,
                    group_value: str | None = None) -> pd.DataFrame:
    """Chuẩn hoá về DataFrame số, index = ngày (tăng dần, không trùng).
    Nếu là panel: lọc 1 nhóm hoặc lấy trung bình theo ngày."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if group_col and group_value is not None:
        d = d[d[group_col].astype(str) == str(group_value)]
    num = numeric_columns(d, exclude=[date_col])
    out = d.groupby(date_col)[num].mean().sort_index()
    out.index.name = "date"
    return out


# ----------------------------------------------------------------------------- dữ liệu mẫu
def make_demo_data(n: int = 1400, seed: int = 7) -> pd.DataFrame:
    """Dữ liệu mô phỏng (có đủ các "bẫy": đa cộng tuyến, tần suất giả, khoảng trống lớn, phi tuyến)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    t = np.arange(n)

    def lag(a, k):
        return np.r_[np.full(k, a[0]), a[:-k]]

    robusta = 2000 + np.cumsum(rng.normal(0, 18, n))
    usd = 23000 + np.cumsum(rng.normal(0, 15, n))
    brent = 70 + np.cumsum(rng.normal(0, 0.8, n))
    rain = np.clip(120 + 90 * np.sin(2 * np.pi * t / 260 - 1) + rng.normal(0, 25, n), 0, None)
    temp = 26 + 3 * np.sin(2 * np.pi * t / 260) + rng.normal(0, 0.8, n)
    months = np.asarray(dates.to_period("M").astype(str))
    _, month_idx = np.unique(months, return_inverse=True)
    fert_month = 500 + np.cumsum(rng.normal(0, 15, month_idx.max() + 1))
    fertilizer = fert_month[month_idx]

    d_rob, d_usd = np.diff(robusta, prepend=robusta[0]), np.diff(usd, prepend=usd[0])
    rain_z = (lag(rain, 5) - 120) / 90
    dy = 9 * lag(d_rob, 3) + 0.4 * lag(d_usd, 1) + 220 * (rain_z ** 2 - 1) + rng.normal(0, 260, n)
    price = 60000 + np.cumsum(dy)

    gappy = 100 + np.cumsum(rng.normal(0, 1, n))
    gappy[400:700] = np.nan
    gappy[rng.random(n) < 0.05] = np.nan
    usd_s = usd.copy()
    usd_s[rng.random(n) < 0.01] = np.nan

    return pd.DataFrame({
        "date": dates,
        "coffee_price_vnd_kg": price,
        "robusta_futures": robusta,
        "robusta_copy": robusta * (1 + rng.normal(0, 0.002, n)),
        "usd_vnd": usd_s,
        "brent_oil": brent,
        "rainfall_mm": rain,
        "temperature_c": temp,
        "fertilizer_index_monthly": fertilizer,
        "gappy_index": gappy,
    })


# ----------------------------------------------------------------------------- khoảng trống & nội suy
def split_index(n: int, train_ratio: float) -> int:
    return int(n * train_ratio)


def _gap_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Các đoạn NaN liên tiếp (start, length)."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def gap_report(df: pd.DataFrame, train_n: int) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        mask = df[c].isna().to_numpy()
        runs = _gap_runs(mask)
        longest = max(runs, key=lambda r: r[1]) if runs else None
        rows.append({
            "Biến": c,
            "% khuyết (toàn bộ)": round(100 * mask.mean(), 2),
            "% khuyết (TRAIN)": round(100 * mask[:train_n].mean(), 2) if train_n else 0.0,
            "Số đoạn khuyết": len(runs),
            "Max gap (số dòng)": longest[1] if longest else 0,
            "Gap dài nhất bắt đầu": df.index[longest[0]].strftime("%Y-%m-%d") if longest else "",
        })
    return pd.DataFrame(rows)


INTERP_METHODS = {
    "Forward-fill": "ffill",
    "Linear": "linear",
    "Time (theo trục thời gian)": "time",
    "Spline bậc 3": "spline",
}


def interpolate_train_safe(df: pd.DataFrame, train_n: int, method: str) -> tuple[pd.DataFrame, int]:
    """Nội suy chống look-ahead.
    - Đoạn TRAIN: dùng phương pháp được chọn, chỉ nội suy *bên trong* (inside) rồi ffill phần đuôi.
    - Đoạn TEST: chỉ forward-fill (không dùng giá trị tương lai).
    Trả về (df đã cắt các dòng đầu còn NaN, train_n mới)."""
    tr, te = df.iloc[:train_n].copy(), df.iloc[train_n:].copy()
    if method == "ffill":
        tr = tr.ffill()
    elif method == "spline":
        tr = tr.interpolate(method="spline", order=3, limit_area="inside").ffill()
    else:
        tr = tr.interpolate(method=method, limit_area="inside").ffill()
    if len(te):
        te = pd.concat([tr.iloc[[-1]], te]).ffill().iloc[1:]
    out = pd.concat([tr, te])
    firsts = [out[c].first_valid_index() for c in out.columns]
    if any(f is None for f in firsts):  # có cột rỗng hoàn toàn → không dùng được
        return out.iloc[0:0], 0
    start = out.index.get_loc(max(firsts))
    out = out.iloc[start:]
    return out, train_n - start


def zero_diff_ratio(s: pd.Series) -> float:
    d = s.diff().dropna()
    return float((d == 0).mean()) if len(d) else np.nan


# ----------------------------------------------------------------------------- sai phân / scale
def difference(df: pd.DataFrame, d: int) -> pd.DataFrame:
    out = df.copy()
    for _ in range(int(d)):
        out = out.diff()
    return out.iloc[int(d):]


def reconstruct_level(level: np.ndarray, t_idx: np.ndarray, d: int, pred_d: np.ndarray) -> np.ndarray:
    """Dự báo 1 bước ở không gian sai phân -> mức giá gốc (dùng giá thực tại t-1, t-2)."""
    t_idx = np.asarray(t_idx)
    if d == 0:
        return pred_d
    if d == 1:
        return level[t_idx - 1] + pred_d
    if d == 2:
        return 2 * level[t_idx - 1] - level[t_idx - 2] + pred_d
    raise ValueError("Chỉ hỗ trợ d ∈ {0,1,2}")


# ----------------------------------------------------------------------------- metric
def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / np.where(y == 0, np.nan, y))) * 100)


def metrics(y, yhat) -> dict:
    return {"RMSE": rmse(y, yhat), "MAE": mae(y, yhat), "MAPE (%)": mape(y, yhat)}


# ----------------------------------------------------------------------------- xuất dữ liệu huấn luyện
def _with_split(frame: pd.DataFrame, n_train: int) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "split", np.where(np.arange(len(out)) < n_train, "train", "test"))
    return out


def export_datasets(df: pd.DataFrame, target: str, base: list[str], ext: list[str], d: int, train_n: int,
                    out_dir: Path | None = None) -> dict:
    """Sinh 3 file: (1) Tập Cơ Sở sai phân bậc d, (2) Tập Cơ Sở scaled (d=0), (3) Tập Mở Rộng scaled (d=0).
    Standard Scaler chỉ khớp trên TRAIN. Cột `split` đánh dấu train/test."""
    out_dir = Path(out_dir or DATA_PROC)
    out_dir.mkdir(parents=True, exist_ok=True)

    def scaled(cols):
        sub = df[[target] + cols].astype(float)
        mu, sd = sub.iloc[:train_n].mean(), sub.iloc[:train_n].std().replace(0, 1.0)
        return _with_split((sub - mu) / sd, train_n)

    frames = {
        "train_test_base_differenced.csv": _with_split(difference(df[[target] + base], d), train_n - d),
        "train_test_base_scaled.csv": scaled(base),
        "train_test_expanded_scaled.csv": scaled(ext),
    }
    paths = {}
    for name, fr in frames.items():
        path = out_dir / name
        fr.to_csv(path, index_label="date")
        paths[name] = path
    return {"paths": paths, "frames": frames}

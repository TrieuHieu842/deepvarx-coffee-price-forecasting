# -*- coding: utf-8 -*-
"""
DeepVARX — Trang 5: Phân tích đặc tính chuỗi thời gian (bản 2)

Đặt tại: pages/3_Phan_tich_chuoi_thoi_gian.py  (sau pages/2_Phan_tich_tuong_quan.py)
Cần: streamlit, pandas, numpy, scipy, plotly, statsmodels

CÂU HỎI CỦA TRANG
    Chuỗi giá có tính chất thời gian gì, và điều đó buộc chọn biến đổi nào
    (mức / sai phân / VECM)? Mọi kiểm định chỉ chạy trên đoạn TRAIN.

LUỒNG (mọi thứ tự chạy, không cần bấm nút tính)
    Tham số khai báo trước -> 5 tab kết quả -> tab Quyết định -> nút Lưu

THAY ĐỔI SO VỚI BẢN 1
    Nhất quán : toàn trang chạy trên CÙNG một chuỗi (log(Y) hoặc Y thô, có công tắc); chuỗi dừng hóa theo
                d từng chuỗi dùng chung cho chọn bậc trễ, BDS và phần dư (không ép Δ cho chuỗi d = 0)
    Bậc d     : chính sách thận trọng — ADF/KPSS mâu thuẫn ở MỨC thì xét tiếp sai phân (tắt được)
    Bậc trễ   : AutoReg dùng hold_back để mọi p chung một mẫu (AIC/BIC so sánh được)
    Mốc TRAIN : đọc NGÀY cắt từ Trang 4 (analysis_scopes / corr_settings); chỉ dùng tỷ lệ khi không có
    Dữ liệu   : tập tỉnh cố định / median chống 'composition effect'; đưa về lưới thời gian đều
                (B hoặc D), điền tiến tối đa k kỳ; Δ chỉ tính giữa hai kỳ liền kề
    Mùa vụ    : ngưỡng F_S hiệu chuẩn bằng giả lập (hoán vị Δ rồi cộng dồn); Kruskal–Wallis trên Δ chuẩn hóa
                theo độ lệch chuẩn trượt (giảm ảnh hưởng ARCH)
    Bước ngẫu nhiên : Holm qua các lag/q; chú thích khi LB bác bỏ mà VR (bền vững) không
    Johansen  : det_order theo 'reg', mức theo α, k_ar_diff cho phép 0, cùng IC_MAXLAG với phần lag;
                thêm đồng liên kết Y–X (X là I(1))
    Mới       : ổn định tham số (Welch, Levene, CUSUM) + ngoại lai; chẩn đoán phần dư VAR/AR
    Cache     : order_curves, Johansen, ARCH-LM, Ljung–Box/VR, chẩn đoán phần dư
    Đầu ra    : thêm diff_order_by_series, ts_profile, transform_decision_x_df, use_log_y

ĐẦU RA (st.session_state) — ghi khi bấm "Lưu kết quả"
    diff_order            : bậc sai phân gợi ý (max d của các Y; Trang 13 dùng làm mặc định)
    diff_order_by_series  : {tên chuỗi: d} cho cả Y và X
    analysis_scopes       : ngày cuối đoạn TRAIN đã dùng (Trang 13 kiểm toán rò rỉ)
    transform_decision_df : bảng quyết định biến đổi cho Y
    transform_decision_x_df : bảng quyết định cho X
    evidence_log          : thẻ minh chứng (schema giữ nguyên)
    var_lag_suggestion    : bậc trễ p tối ưu theo AIC/BIC/HQ trên chuỗi dừng hóa
    ts_profile            : tóm tắt cấu trúc (d, Johansen, mùa vụ, ARCH, RW, BDS, phần dư, mốc TRAIN)
    use_log_y             : True nếu Y được phân tích trên log
"""
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats as sps

st.set_page_config(page_title="Phân tích chuỗi thời gian", page_icon="☕", layout="wide")

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, breaks_cusumolsresid
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tools.sm_exceptions import InterpolationWarning, ValueWarning
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, bds, kpss, pacf
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False

if STATSMODELS_OK:
    warnings.filterwarnings("ignore", category=InterpolationWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIG (đồng bộ với Trang 1, 4, 13)
# ============================================================
APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá thu mua cà phê Việt Nam"
PROVINCE_COL = "province"
COFFEE_TYPE_COL = "coffee_type"
META_COLS = {"date", "province", "coffee_type", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"
SCOPE_NAME = "Trang 5 (chuỗi thời gian)"

# Tham số ứng dụng (không phải chuẩn học thuật)
ARCH_LAGS = 5            # số lag của ARCH-LM
ROLL_WIN = 30            # cửa sổ rolling std để vẽ / chuẩn hóa Δ
SEASON_FS_THR = 0.3      # ngưỡng sàn của sức mạnh mùa vụ F_S (ngưỡng thực dùng = max(sàn, phân vị giả lập))
SEASON_SIM_DEFAULT = 200  # số lần giả lập mặc định khi hiệu chuẩn ngưỡng F_S (chỉnh được ở tùy chọn nâng cao)
SEASON_Q = 0.95          # phân vị của phân phối F_S dưới H0 'không có mùa vụ'
SEASON_BOOT = 500        # số lần bootstrap để ước lượng khoảng tin cậy của phân vị
SEASON_CI_WIDTH = 0.05   # KTC 95% của ngưỡng rộng hơn mức này thì coi là phân vị chưa ổn định
ODIFF_ACF1 = -0.4        # ACF(1) của chuỗi sai phân thấp hơn mức này: nghi sai phân quá mức
IC_MAXLAG = 20           # p tối đa khi chọn bậc trễ (cả chuỗi dừng hóa lẫn mức cho Johansen)
BDS_CAP, BDS_DIM = 2000, 4
MIN_TRAIN = 60
MAX_X_JOH = 4            # số biến X (I(1)) tối đa đưa vào Johansen Y–X
OUTLIER_Z = 5.0          # ngưỡng |z| bền vững (median/MAD) đếm ngoại lai
GAP_FACTOR = 1.5         # khoảng cách ngày > GAP_FACTOR × trung vị coi là khuyết
BALANCE_COV = 0.95       # tỉnh phải có dữ liệu ≥ 95% số ngày mới được giữ khi 'cố định tập tỉnh'

ALPHA_KEY = {0.01: "1%", 0.05: "5%", 0.10: "10%"}   # khóa giá trị tới hạn KPSS
ALPHA_JOH_IDX = {0.10: 0, 0.05: 1, 0.01: 2}         # cột của cvt/cvm trong coint_johansen: 90%, 95%, 99%
LAB = {0: "Mức", 1: "Δ", 2: "Δ²"}
CODE_TEXT = {
    "STAT": "Dừng (ADF bác bỏ nghiệm đơn vị; KPSS không bác bỏ tính dừng)",
    "UR": "Nghiệm đơn vị (ADF không bác bỏ; KPSS bác bỏ tính dừng)",
    "CONFLICT": "Mâu thuẫn: cả hai cùng bác bỏ (dừng quanh xu hướng hoặc có gãy cấu trúc)",
    "NOEVID": "Thiếu bằng chứng: cả hai không bác bỏ",
}


def _streamlit_version():
    return tuple(int(re.sub(r"\D", "", p) or 0) for p in st.__version__.split(".")[:2])


STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


# ═══════════════════════════════════════════════════════════════════════════
# HÀM TÍNH TOÁN (không phụ thuộc giao diện, có thể kiểm thử riêng)
# ═══════════════════════════════════════════════════════════════════════════
def fmt_kpss_p(p):
    """statsmodels nội suy p-value KPSS từ bảng nên bị chặn trong [0.01, 0.10]."""
    if p <= 0.0101:
        return "≤ 0.01"
    if p >= 0.0999:
        return "≥ 0.10"
    return f"{p:.3f}"


def holm_min(pvals):
    """p nhỏ nhất sau hiệu chỉnh Holm (bỏ NaN); NaN nếu không có p nào."""
    p = np.asarray(list(pvals), float)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return float("nan")
    return float(multipletests(p, method="holm")[1].min())


def p4_train_end(cs, scopes):
    """Ngày cuối TRAIN do Trang 4 lưu (corr_settings hoặc analysis_scopes); None nếu không có."""
    for k in ("train_end", "train_end_date", "last_train", "train_last_date"):
        v = (cs or {}).get(k)
        if v:
            try:
                return pd.Timestamp(v)
            except Exception:
                pass
    for name, v in (scopes or {}).items():
        if "Trang 4" in str(name) or "tương quan" in str(name).lower():
            try:
                return pd.Timestamp(v)
            except Exception:
                pass
    return None


def _agg_provinces(sub, date_col, ycol, how, balanced, min_cov, fit_end=None):
    """Gộp các tỉnh thành một chuỗi theo ngày; có thể cố định tập tỉnh để tránh composition effect.
    Độ phủ dùng để chọn tập tỉnh CHỈ tính trên các ngày ≤ fit_end (TRAIN); tập đã chọn áp cho mọi ngày,
    nên việc chọn không phụ thuộc dữ liệu validation/test."""
    if PROVINCE_COL in sub.columns:
        P = sub.pivot_table(index=date_col, columns=PROVINCE_COL, values=ycol, aggfunc="mean")
        n_all = P.shape[1]
        if balanced and n_all > 1:
            cov_src = P.loc[:fit_end] if fit_end is not None else P
            cov = cov_src.notna().mean()
            keep = cov[cov >= min_cov].index
            if len(keep) > 0:
                P = P[keep]
        s = P.median(axis=1) if how == "median" else P.mean(axis=1)
        cnt = P.notna().sum(axis=1)
        info = dict(n_all=int(n_all), n_used=int(P.shape[1]),
                    cnt_min=int(cnt.min()) if len(cnt) else 0, cnt_max=int(cnt.max()) if len(cnt) else 0,
                    share_var=float((cnt != cnt.max()).mean()) if len(cnt) else 0.0)
        return s, info
    s = sub.groupby(date_col)[ycol].mean()
    return s, dict(n_all=1, n_used=1, cnt_min=1, cnt_max=1, share_var=0.0)


def build_y(d, date_col, ycol, types, how="mean", balanced=True, min_cov=BALANCE_COV, fit_end=None):
    """Y: mỗi loại cà phê một chuỗi (gộp các tỉnh theo ngày), giống Trang 13. Trả (Y, bảng chẩn đoán).
    fit_end: ngày cuối TRAIN — tập tỉnh cố định được chọn chỉ từ dữ liệu đến ngày này."""
    parts, diag = {}, []
    if COFFEE_TYPE_COL in d.columns and types:
        for t in types:
            sub = d[d[COFFEE_TYPE_COL].astype(str) == t]
            if sub.empty:
                continue
            s, info = _agg_provinces(sub, date_col, ycol, how, balanced, min_cov, fit_end)
            parts[t] = s
            info["Chuỗi"] = t
            diag.append(info)
    else:
        s, info = _agg_provinces(d, date_col, ycol, how, balanced, min_cov, fit_end)
        parts[ycol] = s
        info["Chuỗi"] = ycol
        diag.append(info)
    return pd.DataFrame(parts).sort_index(), pd.DataFrame(diag)


def _month_end_alias():
    try:
        pd.tseries.frequencies.to_offset("ME")
        return "ME"
    except Exception:
        return "M"


def infer_grid(idx):
    """Tần suất lưới của chỉ mục thời gian: 'B' (ngày làm việc), 'D' (ngày), 'W-<thứ>' (tuần), 'MS'/'ME' (tháng).
    None nếu không đều hoặc không nhận ra."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 10:
        return None
    gaps = idx.to_series().diff().dropna().dt.days
    med = float(gaps.median())
    if med <= 1:
        return "B" if (idx.dayofweek >= 5).mean() < 0.01 else "D"
    if 6 <= med <= 8:
        wd = idx.dayofweek.value_counts()
        if wd.iloc[0] / len(idx) >= 0.9:
            return "W-" + ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(wd.index[0])]
        return None
    if 27 <= med <= 32:
        if idx.is_month_start.mean() >= 0.9:
            return "MS"
        if idx.is_month_end.mean() >= 0.9:
            return _month_end_alias()
    return None


def grid_label(g):
    if g is None:
        return "không đều / không nhận ra (giữ nguyên, ngắt đoạn khi khoảng cách quá lớn)"
    g = str(g)
    if g == "B":
        return "ngày làm việc (B)"
    if g == "D":
        return "ngày (D)"
    if g.startswith("W"):
        return f"tuần ({g})"
    return f"tháng ({g})"


def season_period(grid, idx, override=0):
    """Chu kỳ mùa vụ năm theo TẦN SUẤT LƯỚI (không suy từ số dòng). Trả (chu kỳ, nguồn).
    B -> 261, D -> 365, tuần -> 52, tháng -> 12; không có lưới thì ước lượng từ số dòng/năm (kém tin cậy)."""
    if override and int(override) >= 4:
        return int(override), "người dùng chỉ định"
    if grid:
        g = str(grid)
        if g == "B":
            return 261, "lưới ngày làm việc (~261 kỳ/năm)"
        if g == "D":
            return 365, "lưới ngày (365 kỳ/năm)"
        if g.startswith("W"):
            return 52, "lưới tuần (52 kỳ/năm)"
        return 12, "lưới tháng (12 kỳ/năm)"
    return infer_period(idx), "ước lượng từ số dòng/năm (không có lưới đều: kém tin cậy)"


def diff1(s, max_gap=None):
    """Sai phân bậc 1 chỉ giữa hai kỳ LIỀN KỀ: ô khuyết cho NaN; nếu max_gap (ngày) thì cặp cách nhau xa hơn
    mức này cũng cho NaN. Không bỏ dòng nên vị trí trên lưới được giữ nguyên."""
    s = pd.Series(s).astype(float)
    d = s.diff()
    if max_gap and isinstance(s.index, pd.DatetimeIndex):
        gap = s.index.to_series().diff().dt.days.values
        d[np.nan_to_num(gap, nan=0.0) > max_gap] = np.nan
    return d


def longest_block(obj, max_gap=None):
    """Đoạn LIÊN TỤC dài nhất không có NaN (Series hoặc DataFrame: mọi cột). Hai hàng liền nhau trong đoạn phải
    kề nhau trên lưới (và, nếu max_gap, cách nhau không quá max_gap ngày). Tránh nối các đoạn qua khoảng trống
    khi chạy ADF/ACF/VAR/STL/Johansen. Khi có nhiều đoạn bằng nhau thì lấy đoạn gần nhất."""
    x = obj.dropna()
    if len(x) <= 1:
        return x
    pos = obj.index.get_indexer(x.index)
    brk = np.diff(pos) != 1
    if max_gap and isinstance(x.index, pd.DatetimeIndex):
        gaps = x.index.to_series().diff().dt.days.values[1:]
        brk = brk | (gaps > max_gap)
    seg = np.concatenate([[0], np.cumsum(brk)])
    sizes = np.bincount(seg)
    best = len(sizes) - 1 - int(np.argmax(sizes[::-1]))
    return x[seg == best]


def rank_x_by_corr(train, ys, xs_i1, max_gap=None):
    """Xếp hạng X (I(1)) theo trung bình |tương quan của ΔX với ΔY| trên TRAIN; hòa thì theo tên.
    Không phụ thuộc thứ tự biến trong danh sách. Trả (danh sách đã xếp, điểm)."""
    dY = {c: diff1(train[c], max_gap) for c in ys}
    scores = {}
    for x in xs_i1:
        dx = diff1(train[x], max_gap)
        cors = []
        for c in ys:
            both = pd.concat([dx, dY[c]], axis=1).dropna()
            if len(both) >= 30:
                v = both.iloc[:, 0].corr(both.iloc[:, 1])
                if pd.notna(v):
                    cors.append(abs(float(v)))
        scores[x] = float(np.mean(cors)) if cors else 0.0
    return sorted(xs_i1, key=lambda k: (-scores[k], str(k))), scores


def regularize(df, limit):
    """Đưa về lưới thời gian đều (nếu theo ngày) và điền tiến tối đa `limit` kỳ. Chỉ dùng giá trị quá khứ
    nên không rò rỉ. Với dữ liệu tuần/tháng chỉ báo mức độ khuyết."""
    idx = df.index
    grid = infer_grid(idx)
    info = dict(grid=grid, n_before=int(len(df)))
    if grid is None:
        gaps = idx.to_series().diff().dropna().dt.days
        med = float(gaps.median()) if len(gaps) else 0.0
        info.update(irregular=float((gaps > GAP_FACTOR * med).mean()) if med > 0 else 0.0,
                    max_gap=int(gaps.max()) if len(gaps) else 0, n_after=int(len(df)),
                    miss_before=float(df.isna().mean().mean()), miss_after=float(df.isna().mean().mean()),
                    dropped=0)
        return df, info
    full = pd.date_range(idx[0], idx[-1], freq=grid)
    out = df.reindex(full)
    out.index.name = df.index.name
    miss_before = float(out.isna().mean().mean())
    if limit > 0:
        out = out.ffill(limit=int(limit))
    info.update(n_after=int(len(out)), miss_before=miss_before, miss_after=float(out.isna().mean().mean()),
                dropped=int(len(idx.difference(full))), irregular=0.0, max_gap=0)
    return out, info


def run_adf_kpss(s, reg, alpha):
    """ADF (H0: nghiệm đơn vị) + KPSS (H0: dừng), diễn giải theo bảng 4 trường hợp."""
    x = pd.Series(s).dropna().values.astype(float)
    a = adfuller(x, regression=reg, autolag="AIC")
    k_stat, k_p, k_lags, k_crit = kpss(x, regression=reg, nlags="auto")
    adf_rej = a[1] < alpha
    kpss_rej = k_stat > k_crit[ALPHA_KEY[alpha]]      # so với giá trị tới hạn, tránh bị chặn p
    if adf_rej and not kpss_rej:
        code = "STAT"
    elif (not adf_rej) and kpss_rej:
        code = "UR"
    elif adf_rej and kpss_rej:
        code = "CONFLICT"
    else:
        code = "NOEVID"
    return dict(n=len(x), adf_stat=float(a[0]), adf_p=float(a[1]), kpss_stat=float(k_stat),
                kpss_p=float(k_p), adf_rej=bool(adf_rej), kpss_rej=bool(kpss_rej),
                code=code, text=CODE_TEXT[code])


def find_d(s, reg, alpha, conservative=True, max_gap=None):
    """
    Bậc tích hợp d: bậc nhỏ nhất (0, 1, 2) mà ADF bác bỏ nghiệm đơn vị; KPSS đóng vai xác nhận.
    Sai phân chỉ tính giữa hai kỳ liền kề (khuyết -> NaN) và mỗi phép kiểm định chạy trên ĐOẠN LIÊN TỤC dài
    nhất của chuỗi, không nối các đoạn qua khoảng trống.
    conservative = True: ADF và KPSS cùng bác bỏ ở MỨC (hay gặp khi có gãy cấu trúc, ADF mất sức mạnh)
    thì không dừng ở d = 0 mà xét tiếp sai phân; chỉ quay về d = 0 nếu sai phân cũng không đạt.
    Cấu hình tất định 'reg' chỉ áp cho mức; Δ và Δ² dùng 'c'.
    """
    raw = pd.Series(s).astype(float)
    x = raw
    rows, d_found, conf, level_conflict = [], None, "", False
    for d in range(3):
        if d > 0:
            x = diff1(x, max_gap)
        cur = longest_block(x, max_gap)
        if len(cur) < 30:
            break
        r = run_adf_kpss(cur, reg if d == 0 else "c", alpha)
        r["d"] = d
        r["acf1"] = float(acf(cur, nlags=1, fft=True)[1])
        rows.append(r)
        if r["adf_rej"]:
            if conservative and d == 0 and r["code"] == "CONFLICT":
                level_conflict = True
                continue
            d_found = d
            if level_conflict:
                conf = "Thận trọng (mức: ADF/KPSS mâu thuẫn → xét sai phân)"
            elif r["code"] == "STAT":
                conf = "Xác nhận (ADF và KPSS đồng thuận)"
            else:
                conf = "Chỉ ADF (KPSS mâu thuẫn)"
            break
    if d_found is None and level_conflict:
        d_found, conf = 0, "Chỉ ADF (KPSS mâu thuẫn)"
    return d_found, conf, rows


def arch_lm(r, nlags):
    """ARCH-LM (Engle, 1982) trên phần dư của mô hình trung bình hằng số."""
    e = np.asarray(r, float)
    e = e - e.mean()
    lm, p, _, _ = het_arch(e, nlags)
    return float(lm), float(p)


def stability_tests(r):
    """Ổn định tham số (gãy cấu trúc nhẹ) + ngoại lai trên Δ: Welch (trung bình hai nửa), Levene (phương sai
    hai nửa), CUSUM (trung bình), ngoại lai theo z bền vững."""
    r = np.asarray(r, float)
    n = len(r)
    h = n // 2
    a, b = r[:h], r[h:]
    mean_p = float(sps.ttest_ind(a, b, equal_var=False).pvalue)
    var_p = float(sps.levene(a, b, center="median").pvalue)
    try:
        cusum_p = float(breaks_cusumolsresid(r - r.mean(), ddof=1)[1])
    except Exception:
        cusum_p = float("nan")
    med = np.median(r)
    mad = np.median(np.abs(r - med)) + 1e-12
    z = 0.6745 * (r - med) / mad
    n_out = int((np.abs(z) > OUTLIER_Z).sum())
    return dict(mean_p=mean_p, var_p=var_p, cusum_p=cusum_p, n_out=n_out, share_out=n_out / n, n=n)


def infer_period(idx):
    """Số quan sát mỗi năm (chu kỳ mùa vụ năm); None nếu không phải chỉ mục thời gian."""
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return None
    span = (idx[-1] - idx[0]).days / 365.25
    return int(round(len(idx) / span)) if span > 0 else None


def seasonal_strength(s, period):
    """F_S = max(0, 1 − Var(R)/Var(S+R)) trên STL (Hyndman & Athanasopoulos, 2021)."""
    res = STL(np.asarray(s, float), period=period, robust=True).fit()
    fs = max(0.0, 1.0 - np.var(res.resid) / np.var(res.seasonal + res.resid))
    return float(fs), res


def kruskal_by(d, key):
    """Kruskal–Wallis giữa các nhóm (tháng)."""
    groups = [g.values for _, g in d.groupby(key) if len(g) >= 3]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = sps.kruskal(*groups)
    return float(h), float(p)


def order_curves(df, maxlags):
    """AIC/BIC/HQ theo p trên MẪU CHUNG. K ≥ 2: VAR.select_order. K = 1: AutoReg với hold_back = maxlags."""
    df = df.dropna().reset_index(drop=True)
    K, n = df.shape[1], len(df)
    maxlags = int(max(1, min(maxlags, (n - 1) // (2 * (K + 1)))))
    if K >= 2:
        sel = VAR(df).select_order(maxlags=maxlags)
        ics = pd.DataFrame(sel.ics)[["aic", "bic", "hqic"]]
        ics = ics.iloc[1:].copy()                    # bỏ p = 0
        ics.index = range(1, len(ics) + 1)
    else:
        rows = {}
        y = df.iloc[:, 0]
        for p in range(1, maxlags + 1):
            r = AutoReg(y, lags=p, trend="c", hold_back=maxlags).fit()   # cùng số quan sát cho mọi p
            rows[p] = dict(aic=r.aic, bic=r.bic, hqic=r.hqic)
        ics = pd.DataFrame(rows).T
    ics.index.name = "p"
    ics.columns = ["AIC", "BIC", "HQ"]
    return ics


def model_resid(df, p):
    """Phần dư VAR(p) (K ≥ 2) hoặc AR(p) (K = 1) — đầu vào cho BDS và chẩn đoán phần dư."""
    df = df.dropna().reset_index(drop=True)
    if df.shape[1] >= 2:
        r = VAR(df).fit(p)
        return pd.DataFrame(np.asarray(r.resid), columns=df.columns)
    r = AutoReg(df.iloc[:, 0], lags=p, trend="c").fit()
    return pd.DataFrame({df.columns[0]: np.asarray(r.resid)})


def johansen_sequential(data, det_order, k_ar_diff, ci=1):
    """Johansen (1991): kiểm định tuần tự H0: r ≤ i. ci = cột giá trị tới hạn (0: 10%, 1: 5%, 2: 1%)."""
    res = coint_johansen(np.asarray(data, float), det_order=det_order, k_ar_diff=k_ar_diff)
    K = data.shape[1]
    tr, me = [], []
    for i in range(K):
        tr.append({"H0": f"r ≤ {i}", "Thống kê": float(res.lr1[i]), "Tới hạn": float(res.cvt[i, ci]),
                   "Bác bỏ?": "Có" if res.lr1[i] > res.cvt[i, ci] else "Không"})
        me.append({"H0": f"r ≤ {i}", "Thống kê": float(res.lr2[i]), "Tới hạn": float(res.cvm[i, ci]),
                   "Bác bỏ?": "Có" if res.lr2[i] > res.cvm[i, ci] else "Không"})

    def seq_r(rows):
        r = 0
        for row in rows:             # dừng ở lần không bác bỏ đầu tiên
            if row["Bác bỏ?"] == "Có":
                r += 1
            else:
                break
        return r
    return pd.DataFrame(tr), pd.DataFrame(me), seq_r(tr), seq_r(me)


def variance_ratio(r, q):
    """VR(q) = Var(r_q)/(q·Var(r)) với z* bền vững với phương sai đổi (Lo & MacKinlay, 1988)."""
    r = np.asarray(r, float)
    N = len(r)
    mu = r.mean()
    d = r - mu
    s_a = np.sum(d ** 2) / (N - 1)
    m = q * (N - q + 1) * (1 - q / N)
    rq = np.convolve(r, np.ones(q), mode="valid")
    s_c = np.sum((rq - q * mu) ** 2) / m
    vr = s_c / s_a
    d2 = d ** 2
    denom = d2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(d2[j:] * d2[:-j]) / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta)
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    return float(vr), float(z), float(p)


def stationary_frame(train, ys, d_eff, max_gap=None):
    """Khung Y dừng hóa theo bậc d_eff từng chuỗi (0: mức, 1: Δ, 2: Δ²). d_eff phải được truyền tường minh
    (chuỗi chưa xác định d đã được gán d tạm và ghi nhận ở nơi gọi). Sai phân chỉ giữa kỳ liền kề; kết quả là
    ĐOẠN LIÊN TỤC dài nhất để VAR/BDS không nối qua khoảng trống."""
    cols = {}
    for c in ys:
        x = train[c]
        for _ in range(int(d_eff[c])):
            x = diff1(x, max_gap)
        cols[c] = x
    return longest_block(pd.DataFrame(cols), max_gap)


# ---- Các bước nặng được cache theo (dữ liệu, tham số) ----
@st.cache_data(show_spinner=False)
def cached_find_d(s, reg, alpha, conservative, max_gap):
    return find_d(s, reg, alpha, conservative, max_gap)


@st.cache_data(show_spinner=False)
def cached_seasonality(s, d1, period, alpha, thr, nsim):
    if len(s) < 2 * period:
        return dict(err=f"Cần ≥ {2 * period} quan sát (đang có {len(s)}).")
    try:
        fs, res = seasonal_strength(s, int(period))
        # Hiệu chuẩn ngưỡng: hoán vị Δ (phá mùa vụ, giữ phân phối) rồi cộng dồn -> phân phối F_S dưới H0
        rng = np.random.default_rng(12345)
        dv = d1.values
        x0 = float(s.iloc[0])
        null = []
        for _ in range(int(nsim)):
            z = x0 + np.concatenate([[0.0], np.cumsum(rng.permutation(dv))])
            null.append(seasonal_strength(z, int(period))[0])
        null = np.asarray(null, float)
        thr_sim = float(np.quantile(null, SEASON_Q)) if len(null) else 0.0
        # Độ ổn định của phân vị: KTC bootstrap 95% và so sánh hai nửa số lần mô phỏng
        rng2 = np.random.default_rng(777)
        boot = [np.quantile(rng2.choice(null, size=len(null), replace=True), SEASON_Q)
                for _ in range(SEASON_BOOT)] if len(null) >= 10 else [thr_sim]
        thr_lo, thr_hi = (float(v) for v in np.quantile(boot, [0.025, 0.975]))
        half = len(null) // 2
        half_diff = (abs(float(np.quantile(null[:half], SEASON_Q)) - float(np.quantile(null[half:], SEASON_Q)))
                     if half >= 5 else float("nan"))
    except Exception as e:
        return dict(err=str(e))
    thr_eff = max(float(thr), thr_sim)
    lo_eff, hi_eff = max(float(thr), thr_lo), max(float(thr), thr_hi)
    ci_stable = bool((thr_hi - thr_lo) <= SEASON_CI_WIDTH)
    borderline = bool(lo_eff <= fs < hi_eff)       # kết luận 'mạnh/yếu' đổi theo cận của KTC
    # Kruskal–Wallis trên Δ chuẩn hóa theo độ lệch chuẩn trượt để giảm ảnh hưởng phương sai đổi
    sd = d1.rolling(2 * ROLL_WIN, min_periods=20, center=True).std()
    zs = (d1 / sd.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    _, p_m = kruskal_by(zs, zs.index.month)
    strong = fs >= thr_eff
    sig_m = (p_m < alpha) if not np.isnan(p_m) else False
    if strong and sig_m:
        concl = "Có mùa vụ"
    elif strong or sig_m:
        concl = "Không đồng thuận — chưa kết luận"
    else:
        concl = "Chưa có bằng chứng mùa vụ"
    if borderline:
        concl += " — sát ngưỡng, kết luận nhạy với phân vị mô phỏng"
    return dict(fs=fs, p_m=p_m, concl=concl, period=int(period), thr_sim=thr_sim, thr_eff=thr_eff,
                thr_lo=thr_lo, thr_hi=thr_hi, half_diff=half_diff, ci_stable=ci_stable, borderline=borderline,
                nsim=int(nsim), seasonal=np.asarray(res.seasonal))


@st.cache_data(show_spinner=False)
def cached_order_curves(df, maxlags):
    return order_curves(df, maxlags)


@st.cache_data(show_spinner=False)
def cached_joh_system(df, reg, crit, alpha):
    """Johansen cho một hệ chuỗi mức: chọn k_ar_diff = p − 1 (p theo `crit` trên mức), det_order theo `reg`."""
    ic = order_curves(df, IC_MAXLAG)
    kd = max(int(ic[crit].idxmin()) - 1, 0)
    det = 0 if reg == "c" else 1
    tr, me, r_tr, r_me = johansen_sequential(df, det, kd, ALPHA_JOH_IDX[alpha])
    return dict(tr=tr, me=me, r_tr=int(r_tr), r_me=int(r_me), K=int(df.shape[1]), kd=int(kd),
                n=int(len(df)), det=det)


@st.cache_data(show_spinner=False)
def cached_arch(r, nlags):
    return arch_lm(r, nlags)


@st.cache_data(show_spinner=False)
def cached_rw(r, alpha):
    """Ljung–Box + Variance Ratio; mỗi nhóm test hiệu chỉnh Holm qua các lag / q."""
    lags = [l for l in (5, 10, 20) if l < len(r) // 2]
    if not lags:
        return None
    lb = acorr_ljungbox(r, lags=lags, return_df=True)
    vr = {q: variance_ratio(r.values, q) for q in (2, 5, 10)}
    lbp = {int(l): float(p) for l, p in lb["lb_pvalue"].items()}
    lb_rej = holm_min(lbp.values()) < alpha
    vr_rej = holm_min([v[2] for v in vr.values()]) < alpha
    return dict(lb=lbp, vr=vr, lb_rej=bool(lb_rej), vr_rej=bool(vr_rej))


@st.cache_data(show_spinner=False)
def cached_bds(df_t, p, dmax, cap):
    resid = model_resid(df_t, int(p))
    out = {}
    for c in df_t.columns:
        e = resid[c].values[-cap:]
        stat, pv = bds(e, max_dim=dmax)
        stat, pv = np.atleast_1d(stat), np.atleast_1d(pv)
        out[c] = dict(stat=stat, p=pv, ph=multipletests(pv, method="holm")[1], n=len(e))
    return out


@st.cache_data(show_spinner=False)
def cached_resid_diag(df_t, p):
    """Phần dư VAR(p)/AR(p): tính trắng (Portmanteau / Ljung–Box) và tính chuẩn (VAR: Jarque–Bera đa biến)."""
    d = df_t.dropna().reset_index(drop=True)
    resid = model_resid(d, int(p))
    if d.shape[1] >= 2:
        r = VAR(d).fit(int(p))
        nl = int(min(p + 10, max(p + 2, len(resid) // 4)))
        w = r.test_whiteness(nlags=nl, adjusted=True)
        nm = r.test_normality()
        return dict(white_p=float(w.pvalue), norm_p=float(nm.pvalue), nlags=nl, n=len(resid), K=d.shape[1])
    e = resid.iloc[:, 0].values
    nl = int(min(10, len(e) // 4))
    lb = acorr_ljungbox(e, lags=[nl], return_df=True)
    return dict(white_p=float(lb["lb_pvalue"].iloc[0]), norm_p=float(sps.jarque_bera(e).pvalue), nlags=nl,
                n=len(e), K=1)


# ═══════════════════════════════════════════════════════════════════════════
# GIAO DIỆN
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    _sel = st.session_state.get("corr_selected_vars") or []
    if _sel:
        st.info(f"Biến từ Trang 4: {len(_sel)}")
    if st.button("← Về Trang 4 (tương quan)", key="sb_back", **STRETCH):
        st.switch_page("pages/2_Phan_tich_tuong_quan.py")

st.caption(APP_HEADER)
st.title("Trang 5 — Phân tích đặc tính chuỗi thời gian")
st.markdown(
    "**Câu hỏi của trang:** chuỗi giá có tính chất thời gian gì, và điều đó buộc ta chọn biến đổi nào "
    "(mức, sai phân hay VECM)? **Mọi kiểm định chỉ chạy trên đoạn TRAIN**; kết quả tự cập nhật khi đổi tham số."
)

if not STATSMODELS_OK:
    st.error("Thiếu thư viện statsmodels. Cài bằng: `pip install statsmodels` rồi chạy lại.")
    st.stop()

# ── Dữ liệu ───────────────────────────────────────────────────────────────
_used_processed = st.session_state.get("processed_data") is not None
df = st.session_state.get("processed_data")
if df is None:
    df = st.session_state.get("raw_data")
if df is None:
    st.warning("Chưa có dữ liệu. Quay lại Trang 1 để nạp dữ liệu.")
    if st.button("← Về Trang 1", key="no_data_back"):
        st.switch_page("pages/1_Tai_du_lieu.py")
    st.stop()

date_col = st.session_state.get("date_col", "date")
if date_col not in df.columns:
    st.error(f"Không tìm thấy cột ngày `{date_col}`. Quay lại Trang 1.")
    st.stop()
d_all = df.copy()
d_all[date_col] = pd.to_datetime(d_all[date_col], errors="coerce")
d_all = d_all.dropna(subset=[date_col])

cols_num = [c for c in d_all.select_dtypes(include="number").columns if c not in META_COLS]
cs = st.session_state.get("corr_settings") or {}
if not cols_num:
    st.error("Không có cột số để phân tích.")
    st.stop()


def _is_price(name):
    toks = set(re.sub(r"[^0-9a-zà-ỹ]+", " ", str(name).lower()).split())
    return bool(toks & {"price", "gia", "giá", "target"}) or any(t.startswith("price") for t in toks)


if cs.get("y") in cols_num:
    y_def = cs["y"]
elif TARGET_COL in cols_num:
    y_def = TARGET_COL
else:
    y_def = next((c for c in cols_num if _is_price(c)), cols_num[0])

types_all = (sorted(d_all[COFFEE_TYPE_COL].dropna().astype(str).unique())
             if COFFEE_TYPE_COL in d_all.columns else [])
p4_end = p4_train_end(cs, st.session_state.get("analysis_scopes"))
sel_x = [c for c in (st.session_state.get("corr_selected_vars") or st.session_state.get("exogenous_vars") or [])
         if c in cols_num and c != y_def]

# ── Tham số khai báo trước ────────────────────────────────────────────────
with st.container(border=True):
    st.markdown("**Tham số khai báo trước**")
    c1, c2, c3 = st.columns([2, 3, 2])
    ycol = c1.selectbox("Biến giá Y", cols_num, index=cols_num.index(y_def))
    if types_all:
        ys_types = c2.multiselect("Chuỗi Y (mỗi loại cà phê = gộp các tỉnh)", types_all, default=types_all)
    else:
        ys_types = []
        c2.caption("Dữ liệu không có cột loại cà phê: dùng 1 chuỗi Y (mean theo ngày).")
    alpha = c3.select_slider("Mức ý nghĩa α", [0.01, 0.05, 0.10], value=0.05)
    y_pos_ok = bool((d_all[ycol].dropna() > 0).all())

    c4, c5, c6 = st.columns([2, 3, 2])
    reg = c4.radio("Thành phần tất định (chuỗi mức)", ["c", "ct"], horizontal=True,
                   format_func=lambda v: "hằng số" if v == "c" else "hằng số + xu hướng",
                   help="Chuỗi sai phân luôn dùng hằng số. Johansen dùng det_order tương ứng (0 hoặc 1).")
    crit = c5.radio("Tiêu chí chọn bậc trễ p", ["AIC", "BIC", "HQ"], index=1, horizontal=True,
                    help="BIC thiên về p nhỏ, AIC thiên về p lớn.")
    use_log = c6.checkbox("Phân tích trên log(Y)", value=y_pos_ok, disabled=not y_pos_ok,
                          help="Toàn trang (d, bậc trễ, Johansen, BDS, ARCH…) chạy trên cùng một chuỗi. "
                               "Nên khớp với biến đổi Trang 13 sẽ dùng. Chỉ bật được khi Y > 0.")
    use_log = bool(use_log and y_pos_ok)

    c7, c8 = st.columns([3, 2])
    if p4_end is not None:
        cut_mode = c7.radio("Mốc TRAIN", ["p4", "ratio"], horizontal=True,
                            format_func=lambda v: f"Theo Trang 4 ({p4_end.date()})" if v == "p4" else "Theo tỷ lệ")
    else:
        cut_mode = "ratio"
        c7.caption("Không tìm thấy ngày cắt TRAIN từ Trang 4: dùng tỷ lệ (lưu ý đối chiếu với Trang 4).")
    train_frac = c8.slider("Tỷ lệ TRAIN", 0.5, 0.9, float(min(max(cs.get("train_frac", 0.7), 0.5), 0.9)), 0.05,
                           disabled=(cut_mode == "p4"),
                           help="Chỉ dùng khi chọn mốc theo tỷ lệ; mặc định lấy từ Trang 4.")

    with st.expander("Tùy chọn dữ liệu nâng cao"):
        e1, e2, e3, e4 = st.columns(4)
        agg_how = e1.radio("Gộp các tỉnh", ["mean", "median"], horizontal=True,
                           format_func=lambda v: "trung bình" if v == "mean" else "trung vị",
                           help="Trung vị bền vững hơn khi một vài tỉnh có giá bất thường.")
        balanced = e2.checkbox("Cố định tập tỉnh", value=True,
                               help=f"Chỉ giữ tỉnh có dữ liệu ≥ {BALANCE_COV:.0%} số ngày để số tỉnh không đổi theo "
                                    "ngày (tránh bước nhảy giả do thành phần thay đổi).")
        ffill_limit = e3.slider("Điền tiến tối đa (kỳ)", 0, 5, 2,
                                help="Chỉ áp dụng cho dữ liệu theo ngày. Chỉ dùng giá trị quá khứ nên không rò rỉ; "
                                     "0 = không điền (Δ vẫn không nối qua khoảng trống).")
        conservative = e4.checkbox("Thận trọng khi ADF/KPSS mâu thuẫn ở mức", value=True,
                                   help="Mâu thuẫn ở mức thường do gãy cấu trúc làm ADF mất sức mạnh: xét tiếp sai "
                                        "phân thay vì kết luận 'dừng'.")
        f1, f2, f3, f4 = st.columns(4)
        d_fb = f1.radio("Khi d chưa xác định, tạm dùng", [1, 0], horizontal=True,
                        format_func=lambda v: "d = 1 (Δ)" if v == 1 else "d = 0 (mức)",
                        help="Chỉ là giá trị tạm để các bước sau (bậc trễ, BDS, phần dư) vẫn chạy. Mọi chuỗi bị áp "
                             "d tạm được ghi nhận bằng thẻ minh chứng, cảnh báo và danh sách d_unresolved khi lưu.")
        period_override = f2.number_input("Chu kỳ mùa vụ (kỳ/năm, 0 = theo lưới)", min_value=0, max_value=400,
                                          value=0, step=1,
                                          help="Mặc định suy từ tần suất lưới: ngày làm việc 261, ngày 365, tuần 52, "
                                               "tháng 12. Chỉ nhập khi lịch dữ liệu khác chuẩn.")
        season_sim = f3.slider("Số lần mô phỏng ngưỡng mùa vụ", 50, 500, SEASON_SIM_DEFAULT, 50,
                               help="Càng nhiều càng ổn định phân vị 95% nhưng càng chậm (kết quả được cache). "
                                    "Tab Mùa vụ báo khoảng tin cậy của ngưỡng.")
        x_joh_sel = f4.multiselect("X đưa vào Johansen Y–X (tối đa %d)" % MAX_X_JOH, sel_x, default=[],
                                   max_selections=MAX_X_JOH,
                                   help="Để trống = tự chọn các X có d = 1 xếp theo trung bình |tương quan của ΔX với "
                                        "ΔY| trên TRAIN (không phụ thuộc thứ tự danh sách). X được chọn mà d ≠ 1 "
                                        "sẽ bị bỏ và có ghi chú.")

if types_all and not ys_types:
    st.warning("Chọn ít nhất 1 chuỗi Y.")
    st.stop()

# ── Dựng chuỗi ────────────────────────────────────────────────────────────
sel_x = [c for c in sel_x if c != ycol]

# Mốc cắt TRAIN phải biết TRƯỚC khi dựng Y, vì việc cố định tập tỉnh chỉ được dựa trên TRAIN
all_dates = pd.DatetimeIndex(np.sort(d_all[date_col].unique()))
if cut_mode == "p4" and p4_end is not None:
    cut_date = p4_end
else:
    k_cut = max(int(len(all_dates) * train_frac), 1)
    cut_date = all_dates[min(k_cut, len(all_dates)) - 1]

Y_raw, cov_diag = build_y(d_all, date_col, ycol, ys_types, agg_how, balanced, fit_end=cut_date)
if Y_raw.shape[1] == 0:
    st.error("Không dựng được chuỗi Y nào.")
    st.stop()
if use_log:
    Y_raw = np.log(Y_raw)
ys = list(Y_raw.columns)
X = d_all.groupby(date_col)[sel_x].mean().sort_index() if sel_x else pd.DataFrame(index=Y_raw.index)
agg_raw = pd.concat([Y_raw, X], axis=1).sort_index()
agg, reg_info = regularize(agg_raw, ffill_limit)
xs = list(X.columns)
use = ys + xs

# Khoảng cách tối đa (ngày) giữa hai quan sát liền kề khi KHÔNG có lưới đều; có lưới thì khoảng trống là ô NaN
MAXGAP = None
if reg_info.get("grid") is None:
    _g = agg_raw.index.to_series().diff().dt.days.dropna()
    _med = float(_g.median()) if len(_g) else 0.0
    MAXGAP = GAP_FACTOR * _med if _med > 0 else None

train = agg.loc[:cut_date]
if len(train) < MIN_TRAIN:
    st.error(f"Đoạn TRAIN chỉ có {len(train)} quan sát (< {MIN_TRAIN}): quá ít để kiểm định.")
    st.stop()
last_train = train.index[-1]
SCOPE = f"TRAIN n={len(train)} (đến {last_train.date()})"
_dh = int(pd.util.hash_pandas_object(train, index=True).sum() % (2 ** 62))
SIG = hashlib.md5(f"{ycol}|{ys}|{xs}|{last_train}|{_dh}|{alpha}|{reg}|{crit}|{use_log}|{conservative}|"
                  f"{ffill_limit}|{agg_how}|{balanced}|{d_fb}|{period_override}|{season_sim}|{x_joh_sel}|"
                  f"{MAXGAP}".encode()).hexdigest()[:10]
unit = {"Daily": "ngày", "Weekly": "tuần", "Monthly": "tháng"}.get(st.session_state.get("data_frequency"), "kỳ")
TX = "log(Y)" if use_log else "Y"
RET_LABEL = "Δlog" if use_log else "Δ"

st.caption(f"TRAIN: **{len(train):,}** / {len(agg):,} {unit}, đến **{last_train.date()}** · "
           f"{len(ys)} chuỗi Y trên **{TX}**" + (f" · {len(xs)} biến X (từ Trang 4, giữ nguyên thang đo)" if xs
                                                  else " · chưa có biến X từ Trang 4") +
           ". Sai phân chỉ tính giữa hai kỳ liền kề.")

# Cảnh báo mốc TRAIN
if p4_end is not None and cut_mode == "ratio" and last_train.normalize() != p4_end.normalize():
    st.warning(f"Mốc TRAIN ở đây ({last_train.date()}) khác Trang 4 ({p4_end.date()}). Hai trang nên dùng cùng một "
               "mốc để Trang 13 không báo rò rỉ.")
if p4_end is not None and cut_mode == "p4" and p4_end > agg.index[-1]:
    st.warning(f"Mốc Trang 4 ({p4_end.date()}) vượt quá dữ liệu (đến {agg.index[-1].date()}): đang dùng toàn bộ dữ liệu.")

# Chẩn đoán dữ liệu đầu vào
_warn_data = []
if len(cov_diag) and (cov_diag["share_var"] > 0.05).any() and not balanced:
    _warn_data.append("Số tỉnh có dữ liệu thay đổi theo ngày: trung bình/trung vị có thể nhảy giả "
                      "(composition effect). Bật “Cố định tập tỉnh”.")
if reg_info.get("grid") is None and reg_info.get("irregular", 0) > 0.05:
    _warn_data.append(f"Khoảng cách giữa các quan sát không đều ({reg_info['irregular']:.1%} khoảng cách lớn hơn "
                      f"{GAP_FACTOR}× trung vị, lớn nhất {reg_info['max_gap']} ngày): ADF/ACF/VR có thể lệch.")
if reg_info.get("grid") is not None and reg_info["miss_after"] > 0.02:
    _warn_data.append(f"Sau khi điền tiến còn {reg_info['miss_after']:.1%} ô khuyết trên lưới "
                      f"{grid_label(reg_info['grid'])}: các kiểm định chạy trên đoạn liên tục dài nhất của từng chuỗi.")
if reg_info.get("grid") is not None and reg_info["n_before"] > 0 and reg_info["dropped"] / reg_info["n_before"] > 0.02:
    _warn_data.append(f"{reg_info['dropped']} quan sát nằm ngoài lưới {grid_label(reg_info['grid'])} bị bỏ "
                      f"({reg_info['dropped'] / reg_info['n_before']:.1%}).")
_yv = d_all.loc[d_all[date_col] <= cut_date, ycol].dropna()        # chỉ nhìn TRAIN
if len(_yv) > 10 and abs(_yv.mean()) < 0.05 and abs(_yv.std() - 1) < 0.05:
    _warn_data.append(f"`{ycol}` có vẻ đã được chuẩn hóa (trung bình ≈ 0, độ lệch chuẩn ≈ 1). Nếu chuẩn hóa dùng "
                      "toàn mẫu thì chỉ cắt TRAIN ở đây chưa đủ chống rò rỉ.")
for _w in _warn_data:
    st.warning(_w)
with st.expander("Chẩn đoán dữ liệu đầu vào"):
    if _used_processed:
        st.caption("Dữ liệu lấy từ Trang 1 sau xử lý: nếu điền khuyết/chuẩn hóa dùng thống kê toàn mẫu thì TRAIN đã "
                   "nhiễm thông tin tương lai ngay từ bước đó.")
    if len(cov_diag):
        st.dataframe(cov_diag.rename(columns={"n_all": "Số tỉnh", "n_used": "Số tỉnh dùng",
                                              "cnt_min": "Ít nhất/ngày", "cnt_max": "Nhiều nhất/ngày",
                                              "share_var": "% ngày đổi số tỉnh"})
                     [["Chuỗi", "Số tỉnh", "Số tỉnh dùng", "Ít nhất/ngày", "Nhiều nhất/ngày", "% ngày đổi số tỉnh"]]
                     .style.format({"% ngày đổi số tỉnh": "{:.1%}"}), hide_index=True, **STRETCH)
    st.caption(f"Lưới thời gian: {grid_label(reg_info.get('grid'))}"
               f" · {reg_info['n_before']:,} → {reg_info['n_after']:,} kỳ · ô khuyết {reg_info['miss_before']:.1%} → "
               f"{reg_info['miss_after']:.1%} · quan sát ngoài lưới bị bỏ: {reg_info['dropped']}"
               + (f" · ngắt đoạn khi hai quan sát cách nhau > {MAXGAP:.0f} ngày" if MAXGAP else "") + ".")
    st.caption(f"Tập tỉnh cố định chọn **chỉ từ dữ liệu đến {pd.Timestamp(cut_date).date()}** (TRAIN); tập đã chọn "
               "áp cho mọi ngày, nên không phụ thuộc validation/test." if balanced else
               "Không cố định tập tỉnh: số tỉnh có thể đổi theo ngày.")


def S(c):
    """Chuỗi mức của TRAIN: ĐOẠN LIÊN TỤC dài nhất (không nối qua khoảng trống)."""
    return longest_block(train[c], MAXGAP)


def Sd(c, k=1):
    """Sai phân bậc k của TRAIN, chỉ giữa các kỳ liền kề, lấy đoạn liên tục dài nhất."""
    x = train[c]
    for _ in range(k):
        x = diff1(x, MAXGAP)
    return longest_block(x, MAXGAP)


EV = {}


def log_ev(eid, kl, so, pp, nguon, gh):
    EV[eid] = {"ID": eid, "Trang": "5", "Kết luận": kl, "Con số": so, "Phương pháp": pp,
               "Đoạn dữ liệu": SCOPE, "Nguồn": nguon, "Giới hạn": gh, "sig": SIG}


# ═══════════════════════════════════════════════════════════════════════════
# TÍNH TOÁN (tuần tự, dùng chung cho các tab)
# ═══════════════════════════════════════════════════════════════════════════
with st.spinner("Đang tính các kiểm định trên đoạn TRAIN…"):
    # --- Ổn định tham số + ngoại lai (trên Δ) ---
    stab_res = {}
    for c in ys:
        r = Sd(c, 1)
        if len(r) < 40:
            continue
        sr = stability_tests(r.values)
        stab_res[c] = sr
        unstable = bool(sr["cusum_p"] < alpha or sr["var_p"] < alpha or sr["mean_p"] < alpha)
        sr["unstable"] = unstable
        log_ev(f"T5-STAB-{c}", f"{'Tham số kém ổn định' if unstable else 'Chưa thấy bất ổn tham số'} (α={alpha})",
               f"Welch p={sr['mean_p']:.3g}; Levene p={sr['var_p']:.3g}; CUSUM p={sr['cusum_p']:.3g}; "
               f"ngoại lai |z|>{OUTLIER_Z:g}: {sr['n_out']} ({sr['share_out']:.1%})",
               f"Welch + Levene (hai nửa mẫu) + CUSUM trên {RET_LABEL}; ngoại lai theo z bền vững (MAD)",
               "Brown & Forsythe (1974); Brown, Durbin & Evans (1975)",
               "Chỉ báo hiệu có gãy/ngoại lai, không định vị điểm gãy; nhiều test chạy cùng lúc, chưa hiệu chỉnh.")

    # --- ARCH-LM ---
    arch_res = {}
    for c in ys:
        r = Sd(c, 1)
        if len(r) < 4 * ARCH_LAGS:
            continue
        lm, p = cached_arch(r, ARCH_LAGS)
        arch_res[c] = dict(kind=RET_LABEL, lm=lm, p=p, n=len(r), rej=p < alpha)
        log_ev(f"T5-ARCH-{c}", f"{'Có hiệu ứng ARCH' if p < alpha else 'Chưa có bằng chứng ARCH'} (α={alpha})",
               f"LM={lm:.2f}, p={p:.4g}, m={ARCH_LAGS}, n={len(r)}",
               f"ARCH-LM trên phần dư mô hình trung bình hằng số của {RET_LABEL}", "Engle (1982)",
               "Chỉ chứng minh phương sai có điều kiện đổi theo thời gian; không suy ra 'phải dùng deep learning'.")

    # --- Mùa vụ ---
    per, per_src = season_period(reg_info.get("grid"), train.index, period_override)
    season_res = {}
    if per is not None and per >= 4:
        for c in ys:
            season_res[c] = cached_seasonality(S(c), Sd(c, 1), per, alpha, SEASON_FS_THR, season_sim)
            r_ = season_res[c]
            if "err" not in r_:
                log_ev(f"T5-SEAS-{c}", f"{r_['concl']} (α={alpha}, ngưỡng F_S={r_['thr_eff']:.3f})",
                       f"F_S={r_['fs']:.3f}; ngưỡng giả lập (phân vị {SEASON_Q:.0%}, {r_['nsim']} lần)="
                       f"{r_['thr_sim']:.3f}, KTC bootstrap 95% [{r_['thr_lo']:.3f}; {r_['thr_hi']:.3f}], "
                       f"chênh hai nửa={r_['half_diff']:.3f}, "
                       f"{'ổn định' if r_['ci_stable'] else 'CHƯA ỔN ĐỊNH (tăng số lần mô phỏng)'}; "
                       f"Kruskal–Wallis theo tháng (Δ chuẩn hóa) p={r_['p_m']:.4g}; chu kỳ={r_['period']} ({per_src})",
                       "STL (robust) + ngưỡng hiệu chuẩn bằng hoán vị Δ + Kruskal–Wallis",
                       "Cleveland et al. (1990); Hyndman & Athanasopoulos (2021)",
                       "Ngưỡng sàn F_S là tham số ứng dụng; dữ liệu chỉ vài năm thì có ít chu kỳ năm; "
                       "chu kỳ lấy theo tần suất lưới nên lệch nhẹ nếu lịch dữ liệu khác chuẩn.")

    # --- Tính dừng (Y và X) ---
    stat_rows, d_map, conf_map, acf_flag, block_info = [], {}, {}, {}, {}
    for c in use:
        role = "Y" if c in ys else "X"
        s_c = train[c]
        _blk = S(c)
        block_info[c] = dict(role=role, n_obs=int(s_c.notna().sum()), n_block=int(len(_blk)),
                             start=_blk.index[0] if len(_blk) else None, end=_blk.index[-1] if len(_blk) else None)
        if len(_blk) < MIN_TRAIN:
            stat_rows.append({"Chuỗi": c, "Vai trò": role, "Biến đổi": "—",
                              "Diễn giải": f"Đoạn liên tục dài nhất chỉ {len(_blk)} quan sát (< {MIN_TRAIN})"})
            d_map[c], conf_map[c] = None, "không đủ mẫu liên tục"
            continue
        try:
            d_found, conf, res = cached_find_d(s_c, reg, alpha, conservative, MAXGAP)
        except Exception as e:
            stat_rows.append({"Chuỗi": c, "Vai trò": role, "Biến đổi": "—", "Diễn giải": f"Lỗi: {e}"})
            d_map[c], conf_map[c] = None, "lỗi"
            continue
        d_map[c], conf_map[c] = d_found, (conf if d_found is not None else "chưa xác định")
        for r in res:
            stat_rows.append({"Chuỗi": c, "Vai trò": role, "Biến đổi": LAB[r["d"]], "n": r["n"],
                              "ADF stat": r["adf_stat"], "ADF p": r["adf_p"],
                              "KPSS stat": r["kpss_stat"], "KPSS p": fmt_kpss_p(r["kpss_p"]),
                              "ACF(1)": r["acf1"], "Diễn giải": r["text"]})
        last = res[-1] if res else None
        acf_flag[c] = bool(d_found and d_found >= 1 and last and last["d"] == d_found and last["acf1"] < ODIFF_ACF1)
        dtxt = "chưa xác định (ADF không bác bỏ đến Δ²)" if d_found is None else f"d = {d_found}"
        tx_c = TX if c in ys else "X (thang gốc)"
        log_ev(f"T5-STAT-{c}", f"Bậc tích hợp: {dtxt} — {conf_map[c]}",
               "; ".join(f"{LAB[r['d']]}: ADF τ={r['adf_stat']:.2f} (p={r['adf_p']:.4g}), "
                         f"KPSS η={r['kpss_stat']:.3f} → {r['code']}" for r in res),
               f"ADF (AIC, '{reg}' ở mức) làm gốc + KPSS xác nhận; α={alpha}; chuỗi {tx_c}; "
               f"{'thận trọng khi mâu thuẫn ở mức' if conservative else 'theo ADF thuần'}",
               "Dickey & Fuller (1979); Kwiatkowski et al. (1992)",
               "ADF giảm sức mạnh khi có gãy cấu trúc (Perron, 1989); kết luận chỉ áp dụng trên TRAIN.")

    # --- d chưa xác định: KHÔNG ép ngầm; gán d tạm do người dùng chọn và ghi nhận từng chuỗi ---
    d_eff, d_forced = {}, []
    for c in use:
        if d_map.get(c) is None:
            d_eff[c] = int(d_fb)
            d_forced.append(c)
            log_ev(f"T5-DFB-{c}", f"d chưa xác định ({conf_map.get(c, '')}): TẠM dùng d = {d_fb} cho các bước sau",
                   f"d_eff={d_fb}; lý do: {conf_map.get(c, '')}",
                   "Gán d tạm theo lựa chọn của người dùng (tùy chọn nâng cao); không phải kết luận kiểm định",
                   "—", "Kết quả bậc trễ, BDS, phần dư và gợi ý diff_order cho chuỗi này phụ thuộc vào d tạm; "
                        "cần xem lại dữ liệu hoặc chốt d bằng tay ở Trang 13.")
        else:
            d_eff[c] = int(d_map[c])

    # --- Chuỗi Y dừng hóa theo d (dùng cho bậc trễ, BDS, phần dư); lấy đoạn liên tục dài nhất ---
    Yt = stationary_frame(train, ys, d_eff, MAXGAP)
    t_lab = {c: LAB[d_eff[c]] + (" (d tạm)" if c in d_forced else "") for c in ys}
    T_NOTE = ", ".join(f"{c}: {t_lab[c]}" for c in ys)

    # --- Bậc trễ (trên chuỗi dừng hóa) ---
    ic_d, ic_err, best_p = None, None, {}
    try:
        ic_d = cached_order_curves(Yt, IC_MAXLAG)
        best_p = {k: int(ic_d[k].idxmin()) for k in ic_d.columns}
        log_ev("T5-LAG-Δ", f"Chọn p = {best_p[crit]} theo {crit} (chuỗi dừng hóa theo d)",
               f"p tối ưu: {best_p}; K={len(ys)}; n={len(Yt)}; biến đổi: {T_NOTE}",
               "VAR.select_order (K ≥ 2) / AutoReg có hold_back (K = 1), mẫu chung cho mọi p", "Lütkepohl (2005)",
               "Chỉ dựa trên các Y (chưa có X nên chỉ là gợi ý ban đầu cho VARX); BIC thiên về p nhỏ, AIC về p lớn.")
    except Exception as e:
        ic_err = str(e)
    p_use = best_p.get(crit)

    # --- Johansen (chỉ khi K ≥ 2 và mọi Y đều d = 1) ---
    joh, joh_msg = None, None
    if len(ys) < 2:
        joh_msg = "Johansen cần ít nhất 2 chuỗi Y (ví dụ Robusta và Arabica)."
    else:
        not_i1 = [c for c in ys if d_map.get(c) != 1]
        if not_i1:
            joh_msg = ("Điều kiện chưa thỏa — các chuỗi không có d = 1: "
                       + ", ".join(f"{c} (d={d_map.get(c)})" for c in not_i1)
                       + ". Chuỗi dừng ở mức thì không có gì để đồng liên kết.")
        else:
            try:
                joh = cached_joh_system(longest_block(train[ys], MAXGAP), reg, crit, alpha)
                log_ev("T5-JOH-system", f"Số quan hệ đồng liên kết r = {joh['r_tr']} (trace), {joh['r_me']} (max-eig); "
                                        f"K={len(ys)}",
                       "; ".join(f"{row['H0']}: {row['Thống kê']:.2f} vs {row['Tới hạn']:.2f}"
                                 for row in joh["tr"].to_dict("records")),
                       f"Johansen tuần tự, det_order={joh['det']}, k_ar_diff={joh['kd']}, mức {ALPHA_KEY[alpha]}",
                       "Johansen (1991)", f"Chỉ xét các Y ({TX}); nhạy với lag và thành phần tất định.")
            except Exception as e:
                joh_msg = f"Không chạy được Johansen: {e}"

    # --- Johansen Y–X (X là I(1)) ---
    # Chọn X: người dùng chỉ định (nếu có), ngược lại xếp hạng theo |tương quan của ΔX với ΔY| trên TRAIN
    # (không phụ thuộc thứ tự biến trong danh sách). Chỉ X có d = 1 được xét.
    x_i1_all = [c for c in xs if d_map.get(c) == 1]
    x_scores, x_skipped = {}, []
    if x_joh_sel:
        x_i1 = [c for c in x_joh_sel if c in x_i1_all][:MAX_X_JOH]
        x_skipped = [c for c in x_joh_sel if c not in x_i1_all]
        x_rule = "người dùng chọn"
    else:
        _rk, x_scores = rank_x_by_corr(train, ys, x_i1_all, MAXGAP) if x_i1_all else ([], {})
        x_i1 = _rk[:MAX_X_JOH]
        x_rule = f"tự động: top {MAX_X_JOH} theo trung bình |tương quan của ΔX với ΔY| trên TRAIN"
    joh_yx, joh_yx_err = {}, {}
    if x_i1:
        for c in ys:
            if d_map.get(c) != 1:
                continue
            try:
                jr = cached_joh_system(longest_block(train[[c] + x_i1], MAXGAP), reg, crit, alpha)
                joh_yx[c] = jr
                log_ev(f"T5-JOHYX-{c}", f"Hệ {c} + {len(x_i1)} biến X (I(1)): r = {jr['r_tr']} (trace), "
                                        f"{jr['r_me']} (max-eig)",
                       "; ".join(f"{row['H0']}: {row['Thống kê']:.2f} vs {row['Tới hạn']:.2f}"
                                 for row in jr["tr"].to_dict("records")),
                       f"Johansen tuần tự trên [{c}, {', '.join(x_i1)}], det_order={jr['det']}, k_ar_diff={jr['kd']}; "
                       f"chọn X: {x_rule}",
                       "Johansen (1991)",
                       "r > 0 chỉ cho biết có quan hệ dài hạn trong cả hệ, chưa chắc có mặt Y; nhạy với tập X được "
                       "chọn và số biến.")
            except Exception as e:
                joh_yx_err[c] = str(e)

    # --- Bước ngẫu nhiên: Ljung–Box + Variance Ratio (Holm qua lag/q) ---
    rw_rows, rw_res = [], {}
    for c in ys:
        r = Sd(c, 1)
        rwc = cached_rw(r, alpha)
        if rwc is None:
            continue
        lb_rej, vr_rej = rwc["lb_rej"], rwc["vr_rej"]
        arch_rej = arch_res.get(c, {}).get("rej", False)
        if not lb_rej and not vr_rej:
            concl = "Chưa bác bỏ bước ngẫu nhiên"
        elif lb_rej and vr_rej:
            concl = "Có cấu trúc phụ thuộc (cả hai test)"
        elif lb_rej and arch_rej:
            concl = "LB bác bỏ, VR không — có thể do phương sai đổi (ưu tiên VR)"
        else:
            concl = "Tín hiệu không đồng thuận"
        rw_res[c] = concl
        row = {"Chuỗi": c, "Biến đổi": RET_LABEL}
        for l, p in rwc["lb"].items():
            row[f"LB p (lag {l})"] = p
        for q, (v, z, p) in rwc["vr"].items():
            row[f"VR({q})"] = v
            row[f"VR({q}) p"] = p
        row["Kết luận"] = concl
        rw_rows.append(row)
        log_ev(f"T5-RW-{c}", f"{concl} (α={alpha})",
               "LB p: " + ", ".join(f"lag {l}={p:.3g}" for l, p in rwc["lb"].items())
               + "; " + ", ".join(f"VR({q})={v:.3f} (p={p:.3g})" for q, (v, z, p) in rwc["vr"].items()),
               f"Ljung–Box + Variance Ratio (z* bền vững) trên {RET_LABEL}; Holm qua lag và qua q",
               "Ljung & Box (1978); Lo & MacKinlay (1988)",
               "Chưa bác bỏ ≠ chứng minh bước ngẫu nhiên; Ljung–Box không bền vững với phương sai đổi.")

    # --- BDS và chẩn đoán phần dư trên chuỗi dừng hóa ---
    bds_res, bds_err, resid_diag = {}, None, None
    if p_use:
        try:
            bds_res = cached_bds(Yt, int(p_use), BDS_DIM, BDS_CAP)
            for c, r in bds_res.items():
                rej = bool((r["ph"] < alpha).any())
                log_ev(f"T5-BDS-{c}", ("Bác bỏ i.i.d. (còn phụ thuộc chưa giải thích)" if rej
                                       else "Chưa bác bỏ i.i.d.") + f" (α={alpha})",
                       "; ".join(f"dim {d}: BDS={s_:.2f}, p_Holm={p_:.3g}"
                                 for d, s_, p_ in zip(range(2, BDS_DIM + 1), r["stat"], r["ph"])),
                       f"BDS trên phần dư VAR/AR({int(p_use)}) của chuỗi dừng hóa; Holm qua chiều nhúng",
                       "Brock et al. (1996)", "Không phân biệt phi tuyến với phương sai đổi; đọc cùng ARCH-LM.")
        except Exception as e:
            bds_err = str(e)
        try:
            resid_diag = cached_resid_diag(Yt, int(p_use))
            log_ev("T5-RESID", f"Phần dư: tính trắng p={resid_diag['white_p']:.3g}, tính chuẩn p={resid_diag['norm_p']:.3g}",
                   f"nlags={resid_diag['nlags']}; n={resid_diag['n']}; K={resid_diag['K']}",
                   f"Portmanteau (VAR) / Ljung–Box (AR) + Jarque–Bera trên phần dư VAR/AR({int(p_use)})",
                   "Lütkepohl (2005)", "Phần dư của mô hình chỉ có Y; mô hình cuối có X sẽ khác.")
        except Exception:
            resid_diag = None

# ═══════════════════════════════════════════════════════════════════════════
# CÁC TAB KẾT QUẢ
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs(["1 · Chuỗi & biến động", "2 · Mùa vụ", "3 · Tính dừng", "4 · Bậc trễ & đồng liên kết",
                "5 · Bước ngẫu nhiên, phi tuyến & phần dư", "Quyết định"])

# ---------- Tab 1 ----------
with tabs[0]:
    st.subheader("Chuỗi giá và biến động (ARCH-LM)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Chuỗi {TX} (đoạn TRAIN)",
                                        f"Độ lệch chuẩn trượt {ROLL_WIN} kỳ của {RET_LABEL}"))
    for c in ys:
        fig.add_trace(go.Scatter(x=train.index, y=train[c], name=c, mode="lines"), 1, 1)
        r = Sd(c, 1)
        fig.add_trace(go.Scatter(x=r.index, y=r.rolling(ROLL_WIN).std(), name=f"std {c}", mode="lines",
                                 showlegend=False), 2, 1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, **STRETCH)
    st.caption("Biểu đồ chỉ để nhìn dữ liệu; kết luận dựa vào kiểm định bên dưới.")

    st.markdown(f"**ARCH-LM (Engle, 1982)**: H₀ là không có hiệu ứng ARCH, m = {ARCH_LAGS} lag, trên {RET_LABEL}.")
    if arch_res:
        st.dataframe(pd.DataFrame([{"Chuỗi": c, "Biến đổi": r["kind"], "n": r["n"], "LM": r["lm"], "p": r["p"],
                                    "Kết luận": "Có hiệu ứng ARCH" if r["rej"] else "Chưa có bằng chứng ARCH"}
                                   for c, r in arch_res.items()]).style.format({"LM": "{:.2f}", "p": "{:.4g}"}),
                     hide_index=True, **STRETCH)
        if any(r["rej"] for r in arch_res.values()):
            st.info("**Cách đọc:** p < α nghĩa là phương sai có điều kiện đổi theo thời gian, nên khoảng dự báo có "
                    "độ rộng cố định sẽ kém hiệu chuẩn. Kết quả này **không** suy ra “phải dùng deep learning”: "
                    "việc đó do so sánh mô hình ở các trang sau quyết định.")
        else:
            st.info("**Cách đọc:** chưa có bằng chứng phương sai đổi theo thời gian. Chưa bác bỏ không có nghĩa là "
                    "phương sai chắc chắn không đổi.")
    else:
        st.info("Không đủ quan sát để chạy ARCH-LM.")

    st.subheader("Ổn định tham số và ngoại lai")
    st.caption(f"Trên {RET_LABEL}: Welch so trung bình hai nửa mẫu, Levene so phương sai hai nửa, CUSUM kiểm tra "
               f"trung bình ổn định; ngoại lai là |z| > {OUTLIER_Z:g} theo z bền vững (median/MAD). "
               "Dùng để nghi ngờ gãy cấu trúc khi ADF và KPSS mâu thuẫn.")
    if stab_res:
        st.dataframe(pd.DataFrame([{"Chuỗi": c, "n": s_["n"], "Trung bình hai nửa (p)": s_["mean_p"],
                                    "Phương sai hai nửa (p)": s_["var_p"], "CUSUM (p)": s_["cusum_p"],
                                    "Ngoại lai": s_["n_out"], "% ngoại lai": s_["share_out"],
                                    "Kết luận": "Kém ổn định" if s_["unstable"] else "Chưa thấy bất ổn"}
                                   for c, s_ in stab_res.items()]).style.format(
            {"Trung bình hai nửa (p)": "{:.3g}", "Phương sai hai nửa (p)": "{:.3g}", "CUSUM (p)": "{:.3g}",
             "% ngoại lai": "{:.2%}"}, na_rep="—"), hide_index=True, **STRETCH)
        if any(s_["unstable"] for s_ in stab_res.values()):
            st.warning("Có chuỗi kém ổn định tham số: kết quả ADF/ACF trên toàn đoạn TRAIN có thể bị kéo bởi một vài "
                       "giai đoạn (ví dụ các cú sốc giá cà phê). Cân nhắc đối chiếu thêm theo từng giai đoạn.")
    else:
        st.info("Không đủ quan sát để kiểm tra ổn định.")

# ---------- Tab 2 ----------
with tabs[1]:
    st.subheader("Mùa vụ (STL + Kruskal–Wallis)")
    st.caption(f"F_S = max(0, 1 − Var(R)/Var(S+R)) từ STL (Cleveland et al., 1990). Ngưỡng thực dùng = max("
               f"{SEASON_FS_THR}, phân vị {SEASON_Q:.0%} của F_S khi hoán vị Δ rồi cộng dồn, {season_sim} lần), vì chuỗi "
               "bước ngẫu nhiên ngắn cũng có F_S dương do nhiễu. Độ ổn định của phân vị được kiểm tra bằng KTC "
               f"bootstrap 95% ({SEASON_BOOT} lần) và chênh lệch giữa hai nửa số lần mô phỏng. Kruskal–Wallis so sánh "
               "Δ chuẩn hóa theo độ lệch chuẩn trượt giữa các tháng. Chỉ kết luận “có mùa vụ” khi F_S ≥ ngưỡng "
               "**và** p < α.")
    if per is None or per < 4:
        st.warning("Cần chỉ mục thời gian đủ dày để suy ra chu kỳ năm.")
    else:
        st.caption(f"Chu kỳ mùa vụ năm: **{per}** kỳ/năm — nguồn: {per_src}.")
        if per_src.startswith("ước lượng"):
            st.warning("Dữ liệu không theo lưới đều nên chu kỳ chỉ là ước lượng từ số dòng/năm. Có thể nhập chu kỳ "
                       "đúng ở “Tùy chọn dữ liệu nâng cao”.")
        rows = []
        for c, r in season_res.items():
            if "err" in r:
                rows.append({"Chuỗi": c, "Kết luận": "Lỗi: " + r["err"]})
            else:
                rows.append({"Chuỗi": c, "Chu kỳ": r["period"], "F_S": r["fs"], "Ngưỡng giả lập": r["thr_sim"],
                             "KTC 95% ngưỡng": f"[{r['thr_lo']:.3f}; {r['thr_hi']:.3f}]",
                             "Chênh hai nửa": r["half_diff"],
                             "Ngưỡng ổn định": "Có" if r["ci_stable"] else "Chưa (tăng số lần)",
                             "Ngưỡng dùng": r["thr_eff"], "KW theo tháng (p)": r["p_m"], "Kết luận": r["concl"]})
        st.dataframe(pd.DataFrame(rows).style.format({"F_S": "{:.3f}", "Ngưỡng giả lập": "{:.3f}",
                                                      "Chênh hai nửa": "{:.3f}", "Ngưỡng dùng": "{:.3f}",
                                                      "KW theo tháng (p)": "{:.4g}"}, na_rep="—"),
                     hide_index=True, **STRETCH)
        if any(("err" not in r) and not r["ci_stable"] for r in season_res.values()):
            st.warning(f"KTC của ngưỡng rộng hơn {SEASON_CI_WIDTH}: phân vị {SEASON_Q:.0%} chưa ổn định với "
                       f"{season_sim} lần mô phỏng. Tăng số lần ở “Tùy chọn dữ liệu nâng cao” trước khi tin kết luận.")
        okc = [c for c, r in season_res.items() if "err" not in r]
        if okc:
            cs_ = st.selectbox("Xem chi tiết chuỗi", okc, key="seas_view")
            s_ = S(cs_)
            d1 = Sd(cs_, 1)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Thành phần mùa vụ (STL)", f"{RET_LABEL} theo tháng"))
            fig.add_trace(go.Scatter(x=s_.index, y=season_res[cs_]["seasonal"], name="seasonal"), 1, 1)
            for mth in range(1, 13):
                g = d1[d1.index.month == mth]
                if len(g):
                    fig.add_trace(go.Box(y=g, name=str(mth), showlegend=False), 1, 2)
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig, **STRETCH)
        st.caption("Đối chiếu lịch thu hoạch cà phê Việt Nam trước khi diễn giải nguyên nhân mùa vụ.")

# ---------- Tab 3 ----------
with tabs[2]:
    st.subheader("Tính dừng và bậc sai phân d (ADF + KPSS)")
    st.markdown(
        "| Kiểm định | H₀ | Bác bỏ khi |\n|---|---|---|\n"
        "| **ADF** (Dickey & Fuller, 1979) | Có nghiệm đơn vị | p < α |\n"
        "| **KPSS** (Kwiatkowski et al., 1992) | Chuỗi dừng | thống kê > giá trị tới hạn mức α |")
    st.caption(f"Quy tắc d: bậc nhỏ nhất mà **ADF bác bỏ** nghiệm đơn vị; KPSS dùng để xác nhận"
               + (" và, ở mức, nếu hai test mâu thuẫn thì xét tiếp sai phân (chính sách thận trọng). "
                  if conservative else ". ")
               + f"Cấu hình: mức dùng '{reg}', Δ/Δ² dùng 'c'; ADF chọn lag theo AIC; KPSS nlags='auto'; α = {alpha}. "
               f"ACF(1) < {ODIFF_ACF1} của chuỗi đã sai phân báo nguy cơ sai phân quá mức (tham số ứng dụng).")

    def _warn_txt(c):
        w = []
        if conf_map.get(c, "").startswith("Chỉ ADF") and d_map.get(c) == 0:
            w.append("ADF/KPSS mâu thuẫn ở mức")
        if conf_map.get(c, "").startswith("Thận trọng"):
            w.append("mức mâu thuẫn → d tăng")
        if acf_flag.get(c):
            w.append("sai phân quá mức")
        return "; ".join(w)

    summ = pd.DataFrame([{"Chuỗi": c, "Vai trò": "Y" if c in ys else "X",
                          "Bậc d": (f"chưa xác định → tạm d = {d_eff[c]}" if d_map.get(c) is None else d_map[c]),
                          "Độ tin cậy": conf_map.get(c, ""),
                          "Sai phân quá mức": "⚠ Có" if acf_flag.get(c) else "Không",
                          "Lưu ý": _warn_txt(c)} for c in use])
    st.markdown("**Tóm tắt bậc tích hợp**")
    st.dataframe(summ, hide_index=True, **STRETCH)
    if any(str(v).startswith("Chỉ ADF") for v in conf_map.values()):
        st.warning("Có chuỗi ở trạng thái “Chỉ ADF”: ADF và KPSS cùng bác bỏ. Nguyên nhân thường là dừng quanh xu "
                   "hướng hoặc gãy cấu trúc (giá cà phê có các cú sốc lớn): thử cấu hình 'hằng số + xu hướng' hoặc "
                   "đối chiếu mục Ổn định tham số ở tab 1. Nếu d = 0 ở trạng thái này, đừng coi là dừng chắc chắn.")
    if any(str(v).startswith("Thận trọng") for v in conf_map.values()):
        st.info("Có chuỗi được chọn d cao hơn theo chính sách thận trọng (mức ADF/KPSS mâu thuẫn). Tắt tùy chọn này ở "
                "phần “Tùy chọn dữ liệu nâng cao” nếu muốn theo ADF thuần.")
    if any(acf_flag.values()):
        st.warning("Có chuỗi bị cờ sai phân quá mức: cân nhắc giữ bậc d thấp hơn hoặc dùng mức.")
    if d_forced:
        st.error("Chưa xác định được bậc d cho: " + ", ".join(d_forced) + f". Các bước sau đang TẠM dùng d = {d_fb} "
                 "cho những chuỗi này (không phải kết luận kiểm định); việc này đã được ghi vào thẻ minh chứng "
                 "T5-DFB-* và vào `d_unresolved` khi lưu. Cần xem lại dữ liệu hoặc chốt d bằng tay ở Trang 13.")

    st.markdown("**Đoạn liên tục dùng cho kiểm định**")
    st.caption("Mỗi chuỗi được kiểm định trên đoạn liên tục dài nhất (không nối các đoạn qua khoảng trống); Johansen, "
               "VAR và BDS dùng đoạn liên tục dài nhất chung của các cột liên quan.")
    _bt = pd.DataFrame([{"Chuỗi": c, "Vai trò": b["role"], "Quan sát có giá trị": b["n_obs"],
                         "Đoạn liên tục dài nhất": b["n_block"],
                         "Tỷ lệ": (b["n_block"] / b["n_obs"]) if b["n_obs"] else np.nan,
                         "Từ": b["start"].date() if b["start"] is not None else "—",
                         "Đến": b["end"].date() if b["end"] is not None else "—"} for c, b in block_info.items()])
    st.dataframe(_bt.style.format({"Tỷ lệ": "{:.1%}"}, na_rep="—"), hide_index=True, **STRETCH)
    if (_bt["Tỷ lệ"] < 0.8).any():
        st.warning("Có chuỗi mà đoạn liên tục dài nhất chiếm < 80% quan sát: phần còn lại bị loại khỏi kiểm định vì "
                   "bị ngắt bởi khoảng trống. Cân nhắc tăng “Điền tiến tối đa” hoặc xử lý khuyết ở Trang 1.")
    with st.expander("Chi tiết từng phép kiểm định"):
        tbl = pd.DataFrame(stat_rows)
        fmt = {k: v for k, v in {"ADF stat": "{:.3f}", "ADF p": "{:.4g}", "KPSS stat": "{:.3f}",
                                 "ACF(1)": "{:.3f}"}.items() if k in tbl.columns}
        st.dataframe(tbl.style.format(fmt, na_rep="—"), hide_index=True, height=380, **STRETCH)

# ---------- Tab 4 ----------
with tabs[3]:
    st.subheader("Bậc trễ p (ACF/PACF và AIC/BIC/HQ)")
    cs_ = st.selectbox("Chuỗi cho ACF/PACF (dừng hóa theo d)", ys, key="acf_view")
    k_v = int(d_eff[cs_])
    z = Sd(cs_, k_v) if k_v > 0 else S(cs_)
    if cs_ in d_forced:
        st.warning(f"d của {cs_} chưa xác định: đang hiển thị với d tạm = {k_v}.")
    nl = int(min(40, len(z) // 2 - 2))
    if nl >= 2:
        a = acf(z, nlags=nl, fft=True)[1:]
        pa = pacf(z, nlags=nl, method="ywm")[1:]
        band = 1.96 / np.sqrt(len(z))
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ACF của {LAB[k_v]}", f"PACF của {LAB[k_v]}"))
        for c_i, vals in enumerate([a, pa], start=1):
            fig.add_trace(go.Bar(x=list(range(1, nl + 1)), y=vals, showlegend=False), 1, c_i)
            for sgn in (1, -1):
                fig.add_shape(type="line", x0=0, x1=nl + 1, y0=sgn * band, y1=sgn * band,
                              line=dict(dash="dash", color="#ef4444", width=1), row=1, col=c_i)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, **STRETCH)
        st.caption(f"Đường đỏ: ±1.96/√n. Với {nl} lag, khoảng {0.05 * nl:.1f} cột vượt ngưỡng chỉ do ngẫu nhiên. "
                   "PACF chỉ gợi ý bậc AR cho một chuỗi; bậc cho mô hình đa biến chốt bằng AIC/BIC/HQ.")
    else:
        st.info("Không đủ quan sát để vẽ ACF/PACF.")

    if ic_d is not None:
        fig = go.Figure()
        for k in ic_d.columns:
            fig.add_trace(go.Scatter(x=ic_d.index, y=ic_d[k], name=k, mode="lines+markers"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="p",
                          yaxis_title="giá trị tiêu chí")
        st.plotly_chart(fig, **STRETCH)
        st.dataframe(pd.DataFrame([best_p], index=["p tối ưu"]), **STRETCH)
        if len(set(best_p.values())) > 1:
            st.warning(f"Ba tiêu chí cho p khác nhau ({best_p}). Đang dùng **{crit}: p = {p_use}**.")
        else:
            st.success(f"Ba tiêu chí cùng chọn p = {p_use}.")
        st.caption(f"Tính trên chuỗi Y dừng hóa theo d ({T_NOTE}); VAR nếu K ≥ 2, AR (mẫu chung qua hold_back) nếu "
                   f"K = 1; p tối đa {IC_MAXLAG}. Chưa có X nên đây là gợi ý ban đầu cho VARX.")
    else:
        st.error(f"Không tính được tiêu chí thông tin: {ic_err}")

    st.subheader("Đồng liên kết giữa các chuỗi Y (Johansen)")
    st.caption(f"Johansen (1991), kiểm định tuần tự H₀: r ≤ 0, r ≤ 1, …; dừng ở lần không bác bỏ đầu tiên. "
               f"Chỉ áp dụng khi mọi Y đều d = 1. Cấu hình: det_order = {0 if reg == 'c' else 1} "
               f"({'hằng số' if reg == 'c' else 'hằng số + xu hướng'}), mức {ALPHA_KEY[alpha]}, k_ar_diff = p − 1 với "
               f"p chọn theo {crit} trên dữ liệu mức (p tối đa {IC_MAXLAG}).")
    if joh is None:
        st.info(joh_msg)
    else:
        ca, cb = st.columns(2)
        ca.markdown("**Trace**")
        ca.dataframe(joh["tr"].style.format({"Thống kê": "{:.3f}", "Tới hạn": "{:.3f}"}), hide_index=True, **STRETCH)
        cb.markdown("**Max-eigenvalue**")
        cb.dataframe(joh["me"].style.format({"Thống kê": "{:.3f}", "Tới hạn": "{:.3f}"}), hide_index=True, **STRETCH)
        st.metric("Số quan hệ đồng liên kết r (trace / max-eig)", f"{joh['r_tr']} / {joh['r_me']}")
        st.caption(f"k_ar_diff = {joh['kd']}, n = {joh['n']}.")
        if joh["r_tr"] != joh["r_me"]:
            st.warning("Trace và max-eigenvalue cho r khác nhau: nêu cả hai trong báo cáo.")
        if joh["r_tr"] == 0:
            st.success("r = 0 (theo trace): chưa có bằng chứng đồng liên kết, dùng VAR/VARX trên **sai phân**.")
        elif joh["r_tr"] < joh["K"]:
            st.info(f"0 < r = {joh['r_tr']} < K = {joh['K']}: có bằng chứng quan hệ dài hạn, dùng **VECM** hoặc ghi rõ "
                    "rằng sai phân làm mất thông tin dài hạn.")
        else:
            st.error("r = K: các chuỗi thực ra dừng, mâu thuẫn với tab Tính dừng; cần kiểm tra lại.")

    st.subheader("Đồng liên kết giữa Y và X (khi X là I(1))")
    if not xs:
        st.info("Chưa có biến X từ Trang 4.")
    elif not x_i1:
        st.info("Không có biến X nào có d = 1, nên không xét đồng liên kết Y–X.")
    elif not joh_yx and not joh_yx_err:
        st.info("Không có chuỗi Y nào có d = 1 để ghép với X.")
    else:
        st.caption(f"Với mỗi Y có d = 1, chạy Johansen trên hệ [Y + {len(x_i1)} biến X có d = 1"
                   + (f", lấy {MAX_X_JOH} biến đầu" if len([c for c in xs if d_map.get(c) == 1]) > MAX_X_JOH else "")
                   + f"]: {', '.join(x_i1)}. r > 0 là quan hệ dài hạn trong cả hệ, chưa chắc có Y; kết quả nhạy với số biến.")
        rows = [{"Y": c, "K": j["K"], "n": j["n"], "k_ar_diff": j["kd"], "r (trace)": j["r_tr"],
                 "r (max-eig)": j["r_me"]} for c, j in joh_yx.items()]
        rows += [{"Y": c, "Lỗi": e} for c, e in joh_yx_err.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)

# ---------- Tab 5 ----------
with tabs[4]:
    st.subheader("Bước ngẫu nhiên (Ljung–Box + Variance Ratio)")
    st.caption(f"Ljung–Box (1978) trên {RET_LABEL}: H₀ không tự tương quan. VR(q) = 1 dưới bước ngẫu nhiên; "
               "z* bền vững với phương sai đổi (Lo & MacKinlay, 1988). Mỗi nhóm test (3 lag của LB, 3 giá trị q của VR) "
               "được hiệu chỉnh Holm trước khi so với α. Ljung–Box không bền vững với phương sai đổi nên khi ARCH-LM "
               "bác bỏ, ưu tiên VR.")
    if rw_rows:
        rdf = pd.DataFrame(rw_rows)
        fm = {k: "{:.3f}" for k in rdf.columns if k.startswith("VR(") and not k.endswith("p")}
        fm.update({k: "{:.3g}" for k in rdf.columns if k.endswith("p") or "LB p" in k})
        st.dataframe(rdf.style.format(fm), hide_index=True, **STRETCH)
        st.info("**Cách đọc:** nếu ΔY gần nhiễu trắng thì dự báo “không đổi” (Naive) đã là dự báo điểm tốt nhất "
                "cho bước ngẫu nhiên, nên đây là cơ sở đặt kỳ vọng khi so sánh mô hình. Có cấu trúc phụ thuộc chỉ "
                "nghĩa là có chỗ để thắng Naive, chưa phải bằng chứng đã thắng.")
    else:
        st.info("Không đủ quan sát để chạy Ljung–Box / Variance Ratio.")

    st.subheader("Phi tuyến (BDS)")
    st.caption("Brock, Dechert, Scheinkman & LeBaron (1996). H₀: phần dư i.i.d. Chạy trên phần dư VAR/AR của chuỗi dừng "
               f"hóa (p = {p_use}; {T_NOTE}), tối đa {BDS_CAP:,} phần dư gần nhất, chiều nhúng 2–{BDS_DIM}, p hiệu "
               "chỉnh Holm. Bác bỏ nghĩa là còn phụ thuộc chưa được mô hình tuyến tính giải thích: có thể là "
               "**phi tuyến hoặc phương sai đổi**, nên đọc cùng ARCH-LM ở tab 1.")
    if bds_res:
        for c, r in bds_res.items():
            tb = pd.DataFrame({"Chiều nhúng": list(range(2, 2 + len(r["p"]))), "BDS": r["stat"],
                               "p": r["p"], "p (Holm)": r["ph"]})
            st.markdown(f"**{c}** (n = {r['n']})")
            st.dataframe(tb.style.format({"BDS": "{:.3f}", "p": "{:.4g}", "p (Holm)": "{:.4g}"}),
                         hide_index=True, **STRETCH)
            arch_txt = f"ARCH-LM p = {arch_res[c]['p']:.3g}" if c in arch_res else "ARCH-LM chưa chạy"
            if bool((r["ph"] < alpha).any()):
                st.warning(f"BDS bác bỏ i.i.d.: còn phụ thuộc phi tuyến **hoặc** phương sai đổi ({arch_txt}).")
            else:
                st.success("BDS chưa bác bỏ i.i.d. ở các chiều nhúng đã chạy.")
    else:
        st.info(f"Chưa có kết quả BDS{': ' + bds_err if bds_err else ' (cần bậc trễ p từ tab 4)'}.")

    st.subheader("Chẩn đoán phần dư VAR/AR")
    if resid_diag:
        m1, m2 = st.columns(2)
        m1.metric(f"Tính trắng (Portmanteau/Ljung–Box, {resid_diag['nlags']} lag), p", f"{resid_diag['white_p']:.3g}")
        m2.metric("Tính chuẩn (Jarque–Bera), p", f"{resid_diag['norm_p']:.3g}")
        if resid_diag["white_p"] < alpha:
            st.warning("Phần dư còn tự tương quan: p có thể quá nhỏ hoặc mô hình thiếu thành phần; thử tăng p. Kết quả "
                       "BDS khi đó cũng khó diễn giải.")
        else:
            st.success("Chưa bác bỏ tính trắng của phần dư.")
        if resid_diag["norm_p"] < alpha:
            st.info("Phần dư không chuẩn (thường do đuôi dày/ngoại lai): khoảng dự báo dựa trên giả định chuẩn sẽ kém "
                    "hiệu chuẩn, nên dùng khoảng theo phân vị thực nghiệm.")
        st.caption("Phần dư của mô hình chỉ có Y, chưa gồm X; dùng để đặt kỳ vọng, chưa thay cho chẩn đoán mô hình cuối.")
    else:
        st.info("Chưa có chẩn đoán phần dư (cần bậc trễ p từ tab 4).")

# ---------- Quyết định ----------
with tabs[5]:
    st.subheader("Bảng quyết định biến đổi")
    st.caption("Mỗi dòng truy ngược được về thẻ minh chứng. Bảng này là **gợi ý**: người dùng vẫn là người chốt ở "
               "Trang 13 (Chuẩn bị thực nghiệm).")
    K = len(ys)
    rows = []
    for c in ys:
        d = d_map.get(c)
        d_txt = "chưa xác định" if d is None else str(d)
        conf_c = conf_map.get(c, "")
        r_ = joh["r_tr"] if joh else None
        co = f"r = {r_}/{K}" if joh else "không áp dụng"
        jyx = joh_yx.get(c)
        co_yx = f"r = {jyx['r_tr']} (hệ {jyx['K']} biến)" if jyx else "không áp dụng"
        a_ = arch_res.get(c)
        ar_txt = f"p = {a_['p']:.3g} → {'có' if a_['rej'] else 'chưa có bằng chứng'}" if a_ else "—"
        se = season_res.get(c)
        se_txt = se["concl"] if se and "err" not in se else "chưa đủ dữ liệu"
        rw_txt = rw_res.get(c, "—")
        bd_ = bds_res.get(c)
        bd_txt = ("bác bỏ i.i.d." if bool((bd_["ph"] < alpha).any()) else "chưa bác bỏ") if bd_ else "—"
        st_ = stab_res.get(c)
        st_txt = ("kém ổn định" if st_["unstable"] else "chưa thấy bất ổn") if st_ else "—"
        if d == 0:
            sug = "Giữ mức"
            if conf_c.startswith("Chỉ ADF"):
                sug = "Giữ mức — CẢNH BÁO: nên đối chiếu d = 1 (ADF/KPSS mâu thuẫn)"
        elif d == 1 and r_ is not None and r_ == 0:
            sug = "Sai phân bậc 1 (VAR/VARX trên Δ)"
        elif d == 1 and r_ is not None and 0 < r_ < K:
            sug = "VECM, hoặc Δ kèm ghi chú mất quan hệ dài hạn"
        elif d == 1:
            sug = "Sai phân bậc 1"
        elif d == 2:
            sug = "Sai phân bậc 2 (cần xem lại, dễ sai phân quá mức)"
        else:
            sug = "Chưa đủ bằng chứng"
        warns = []
        if d == 0 and conf_c.startswith("Chỉ ADF"):
            warns.append("ADF/KPSS mâu thuẫn ở mức")
        if conf_c.startswith("Thận trọng"):
            warns.append("d chọn theo chính sách thận trọng")
        if acf_flag.get(c):
            warns.append("nguy cơ sai phân quá mức")
        if st_ and st_["unstable"]:
            warns.append("tham số kém ổn định (nghi gãy cấu trúc)")
        if resid_diag and resid_diag["white_p"] < alpha:
            warns.append("phần dư còn tự tương quan")
        ids = [k for k in [f"T5-{t}-{c}" for t in ("STAT", "STAB", "ARCH", "SEAS", "RW", "BDS", "JOHYX")] if k in EV]
        ids += [k for k in ("T5-JOH-system", "T5-LAG-Δ", "T5-RESID") if k in EV]
        rows.append({"Chuỗi": c, "Bậc sai phân d": d_txt, "Độ tin cậy d": conf_c,
                     "Đồng liên kết (hệ thống Y)": co, "Đồng liên kết Y–X": co_yx, "ARCH-LM": ar_txt,
                     "Mùa vụ": se_txt, "Bước ngẫu nhiên": rw_txt, "Phi tuyến (BDS)": bd_txt,
                     "Ổn định tham số": st_txt, "Cảnh báo": "; ".join(warns), "Gợi ý biến đổi": sug,
                     "ID bằng chứng": "; ".join(ids)})
    dec = pd.DataFrame(rows)
    st.dataframe(dec, hide_index=True, **STRETCH)

    # --- Bảng cho X ---
    decx = pd.DataFrame()
    if xs:
        st.markdown("**Biến X (từ Trang 4)**")
        xrows = []
        for c in xs:
            d = d_map.get(c)
            conf_c = conf_map.get(c, "")
            if d is None:
                sug = "Chưa đủ bằng chứng"
            elif d == 0:
                sug = "Giữ mức" + (" — CẢNH BÁO: ADF/KPSS mâu thuẫn" if conf_c.startswith("Chỉ ADF") else "")
            elif d == 1:
                sug = "Sai phân bậc 1"
            else:
                sug = "Sai phân bậc 2 (cần xem lại)"
            w = []
            if acf_flag.get(c):
                w.append("nguy cơ sai phân quá mức")
            if conf_c.startswith("Thận trọng"):
                w.append("d chọn theo chính sách thận trọng")
            xrows.append({"Chuỗi": c, "Bậc sai phân d": "chưa xác định" if d is None else d,
                          "Độ tin cậy d": conf_c, "Cảnh báo": "; ".join(w), "Gợi ý biến đổi": sug,
                          "ID bằng chứng": f"T5-STAT-{c}" if f"T5-STAT-{c}" in EV else ""})
        decx = pd.DataFrame(xrows)
        st.dataframe(decx, hide_index=True, **STRETCH)

    d_known = [d_map[c] for c in ys if d_map.get(c) is not None]
    diff_suggest = int(max(d_known)) if d_known else 0
    x_known = {c: d_map[c] for c in xs if d_map.get(c) is not None}
    st.markdown(f"**Gợi ý chuyển sang Trang 13:** bậc sai phân Y = **{diff_suggest}** (chuỗi {TX})"
                + (f"; bậc sai phân X: " + ", ".join(f"{c}={d}" for c, d in x_known.items()) if x_known else "")
                + (f"; bậc trễ p = **{p_use}** ({crit})" if p_use else "") + ".")
    if len(set(d_known)) > 1:
        st.warning("Các chuỗi Y có bậc d khác nhau: `diff_order` chỉ lấy giá trị lớn nhất. Bậc từng chuỗi được lưu ở "
                   "`diff_order_by_series`.")

    c1, c2 = st.columns(2)
    c1.download_button("Tải bảng quyết định (CSV)", dec.to_csv(index=False).encode("utf-8-sig"),
                       "quyet_dinh_bien_doi.csv", "text/csv")
    evdf = pd.DataFrame(list(EV.values())).drop(columns=["sig", "Trang"]) if EV else pd.DataFrame()
    if len(evdf):
        c2.download_button("Tải thẻ minh chứng (CSV)", evdf.to_csv(index=False).encode("utf-8-sig"),
                           "the_minh_chung_trang5.csv", "text/csv")
        with st.expander(f"Xem {len(evdf)} thẻ minh chứng"):
            st.dataframe(evdf, hide_index=True, **STRETCH)

    if st.button("Lưu kết quả cho các trang sau", type="primary", **STRETCH):
        scopes = dict(st.session_state.get("analysis_scopes", {}))
        scopes[SCOPE_NAME] = str(last_train.date())
        log = st.session_state.setdefault("evidence_log", {})
        for k in [k for k in log if k.startswith("T5-")]:
            del log[k]
        log.update(EV)
        profile = {
            "ys": ys, "xs": xs, "use_log": use_log, "alpha": alpha, "reg": reg, "criterion": crit,
            "train_end": str(last_train.date()), "train_cut_mode": cut_mode,
            "d_y": {c: d_map.get(c) for c in ys}, "d_x": {c: d_map.get(c) for c in xs},
            "d_confidence": {c: conf_map.get(c, "") for c in use},
            "johansen_y": ({"r_trace": joh["r_tr"], "r_maxeig": joh["r_me"], "K": joh["K"], "k_ar_diff": joh["kd"]}
                           if joh else None),
            "johansen_yx": {c: {"r_trace": j["r_tr"], "r_maxeig": j["r_me"], "K": j["K"]} for c, j in joh_yx.items()},
            "seasonality": {c: ({"conclusion": r["concl"], "period": r["period"], "fs": r["fs"]}
                                if "err" not in r else {"error": r["err"]}) for c, r in season_res.items()},
            "arch": {c: bool(r["rej"]) for c, r in arch_res.items()},
            "random_walk": dict(rw_res),
            "bds_reject": {c: bool((r["ph"] < alpha).any()) for c, r in bds_res.items()},
            "unstable": {c: bool(s_["unstable"]) for c, s_ in stab_res.items()},
            "resid": ({"white_p": resid_diag["white_p"], "norm_p": resid_diag["norm_p"]} if resid_diag else None),
            "p": p_use,
        }
        st.session_state.update({
            "transform_decision_df": dec,
            "transform_decision_x_df": decx,
            "diff_order": diff_suggest,
            "diff_order_by_series": {c: d_map.get(c) for c in use},
            "analysis_scopes": scopes,
            "var_lag_suggestion": {"criterion": crit, "p": p_use, "best": best_p,
                                   "data": "Δ" if all(t_lab.get(c) == "Δ" for c in ys) else f"dừng hóa theo d ({T_NOTE})"},
            "ts_profile": profile,
            "use_log_y": use_log,
        })
        st.success(f"Đã lưu: d = {diff_suggest}, p = {p_use}, TRAIN đến {last_train.date()}, {len(EV)} thẻ minh chứng.")

# ============================================================
# ĐIỀU HƯỚNG
# ============================================================
st.divider()
n1, n2 = st.columns(2)
with n1:
    if st.button("← Về Trang 4 (tương quan)", key="nav_back", **STRETCH):
        st.switch_page("pages/2_Phan_tich_tuong_quan.py")
with n2:
    if st.button("Tiếp theo: Giới thiệu mô hình →", key="nav_next", **STRETCH):
        st.switch_page("pages/4_Gioi_thieu_mo_hinh.py")
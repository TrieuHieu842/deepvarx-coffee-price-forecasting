# -*- coding: utf-8 -*-
"""
DeepVARX — Trang 3: Phân tích tương quan & quan hệ dự báo (sàng lọc biến X)

Đặt tại: pages/2_Phan_tich_tuong_quan.py  (chạy sau pages/1_Tai_du_lieu.py)
Cần: streamlit, pandas, numpy, scipy, plotly

LUỒNG
    0. Nạp dữ liệu   -> lấy từ Trang 1 (session_state); không có thì upload / demo
    1. Chọn cột      -> ngày, Y (giá), loại cà phê, các X, tham số
    2. Chuẩn bị      -> gộp theo ngày, cắt TRAIN, lấy sai phân
    3. Bốn phân tích -> Y–Y | X–X (+VIF) | CCF | Granger
    4. Shortlist     -> biến X nên giữ, tải CSV, lưu cho trang sau

CÁC PHƯƠNG PHÁP ĐÃ SÀNG LỌC
    Giữ   : tương quan trên SAI PHÂN (tránh hồi quy giả), VIF, CCF có tiền trắng hóa,
            Granger (Wald F với sai số chuẩn bền HC1)
    Sửa   : ngưỡng CCF hiệu chỉnh theo số lag (nếu không, ~5% lag vượt ngưỡng do ngẫu nhiên,
            gần như biến nào cũng "có lag"); Granger dùng sai số bền vì sai phân giá có
            phương sai thay đổi; lag Granger dựng trên lưới thời gian đầy đủ, không dồn dòng
    Bỏ   : không dùng ADF/KPSS ở đây (thuộc Trang 5), không kiểm định nhân quả thật sự
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ============================================================
# CONFIG (đồng bộ với Trang 1)
# ============================================================
APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá thu mua cà phê Việt Nam"
PROVINCE_COL = "province"
COFFEE_TYPE_COL = "coffee_type"
ALL_TYPES = "(Tất cả loại — lấy trung bình)"

# Tham số ứng dụng (không phải chuẩn học thuật)
MIN_RHO_Y = 0.5          # ρ tối thiểu giữa các tỉnh để dùng 1 chuỗi đại diện
HIGH_CORR_X = 0.8        # |ρ| coi là X trùng thông tin
VIF_WARN, VIF_STRONG = 5.0, 10.0
ZERO_DIFF_FLAG = 50.0    # % sai phân = 0 để cảnh báo X "đứng yên"
MIN_OBS = 40             # số quan sát tối thiểu cho CCF

st.set_page_config(page_title="Phân tích tương quan", page_icon="☕", layout="wide")


def _streamlit_version():
    return tuple(int(re.sub(r"\D", "", p) or 0) for p in st.__version__.split(".")[:2])


STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


# ============================================================
# HÀM TIỆN ÍCH
# ============================================================
def demo_data():
    """Demo: 3 tỉnh × 2 loại; rain dẫn trước giá 3 kỳ, oil trùng thông tin với usd."""
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    rain_shock, usd_shock = rng.normal(size=n), rng.normal(size=n)
    rain, usd = np.cumsum(rain_shock), np.cumsum(usd_shock)
    oil = np.cumsum(0.9 * usd_shock + rng.normal(scale=0.3, size=n))
    noise = np.cumsum(rng.normal(size=n))            # biến không liên quan
    common = np.cumsum(0.8 * np.roll(rain_shock, 3) + rng.normal(size=n))
    rows = []
    for ctype, base in [("Robusta", 50000), ("Arabica", 90000)]:
        for prov in ["Dak Lak", "Lam Dong", "Gia Lai"]:
            price = base + 100 * (common + np.cumsum(rng.normal(scale=0.5, size=n)))
            rows.append(pd.DataFrame({"date": dates, PROVINCE_COL: prov, COFFEE_TYPE_COL: ctype,
                                      "price": price, "rain": rain, "usd": usd, "oil": oil, "noise": noise}))
    return pd.concat(rows, ignore_index=True)


def corr_heatmap(mat, title):
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(mat.values, 2), texttemplate="%{text}"))
    fig.update_layout(title=title, height=max(300, 45 * len(mat) + 110), margin=dict(t=45, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig


def vif_table(X):
    """VIF_j = phần tử đường chéo của nghịch đảo ma trận tương quan."""
    Xc = X.dropna()
    if len(Xc) < X.shape[1] + 5:
        return pd.DataFrame({"Biến X": X.columns, "VIF": np.nan})
    vif = np.diag(np.linalg.pinv(Xc.corr().values))
    return pd.DataFrame({"Biến X": X.columns, "VIF": vif}).sort_values("VIF", ascending=False)


def fit_ar_filter(x, max_p=10):
    """Chọn bậc AR theo AIC, trả hệ số phi (bộ lọc tiền trắng hóa)."""
    x = np.asarray(x, float) - np.mean(x)
    best_phi, best_aic = np.array([0.0]), np.inf
    for p in range(1, max_p + 1):
        A = np.column_stack([x[p - i - 1:len(x) - i - 1] for i in range(p)])
        b = x[p:]
        phi = np.linalg.lstsq(A, b, rcond=None)[0]
        s2 = max(np.mean((b - A @ phi) ** 2), 1e-12)
        aic = np.log(s2) + 2 * p / len(b)
        if aic < best_aic:
            best_phi, best_aic = phi, aic
    return best_phi


def apply_ar_filter(x, phi):
    x = np.asarray(x, float) - np.mean(x)
    p = len(phi)
    return x[p:] - sum(phi[i] * x[p - i - 1:len(x) - i - 1] for i in range(p))


def ccf_table(y, x, max_lag, alpha):
    """
    CCF(k) = Corr(Y_t, X_{t-k}); k > 0: X dẫn trước Y.
    X và Y lọc bằng CÙNG bộ lọc AR (khớp trên X) -> ngưỡng ±z/√n hợp lệ.
    Hai ngưỡng: từng lag (α) và hiệu chỉnh Bonferroni theo max_lag lag.
    """
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(d) < MIN_OBS + max_lag:
        return None, None, None, len(d)
    phi = fit_ar_filter(d["x"].values)
    yf, xf = apply_ar_filter(d["y"].values, phi), apply_ar_filter(d["x"].values, phi)
    n = len(yf)
    if np.std(yf) == 0 or np.std(xf) == 0:
        return None, None, None, n
    r = [np.corrcoef(yf[k:], xf[:n - k])[0, 1] for k in range(max_lag + 1)]
    band = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    band_adj = stats.norm.ppf(1 - alpha / (2 * max_lag)) / np.sqrt(n)
    out = pd.DataFrame({"Lag k": range(max_lag + 1), "CCF": r})
    out["Vượt ngưỡng từng lag"] = out["CCF"].abs() > band
    out["Vượt ngưỡng hiệu chỉnh"] = out["CCF"].abs() > band_adj
    return out, band, band_adj, n


def granger_test(y, x, lag):
    """
    Wald F cho H0: toàn bộ hệ số quá khứ của X = 0 trong hồi quy Y_t ~ Y_{t-1..t-lag} + X_{t-1..t-lag}.
    Sai số chuẩn bền HC1 (sai phân giá có phương sai thay đổi theo cụm).
    Lag dựng trên chuỗi đầy đủ rồi mới bỏ hàng thiếu -> không làm lệch lag khi có NaN.
    """
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1)
    yv, xv, n = d["y"].values, d["x"].values, len(d)
    target = yv[lag:]
    y_lags = np.column_stack([yv[lag - i - 1:n - i - 1] for i in range(lag)])
    x_lags = np.column_stack([xv[lag - i - 1:n - i - 1] for i in range(lag)])
    Z = np.hstack([np.ones((len(target), 1)), y_lags, x_lags])
    ok = np.isfinite(target) & np.isfinite(Z).all(axis=1)
    target, Z = target[ok], Z[ok]
    m, k = Z.shape
    if m < 5 * k or np.linalg.matrix_rank(Z) < k:
        return np.nan, np.nan, m
    beta = np.linalg.lstsq(Z, target, rcond=None)[0]
    e = target - Z @ beta
    bread = np.linalg.inv(Z.T @ Z)
    Ze = Z * e[:, None]
    V = bread @ (Ze.T @ Ze) @ bread * m / (m - k)
    sl = slice(1 + lag, 1 + 2 * lag)
    b, Vb = beta[sl], V[sl, sl]
    F = float(b @ np.linalg.pinv(Vb) @ b) / lag
    return F, float(stats.f.sf(F, lag, m - k)), m


# ============================================================
# SIDEBAR + HEADER
# ============================================================
with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    if st.session_state.get("raw_data") is not None:
        _d = st.session_state["raw_data"]
        st.success(f"Dữ liệu: {_d.shape[0]:,} × {_d.shape[1]}")
    if st.button("← Về Trang 1", key="sb_back", **STRETCH):
        st.switch_page("pages/1_Tai_du_lieu.py")

st.caption(APP_HEADER)
st.title("Trang 3 — Phân tích tương quan & sàng lọc biến X")
st.markdown(
    """
Trang này trả lời 4 câu hỏi rồi gom thành danh sách biến X nên giữ:

1. Các tỉnh có đủ giống nhau để dùng **một chuỗi giá đại diện** không? (Y–Y)
2. Những biến X nào **trùng thông tin** nhau? (X–X, VIF)
3. Biến X **dẫn trước giá** bao nhiêu kỳ? (CCF)
4. Quá khứ của X có **giúp dự báo** giá không? (Granger)

Mọi phép tính dùng **sai phân bậc 1** và chỉ dùng đoạn **TRAIN** để tránh rò rỉ thông tin.
"""
)

# ============================================================
# BƯỚC 0 — NẠP DỮ LIỆU
# ============================================================
st.divider()
st.subheader("0. Nạp dữ liệu")
df = st.session_state.get("raw_data")
if df is not None:
    st.success(f"Dùng dữ liệu từ Trang 1: `{st.session_state.get('data_file_name') or 'raw_data'}` "
               f"({df.shape[0]:,} dòng × {df.shape[1]} cột).")
else:
    st.info("Chưa có dữ liệu từ Trang 1. Bạn có thể upload CSV hoặc dùng dữ liệu demo.")
    source = st.radio("Nguồn dữ liệu", ["Dữ liệu demo", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        f = st.file_uploader("Chọn file CSV (có cột ngày)", type="csv")
        if f is None:
            st.stop()
        df = pd.read_csv(f)
    else:
        df = demo_data()

# ============================================================
# BƯỚC 1 — CHỌN CỘT & THAM SỐ
# ============================================================
st.subheader("1. Chọn cột và tham số")
group_cols = [c for c in (PROVINCE_COL, COFFEE_TYPE_COL) if c in df.columns]
date_default = st.session_state.get("date_col", "date")
date_default = date_default if date_default in df.columns else df.columns[0]
num_cols = [c for c in df.select_dtypes("number").columns if c not in group_cols and c != date_default]
if len(num_cols) < 2:
    st.error("Cần ít nhất 2 cột số (1 biến giá Y và 1 biến X).")
    st.stop()


def _is_price(name):
    toks = set(re.sub(r"[^0-9a-zà-ỹ]+", " ", str(name).lower()).split())
    return bool(toks & {"price", "gia", "giá", "target"}) or any(t.startswith("price") for t in toks)


price_like = [c for c in num_cols if _is_price(c)]
c1, c2, c3 = st.columns(3)
date_col = c1.selectbox("Cột ngày", list(df.columns), index=list(df.columns).index(date_default))
y_col = c2.selectbox("Biến mục tiêu Y (giá)", num_cols,
                     index=num_cols.index(price_like[0]) if price_like else 0)
if COFFEE_TYPE_COL in df.columns:
    types = sorted(df[COFFEE_TYPE_COL].dropna().astype(str).unique())
    ctype = c3.selectbox("Loại cà phê", types + [ALL_TYPES],
                         help="Mỗi loại có mức giá khác nhau; nên phân tích riêng từng loại.")
else:
    ctype = ALL_TYPES
    c3.caption("Dữ liệu không có cột loại cà phê.")
x_options = [c for c in num_cols if c != y_col]
x_cols = st.multiselect("Biến ngoại sinh X", x_options, default=x_options)
if not x_cols:
    st.warning("Chọn ít nhất 1 biến X.")
    st.stop()

p1, p2, p3 = st.columns(3)
train_frac = p1.slider("Tỷ lệ dữ liệu TRAIN", 0.5, 0.9, 0.7, 0.05,
                       help="Chỉ dùng đoạn đầu để chọn biến, tránh rò rỉ từ VAL/TEST.")
corr_method = p2.radio("Kiểu tương quan", ["pearson", "spearman"], horizontal=True,
                       help="Spearman bền hơn khi sai phân giá có nhiều giá trị cực đoan.")
alpha = p3.number_input("Mức ý nghĩa α", 0.001, 0.2, 0.05, 0.005)

# ============================================================
# BƯỚC 2 — CHUẨN BỊ CHUỖI
# ============================================================
d = df.copy()
d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
d = d.dropna(subset=[date_col])
if ctype != ALL_TYPES:
    d = d[d[COFFEE_TYPE_COL].astype(str) == ctype]
if d.empty:
    st.error("Không còn dòng nào sau khi lọc.")
    st.stop()

unit = {"Daily": "ngày", "Weekly": "tuần", "Monthly": "tháng"}.get(st.session_state.get("data_frequency"), "kỳ")

y_series = d.groupby(date_col)[y_col].mean().sort_index()
X_all = d.groupby(date_col)[x_cols].mean().sort_index()
n_train = int(len(y_series) * train_frac)
if n_train < 60:
    st.error(f"Chỉ có {n_train} quan sát TRAIN (< 60): quá ít để phân tích tương quan/lag.")
    st.stop()

y_train, X_train = y_series.iloc[:n_train], X_all.iloc[:n_train]
dy, dX = y_train.diff(), X_train.diff()

# Bỏ X có sai phân hằng số (không tính được tương quan)
const_cols = [c for c in dX.columns if not (dX[c].std() > 0)]
if const_cols:
    st.warning("Đã loại biến X có sai phân hằng số trong đoạn TRAIN: " + ", ".join(const_cols))
    dX = dX.drop(columns=const_cols)
    X_train = X_train.drop(columns=const_cols)
if dX.empty:
    st.error("Không còn biến X hợp lệ.")
    st.stop()
X_names = list(dX.columns)

zero_share = {c: 100 * float((dX[c].dropna() == 0).mean()) for c in X_names}
stuck = [c for c, v in zero_share.items() if v >= ZERO_DIFF_FLAG]
if stuck:
    st.warning("Biến X có ≥ %.0f%% sai phân bằng 0 (nhiều khả năng tần suất thấp hơn bảng, ví dụ dữ liệu tháng "
               "lặp xuống ngày): %s. CCF/Granger với các biến này kém nhạy; xem mục C.6 ở Trang 1."
               % (ZERO_DIFF_FLAG, ", ".join(stuck)))

st.success(f"TRAIN: {n_train:,} / {len(y_series):,} {unit} · {len(X_names)} biến X · "
           f"Y = `{y_col}` · Loại: {ctype}")
st.caption("Y và X được lấy trung bình theo ngày trên các tỉnh (và loại, nếu chọn tất cả). "
           f"Lag tính theo kỳ quan sát (1 kỳ ≈ 1 {unit}).")

# ============================================================
# BƯỚC 3 — BỐN TAB PHÂN TÍCH
# ============================================================
st.divider()
st.subheader("2. Phân tích")
tab1, tab2, tab3, tab4 = st.tabs(["1 · Y–Y (các tỉnh)", "2 · X–X & VIF", "3 · CCF (lag)", "4 · Granger"])

# ---------- Tab 1 ----------
with tab1:
    st.markdown("**Câu hỏi:** có thể thay các tỉnh bằng 1 chuỗi giá đại diện (trung bình) không?")
    if PROVINCE_COL not in d.columns:
        st.info("Dữ liệu không có cột tỉnh nên không cần rút gọn.")
    else:
        wide = (d.pivot_table(index=date_col, columns=PROVINCE_COL, values=y_col, aggfunc="mean")
                .sort_index().iloc[:n_train].dropna(axis=1, how="all"))
        if wide.shape[1] < 2:
            st.info("Chỉ có 1 tỉnh.")
        else:
            c_diff = wide.diff().corr(method=corr_method)
            st.plotly_chart(corr_heatmap(c_diff, "Tương quan giữa các tỉnh (sai phân)"), **STRETCH)
            off = c_diff.where(~np.eye(len(c_diff), dtype=bool))
            weakest = float(off.min().min())
            avg = float(np.nanmean(off.values))
            m1, m2 = st.columns(2)
            m1.metric("ρ trung bình giữa các cặp tỉnh", f"{avg:.2f}")
            m2.metric("ρ của cặp yếu nhất", f"{weakest:.2f}")
            if weakest >= MIN_RHO_Y:
                st.success(f"**Kết luận:** cặp yếu nhất vẫn có ρ = {weakest:.2f} ≥ {MIN_RHO_Y} → "
                           "dùng 1 chuỗi đại diện (trung bình các tỉnh) là hợp lý.")
            else:
                i, j = np.unravel_index(np.nanargmin(off.values), off.shape)
                st.warning(f"**Kết luận:** cặp {off.index[i]} – {off.columns[j]} chỉ có ρ = {weakest:.2f} < {MIN_RHO_Y} "
                           "→ các tỉnh khác nhau đáng kể, cân nhắc giữ riêng hoặc loại tỉnh lệch.")
            st.caption(f"Ngưỡng {MIN_RHO_Y} là tham số ứng dụng, không phải chuẩn học thuật.")

# ---------- Tab 2 ----------
with tab2:
    st.markdown("**Câu hỏi:** những biến X nào mang thông tin giống nhau?")
    vif = vif_table(dX)
    vif_map = dict(zip(vif["Biến X"], vif["VIF"]))
    if len(X_names) < 2:
        st.info("Cần ít nhất 2 biến X.")
    else:
        cmat = dX.corr(method=corr_method)
        st.plotly_chart(corr_heatmap(cmat, "Tương quan X–X (sai phân)"), **STRETCH)
        pairs = [(a, b, cmat.loc[a, b]) for i, a in enumerate(X_names) for b in X_names[i + 1:]
                 if abs(cmat.loc[a, b]) >= HIGH_CORR_X]
        if pairs:
            st.warning(f"**Cặp X trùng thông tin (|ρ| ≥ {HIGH_CORR_X}):** "
                       + "; ".join(f"{a} – {b} (ρ = {r:.2f})" for a, b, r in pairs)
                       + ". Nên giữ 1 biến trong mỗi cặp.")
        else:
            st.success(f"Không có cặp X nào có |ρ| ≥ {HIGH_CORR_X}.")
        st.dataframe(vif.style.format({"VIF": "{:.2f}"}), hide_index=True, **STRETCH)
        n_high = int((vif["VIF"] >= VIF_STRONG).sum())
        if n_high:
            st.warning(f"**Kết luận:** {n_high} biến có VIF ≥ {VIF_STRONG:.0f} → đa cộng tuyến mạnh, "
                       "nên bỏ bớt hoặc gom nhóm.")
        else:
            st.success(f"**Kết luận:** không biến nào có VIF ≥ {VIF_STRONG:.0f} → đa cộng tuyến không đáng lo.")
        st.caption(f"VIF ≥ {VIF_WARN:.0f}: cảnh báo nhẹ; ≥ {VIF_STRONG:.0f}: cảnh báo mạnh (quy tắc kinh nghiệm). "
                   "VIF tính từ ma trận tương quan của sai phân TRAIN.")

# ---------- Tab 3 ----------
with tab3:
    st.markdown("**Câu hỏi:** biến X dẫn trước Y bao nhiêu kỳ? (k > 0: X dẫn trước)")
    default_L = int(min(40, max(5, np.floor(10 * np.log10(max(n_train, 10))))))
    max_lag = st.slider("Số lag tối đa", 5, 60, default_L)
    ccf_res, ccf_best = {}, {}
    for x in X_names:
        tbl, band, band_adj, n_eff = ccf_table(dy, dX[x], max_lag, alpha)
        ccf_res[x] = (tbl, band, band_adj, n_eff)
        if tbl is None:
            ccf_best[x] = None
            continue
        sig = tbl[(tbl["Lag k"] > 0) & tbl["Vượt ngưỡng hiệu chỉnh"]]
        if sig.empty:
            ccf_best[x] = {"lag": None, "r": np.nan, "n_sig": 0}
        else:
            top = sig.loc[sig["CCF"].abs().idxmax()]
            ccf_best[x] = {"lag": int(top["Lag k"]), "r": float(top["CCF"]), "n_sig": len(sig)}

    summ = pd.DataFrame([{
        "Biến X": x,
        "Lag mạnh nhất": ("—" if (ccf_best[x] is None or ccf_best[x]["lag"] is None) else ccf_best[x]["lag"]),
        "CCF tại lag đó": (np.nan if ccf_best[x] is None else ccf_best[x]["r"]),
        "Số lag vượt ngưỡng hiệu chỉnh": ("không đủ mẫu" if ccf_best[x] is None else ccf_best[x]["n_sig"]),
    } for x in X_names])
    st.markdown("**Tóm tắt mọi biến X** (chỉ tính lag k > 0, ngưỡng đã hiệu chỉnh theo số lag)")
    st.dataframe(summ.style.format({"CCF tại lag đó": "{:.3f}"}), hide_index=True, **STRETCH)

    chosen_x = st.selectbox("Xem biểu đồ CCF của biến", X_names)
    tbl, band, band_adj, n_eff = ccf_res[chosen_x]
    if tbl is None:
        st.info(f"Không đủ quan sát liên tục (cần ≥ {MIN_OBS + max_lag}) hoặc chuỗi sau lọc là hằng số.")
    else:
        colors = ["#6366f1" if s else "#94a3b8" for s in tbl["Vượt ngưỡng hiệu chỉnh"]]
        fig = go.Figure(go.Bar(x=tbl["Lag k"], y=tbl["CCF"], marker_color=colors))
        for s in (band, -band):
            fig.add_hline(y=s, line_dash="dash", line_color="#f59e0b")
        for s in (band_adj, -band_adj):
            fig.add_hline(y=s, line_dash="dot", line_color="#dc2626")
        fig.update_layout(title=f"CCF: {chosen_x} → {y_col}", xaxis_title="Lag k (kỳ)",
                          yaxis_title="Tương quan", height=380, showlegend=False)
        st.plotly_chart(fig, **STRETCH)
        b = ccf_best[chosen_x]
        if b and b["lag"] is not None:
            st.success(f"**Kết luận:** {chosen_x} dẫn trước {y_col} rõ nhất tại lag {b['lag']} (CCF = {b['r']:.3f}); "
                       f"có {b['n_sig']} lag vượt ngưỡng hiệu chỉnh.")
        else:
            st.info("**Kết luận:** không có lag k > 0 nào vượt ngưỡng hiệu chỉnh.")
        st.caption(f"Cam (nét đứt) = ngưỡng từng lag ±{band:.3f}; đỏ (chấm) = ngưỡng hiệu chỉnh Bonferroni "
                   f"theo {max_lag} lag ±{band_adj:.3f}; n = {n_eff:,}. Cột tím là lag vượt ngưỡng đỏ. "
                   "X và Y được lọc bằng cùng bộ AR (tiền trắng hóa) để tránh tương quan giả.")

# ---------- Tab 4 ----------
with tab4:
    st.markdown("**Câu hỏi:** sau khi đã biết quá khứ của Y, quá khứ của X có giúp dự báo Y thêm không?")
    g_lag = st.slider("Số lag dùng trong kiểm định", 1, 30, 5)
    rows = []
    for x in X_names:
        F, p, n_used = granger_test(dy, dX[x], g_lag)
        _, p_rev, _ = granger_test(dX[x], dy, g_lag)
        rows.append({"Biến X": x, "F": F, "p (X→Y)": p, "N dùng": n_used, "p ngược (Y→X)": p_rev})
    g = pd.DataFrame(rows)
    n_tests = int(g["p (X→Y)"].notna().sum())
    g["p hiệu chỉnh"] = np.minimum(g["p (X→Y)"] * max(n_tests, 1), 1.0)
    g["Kết quả"] = np.where(g["p (X→Y)"].isna(), "không đủ mẫu",
                            np.where(g["p hiệu chỉnh"] < alpha, "giữ", "chưa đủ bằng chứng"))
    granger_ok = dict(zip(g["Biến X"], g["Kết quả"] == "giữ"))
    granger_p = dict(zip(g["Biến X"], g["p hiệu chỉnh"]))
    st.dataframe(g.style.format({"F": "{:.3f}", "p (X→Y)": "{:.4g}", "p ngược (Y→X)": "{:.4g}",
                                 "p hiệu chỉnh": "{:.4g}"}), hide_index=True, **STRETCH)
    kept = g.loc[g["Kết quả"] == "giữ", "Biến X"].tolist()
    if kept:
        st.success("**Kết luận:** quá khứ của " + ", ".join(kept) + " giúp dự báo Y (đã hiệu chỉnh Bonferroni).")
    else:
        st.info("**Kết luận:** chưa biến X nào có bằng chứng Granger đủ mạnh.")
    feedback = g.loc[(g["p ngược (Y→X)"] < alpha) & g["p ngược (Y→X)"].notna(), "Biến X"].tolist()
    if feedback:
        st.caption("Lưu ý: giá Y cũng giúp dự báo " + ", ".join(feedback) +
                   " (phản hồi hai chiều, p ngược chưa hiệu chỉnh) → các biến này có thể không hoàn toàn ngoại sinh.")
    st.caption("Wald F với sai số chuẩn bền HC1 trên sai phân. Granger chỉ đo quan hệ dự báo tuyến tính, "
               "KHÔNG phải quan hệ nhân quả; không bác bỏ H0 chưa đủ để loại biến.")

# ============================================================
# BƯỚC 4 — SHORTLIST
# ============================================================
st.divider()
st.subheader("3. Shortlist biến X")
short = pd.DataFrame({
    "Biến X": X_names,
    "Lag CCF mạnh nhất": [("—" if (ccf_best[x] is None or ccf_best[x]["lag"] is None) else ccf_best[x]["lag"])
                          for x in X_names],
    "CCF có ý nghĩa": [bool(ccf_best[x] and ccf_best[x]["lag"] is not None) for x in X_names],
    "Granger giữ": [bool(granger_ok.get(x, False)) for x in X_names],
    "p Granger (hiệu chỉnh)": [granger_p.get(x, np.nan) for x in X_names],
    "VIF": [vif_map.get(x, np.nan) for x in X_names],
})
short["Số tiêu chí (0–2)"] = short["CCF có ý nghĩa"].astype(int) + short["Granger giữ"].astype(int)
short["Cảnh báo VIF"] = np.where(short["VIF"] >= VIF_STRONG, "VIF cao: cân nhắc bỏ/gom nhóm", "")
short = short.sort_values(["Số tiêu chí (0–2)", "VIF"], ascending=[False, True]).reset_index(drop=True)
st.dataframe(short.style.format({"p Granger (hiệu chỉnh)": "{:.4g}", "VIF": "{:.2f}"}), hide_index=True, **STRETCH)
st.download_button("⬇ Tải shortlist (CSV)", short.to_csv(index=False).encode("utf-8-sig"),
                   file_name="shortlist_bien_X.csv", mime="text/csv")

default_keep = short.loc[short["Số tiêu chí (0–2)"] >= 1, "Biến X"].tolist() or X_names
keep = st.multiselect("Biến X giữ lại cho bước sau", X_names, default=default_keep,
                      help="Mặc định giữ biến thỏa ≥ 1 tiêu chí. Đây là bước sàng lọc, bạn có thể chỉnh tay.")
if st.button("Lưu lựa chọn", type="primary"):
    st.session_state["selected_vars"] = [y_col] + keep
    st.session_state["corr_results"] = {
        "y": y_col, "coffee_type": ctype, "train_frac": float(train_frac), "alpha": float(alpha),
        "corr_method": corr_method, "max_lag": int(max_lag), "granger_lag": int(g_lag),
        "shortlist": short.to_dict(orient="records"), "kept": keep,
    }
    st.success(f"Đã lưu: Y = {y_col}, X = {', '.join(keep) if keep else '(không có)'}")

st.divider()
if st.button("← Về Trang 1", key="nav_back", **STRETCH):
    st.switch_page("pages/1_Tai_du_lieu.py")
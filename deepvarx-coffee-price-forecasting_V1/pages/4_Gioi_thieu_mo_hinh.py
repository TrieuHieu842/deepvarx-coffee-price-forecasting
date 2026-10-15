# -*- coding: utf-8 -*-
"""
DeepVARX — Trang 7–12: Nhóm mô hình (VAR, VARX, VARX-L, LSTM, DeepVAR, DeepVARX)
Trang mô hình chỉ GIỚI THIỆU và TÍNH SỐ THAM SỐ; đánh giá hiệu năng nằm ở Trang 16.
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
TARGET_COL = "coffee_price_vnd_kg"
COFFEE_TYPE_COL = "coffee_type"

st.set_page_config(page_title="Giới thiệu mô hình", page_icon="☕", layout="wide")


def _streamlit_version():
    return tuple(int(re.sub(r"\D", "", p) or 0) for p in st.__version__.split(".")[:2])


STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


# ============================================================
# HELPERS — SỐ THAM SỐ
# ============================================================
def linear_params(K, M, p, q, jmin):
    """Mỗi phương trình c = 1 + K·p + M·(q − jmin + 1); tổng = K·c. jmin=0 → M(q+1)."""
    n_x_lags = max(q - jmin + 1, 0) if M > 0 else 0
    c = 1 + K * p + M * n_x_lags
    return c, K * c


def lstm_params(n_in, n_h, layers):
    """PyTorch: lớp 1 = 4·n_h·(n_in + n_h + 2); lớp sau n_in = n_h."""
    per = [4 * n_h * ((n_in if l == 0 else n_h) + n_h + 2) for l in range(layers)]
    return sum(per), per


def dense(n_in, n_out):
    return n_in * n_out + n_out


def cov_params(K, r):
    return K * (K + 1) // 2, K * r + K   # (ma trận đầy đủ, hạng thấp r + đường chéo)


def dist_head(n_h, K, kind, r):
    """Đầu ra phân phối N(μ, Σ): μ (K) + tham số Σ."""
    n_sigma = K * (K + 1) // 2 if kind == "Cholesky đầy đủ" else K * r + K
    return dense(n_h, K + n_sigma)


# ============================================================
# HELPERS — VAR / VARX BẰNG NUMPY
# ============================================================
def lagblock(a, L, start):
    a = np.asarray(a, float).reshape(len(a), -1)
    return np.hstack([a[start - i: len(a) - i] for i in range(1, L + 1)])


def var_design(Y, p, X=None, q=0, jmin=1, start=None):
    T = len(Y)
    s = start if start is not None else max(p, q if X is not None else 0)
    parts = [np.ones((T - s, 1)), lagblock(Y, p, s)]
    if X is not None and q >= jmin:
        for j in range(jmin, q + 1):
            parts.append(X[s - j:T - j])
    return np.hstack(parts), Y[s:]


def ols(Z, Yt):
    B = np.linalg.lstsq(Z, Yt, rcond=None)[0]
    return B, Yt - Z @ B


def info_criteria(Z, E):
    Te, K = E.shape
    sign, ld = np.linalg.slogdet(E.T @ E / Te)
    c = Z.shape[1]
    return {"logdet": ld, "c": c, "Te": Te,
            "AIC": ld + 2 * K * c / Te,
            "BIC": ld + np.log(Te) * K * c / Te,
            "HQ": ld + 2 * np.log(np.log(Te)) * K * c / Te}


def coef_matrices(B, K, p):
    return [B[1 + (i - 1) * K: 1 + i * K, :].T for i in range(1, p + 1)]


def companion_modulus(A):
    K, p = A[0].shape[0], len(A)
    F = np.zeros((K * p, K * p))
    F[:K, :] = np.hstack(A)
    if p > 1:
        F[K:, :-K] = np.eye(K * (p - 1))
    return float(np.max(np.abs(np.linalg.eigvals(F))))


def irf_from(A, Sigma, H):
    """IRF trực giao hóa (Cholesky; kết quả phụ thuộc thứ tự biến)."""
    K = Sigma.shape[0]
    P = np.linalg.cholesky(Sigma + 1e-12 * np.eye(K))
    Phi = [np.eye(K)]
    for s in range(1, H + 1):
        Phi.append(sum(Phi[s - i] @ A[i - 1] for i in range(1, min(s, len(A)) + 1)))
    return np.array([ph @ P for ph in Phi])  # (H+1, K_response, K_shock)


def fevd(Theta, h):
    sq = (Theta[:h] ** 2).sum(0)
    return sq / sq.sum(1, keepdims=True)


def bootstrap_irf(Y, p, H, nboot, seed=0):
    Z, Yt = var_design(Y, p)
    B, E = ols(Z, Yt)
    K, T = Y.shape[1], len(Y)
    A, c = coef_matrices(B, K, p), B[0]
    Ec = E - E.mean(0)
    rng = np.random.default_rng(seed)
    Eb = Ec[rng.integers(0, len(Ec), size=(nboot, T))]
    Ys = np.zeros((nboot, T, K))
    Ys[:, :p, :] = Y[:p]
    for t in range(p, T):
        acc = c + Eb[:, t, :]
        for i in range(1, p + 1):
            acc = acc + Ys[:, t - i, :] @ A[i - 1].T
        Ys[:, t, :] = acc
    out = []
    for b in range(nboot):
        Zb, Ytb = var_design(Ys[b], p)
        Bb, Eb_ = ols(Zb, Ytb)
        out.append(irf_from(coef_matrices(Bb, K, p), Eb_.T @ Eb_ / len(Eb_), H))
    return np.array(out)


def penalty_groups(K, p, kind):
    """Nhóm phạt cho các ma trận A_1..A_p (hàng = phương trình, cột = lag × biến). Chỉ để minh họa cấu trúc."""
    G = np.zeros((K, K * p), dtype=int)
    gid = 0
    if kind in ("Basic (lasso theo phần tử)",):
        for a in range(K):
            for c in range(K * p):
                gid += 1
                G[a, c] = gid
    elif kind in ("Lag Group", "Sparse Lag Group"):
        for l in range(p):
            gid += 1
            G[:, l * K:(l + 1) * K] = gid
    else:  # Own/Other
        for l in range(p):
            gid += 1
            own = np.eye(K, dtype=bool)
            G[:, l * K:(l + 1) * K][own] = gid
            gid += 1
            G[:, l * K:(l + 1) * K][~own] = gid
    return G


# ============================================================
# HELPERS — DỮ LIỆU CHO PHẦN BẰNG CHỨNG
# ============================================================
def get_data_for_evidence():
    df = st.session_state.get("processed_data")
    if df is None:
        df = st.session_state.get("raw_data")
    cs = st.session_state.get("corr_settings") or {}
    if df is None:
        return None
    date_col = st.session_state.get("date_col", "date")
    ycol = cs.get("y") or (TARGET_COL if TARGET_COL in df.columns else None)
    if date_col not in df.columns or ycol is None or ycol not in df.columns:
        return None
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if COFFEE_TYPE_COL in d.columns:
        Y = d.pivot_table(index=date_col, columns=COFFEE_TYPE_COL, values=ycol, aggfunc="mean")
    else:
        Y = d.groupby(date_col)[ycol].mean().to_frame(ycol)
    Y = Y.sort_index()
    sel = [c for c in (st.session_state.get("corr_selected_vars") or []) if c in d.columns and c != ycol]
    X = d.groupby(date_col)[sel].mean().sort_index().reindex(Y.index) if sel else pd.DataFrame(index=Y.index)
    last = cs.get("last_train_date")
    if last:
        Y, X = Y.loc[:last], X.loc[:last]
    else:
        n = int(len(Y) * 0.7)
        Y, X = Y.iloc[:n], X.iloc[:n]
    return {"Y": Y, "X": X, "ycol": ycol, "train_defined": bool(last)}


def fig_irf(irf, lo, hi, names, cumulative):
    K = irf.shape[1]
    fig = make_subplots(rows=K, cols=K, subplot_titles=[f"{names[b]} → {names[a]}" for a in range(K) for b in range(K)])
    hh = np.arange(irf.shape[0])
    for a in range(K):
        for b in range(K):
            f = (lambda v: np.cumsum(v)) if cumulative else (lambda v: v)
            fig.add_trace(go.Scatter(x=hh, y=f(irf[:, a, b]), mode="lines", line=dict(color="#4f46e5"),
                                     showlegend=False), row=a + 1, col=b + 1)
            if lo is not None:
                fig.add_trace(go.Scatter(x=np.r_[hh, hh[::-1]], y=np.r_[f(hi[:, a, b]), f(lo[:, a, b])[::-1]],
                                         fill="toself", fillcolor="rgba(99,102,241,0.18)", line=dict(width=0),
                                         showlegend=False, hoverinfo="skip"), row=a + 1, col=b + 1)
    fig.update_layout(height=260 * K, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ============================================================
# SIDEBAR + HEADER
# ============================================================
with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()

st.caption(APP_HEADER)
st.title("Trang 7–12 — Nhóm mô hình")
st.markdown("Giới thiệu và **tính số tham số** của 6 mô hình: VAR, VARX, VARX-L, LSTM, DeepVAR, DeepVARX. "
            "Đánh giá hiệu năng nằm ở **Trang 16**, không nằm ở đây.")

# ── Thông số dùng chung (mặc định lấy từ các trang trước nếu có) ──
cs = st.session_state.get("corr_settings") or {}
sel_vars = st.session_state.get("corr_selected_vars") or []
st.divider()
st.subheader("Thông số dùng chung")
g1, g2, g3, g4, g5, g6 = st.columns(6)
K = g1.number_input("K (chuỗi nội sinh)", 1, 100, 2)
M = g2.number_input("M (biến ngoại sinh)", 0, 500, max(len(sel_vars) - 1, 0) or 10)
p = g3.number_input("p (lookback nội sinh)", 1, 200, 30, help="Độ dài lookback ví dụ của giao diện.")
q = g4.number_input("q (lookback ngoại sinh)", 0, 200, 30)
T_train = g5.number_input("T_train", 50, 1_000_000, int(cs.get("T_train", 1000)))
jmin = g6.radio("Chỉ số j của X bắt đầu từ", [1, 0], horizontal=True,
                help="j = 0 nghĩa là X_t đồng thời tham gia dự báo Y_t.")
if jmin == 0:
    st.warning("j = 0: khi dự báo trước h bước, X_{T+h} chưa biết. Chỉ hợp lệ nếu X là known-future hoặc có dự báo "
               "riêng cho X; nếu không phải bắt đầu từ j = 1 (không được âm thầm đưa X tương lai thật vào đầu vào).")
n_eff = T_train - max(p, q)
c_lin, tot_lin = linear_params(K, M, p, q, jmin)
st.caption(f"VARX: c = 1 + K·p + M·(q − j_min + 1) = **{c_lin:,}** hệ số/phương trình, tổng **{tot_lin:,}**; "
           f"N_eff = T_train − max(p,q) = **{n_eff:,}**; tỷ lệ c/N_eff = **{(c_lin / n_eff) if n_eff > 0 else float('inf'):.3f}** (Trang 4.7).")

tabs = st.tabs(["Trang 7 · VAR", "Trang 8 · VARX", "Trang 9 · VARX-L", "Trang 10 · LSTM",
                "Trang 11 · DeepVAR", "Trang 12 · DeepVARX"])

ev = get_data_for_evidence()


def need_data_note():
    st.info("Cần dữ liệu và Y đã chọn ở Trang 4 để tính phần bằng chứng này.")


# ============================================================
# TRANG 7 — VAR
# ============================================================
with tabs[0]:
    st.subheader("Trang 7 — VAR (Vector AutoRegression)")
    cA, cB = st.columns([3, 2])
    with cA:
        st.markdown("**Mục tiêu.** Mô hình hóa quan hệ động học tuyến tính giữa nhiều chuỗi giá; "
                    "là baseline đa biến tuyến tính.")
        st.latex(r"\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p}\mathbf{A}_i\mathbf{Y}_{t-i} + \boldsymbol{\varepsilon}_t")
        st.markdown("**Tham số.** $p$: bậc trễ; $K$: số chuỗi nội sinh; $\\mathbf{A}_i\\in\\mathbb{R}^{K\\times K}$. "
                    "Điều kiện: các chuỗi dừng (hoặc đã sai phân); không có biến ngoại sinh.")
        st.markdown("**Số tham số.** $K + K^2 p$ (mỗi phương trình $1 + Kp$).")
    with cB:
        st.code("Y(t−p) … Y(t−1)\n      │  hồi quy tuyến tính\n      ▼\n     Y(t)", language="text")
    st.metric("Tổng số tham số VAR(p)", f"{K + K * K * p:,}")

    st.markdown("**Bằng chứng hiển thị**")
    if ev is None:
        need_data_note()
    else:
        Yd = ev["Y"].diff().dropna()
        Yd = Yd.loc[:, Yd.notna().all()]
        if Yd.shape[0] < 200 or Yd.shape[1] < 1:
            st.info("Không đủ quan sát để ước lượng VAR.")
        else:
            if not ev["train_defined"]:
                st.caption("Chưa có đoạn TRAIN từ Trang 4: tạm dùng 70% đầu của dữ liệu.")
            Yv = Yd.to_numpy(float)
            Ke = Yv.shape[1]
            pmax = st.slider("p tối đa để quét tiêu chí thông tin", 1, 20, min(10, max(2, Yv.shape[0] // 50)), key="var_pmax")
            rows = []
            for pp in range(1, pmax + 1):
                Z, Yt = var_design(Yv, pp, start=pmax)
                B, E = ols(Z, Yt)
                ic = info_criteria(Z, E)
                rows.append({"p": pp, "AIC": ic["AIC"], "BIC": ic["BIC"], "HQ": ic["HQ"]})
            ic_df = pd.DataFrame(rows)
            fig = go.Figure()
            for crit in ["AIC", "BIC", "HQ"]:
                fig.add_trace(go.Scatter(x=ic_df["p"], y=ic_df[crit], mode="lines+markers", name=crit))
            fig.update_layout(title="Tiêu chí thông tin theo p (VAR trên sai phân)", xaxis_title="p",
                              height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, **STRETCH)
            best = {c: int(ic_df.loc[ic_df[c].idxmin(), "p"]) for c in ["AIC", "BIC", "HQ"]}
            st.caption(f"p tối ưu: AIC = {best['AIC']}, BIC = {best['BIC']}, HQ = {best['HQ']} "
                       "(đường cong chính thức ở Trang 5.4).")

            p_sel = st.number_input("p dùng cho kiểm tra ổn định / IRF / FEVD", 1, pmax, best["BIC"], key="var_psel")
            Z, Yt = var_design(Yv, int(p_sel))
            B, E = ols(Z, Yt)
            A = coef_matrices(B, Ke, int(p_sel))
            mod = companion_modulus(A)
            Sigma = E.T @ E / len(E)
            m1, m2, m3 = st.columns(3)
            m1.metric("Mô đun lớn nhất của nghiệm riêng (ma trận đồng hành)", f"{mod:.4f}")
            m2.metric("Tham số c / N_eff", f"{Z.shape[1]} / {len(Yt):,} = {Z.shape[1] / len(Yt):.4f}")
            m3.metric("Ổn định?", "Có (< 1)" if mod < 1 else "KHÔNG (≥ 1)")
            if mod >= 1:
                st.error("Mô đun ≥ 1: VAR không ổn định; IRF/FEVD bootstrap bị bỏ qua.")
            else:
                H = st.slider("Horizon IRF/FEVD", 5, 60, 20, key="var_H")
                names = list(Yd.columns)
                irf = irf_from(A, Sigma, H)
                cum = st.checkbox("IRF tích lũy (phản ứng của mức giá)", value=False)
                nb = st.number_input("Số lần bootstrap (phần dư)", 50, 1000, 100, 50)
                if st.button("Tính IRF + khoảng tin cậy bootstrap 95%", key="var_boot"):
                    with st.spinner("Đang bootstrap..."):
                        bs = bootstrap_irf(Yv, int(p_sel), H, int(nb))
                    tr = np.cumsum(bs, axis=1) if cum else bs
                    st.session_state["var_irf_boot"] = (np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0))
                bt = st.session_state.get("var_irf_boot")
                lo, hi = (bt if bt and bt[0].shape == irf.shape else (None, None))
                st.plotly_chart(fig_irf(irf, lo, hi, names, cum), **STRETCH)
                st.caption("IRF trực giao hóa bằng Cholesky: kết quả phụ thuộc thứ tự các chuỗi Y. "
                           "Mô hình chạy trên sai phân nên phản ứng là của thay đổi giá; tích lũy để xem mức giá.")
                fv = pd.DataFrame(fevd(irf, H), index=[f"Phản ứng: {n}" for n in names],
                                  columns=[f"Cú sốc {n}" for n in names])
                st.markdown(f"**FEVD tại horizon {H}**")
                st.dataframe(fv.style.format("{:.3f}"), **STRETCH)
    st.markdown("**Vai trò.** Baseline đa biến tuyến tính. Baseline đầu tiên của quy trình là Naive (Trang 14), không phải VAR.")
    st.markdown("**Tài liệu.** Sims (1980); Lütkepohl (2005); Yashavanth et al. (2017) dùng VAR cho giá cà phê "
                "(ADF kiểm tra dừng, RMSE và MAPE để so sánh). VAR được đề xuất làm công cụ không áp đặt cấu trúc "
                "lý thuyết lên quan hệ giữa các biến (Sims, 1980).")

# ============================================================
# TRANG 8 — VARX
# ============================================================
with tabs[1]:
    st.subheader("Trang 8 — VARX")
    cA, cB = st.columns([3, 2])
    with cA:
        st.markdown("**Mục tiêu.** Kiểm tra đóng góp của biến ngoại sinh.")
        jlow = "0" if jmin == 0 else "1"
        st.latex(r"\mathbf{Y}_t = \mathbf{c} + \sum_{i=1}^{p}\mathbf{A}_i\mathbf{Y}_{t-i} + \sum_{j=" + jlow
                 + r"}^{q}\mathbf{B}_j\mathbf{X}_{t-j} + \boldsymbol{\varepsilon}_t")
        st.markdown("**Số tham số mỗi phương trình.** $c = 1 + Kp + M(q+1)$ khi $j$ bắt đầu từ 0; "
                    "$c = 1 + Kp + Mq$ khi bắt đầu từ 1. Giao diện đang dùng "
                    f"j từ {jmin}.")
        st.warning("Điều chỉnh: $j=0$ nghĩa là $X_t$ đồng thời tham gia dự báo $Y_t$. Khi dự báo trước $h$ bước, "
                   "$X_{T+h}$ chưa biết; nếu không có giá trị tương lai biết trước (hoặc dự báo riêng cho $X$) "
                   "thì $j$ phải bắt đầu từ 1.")
    with cB:
        st.code("Y quá khứ ─┐\n           ├─ VARX → Y tương lai\nX quá khứ ─┘", language="text")
    a1, a2 = st.columns(2)
    a1.metric("c (hệ số / phương trình)", f"{c_lin:,}")
    a2.metric("Tổng hệ số K·c", f"{tot_lin:,}")

    st.markdown("**Bằng chứng hiển thị**")
    if ev is None or ev["X"].shape[1] == 0:
        st.info("Cần dữ liệu, Y và danh sách X đã lưu ở Trang 4 để so sánh VARX với VAR.")
    else:
        Xall = ev["X"]
        xs = st.multiselect("Biến X đưa vào VARX", list(Xall.columns), default=list(Xall.columns)[:5], key="varx_x")
        pq1, pq2 = st.columns(2)
        pp = pq1.number_input("p", 1, 30, 5, key="varx_p")
        qq = pq2.number_input("q", 1, 30, 5, key="varx_q")
        if xs:
            joint = pd.concat([ev["Y"].diff(), Xall[xs].diff()], axis=1).dropna()
            if len(joint) < 20 * (max(pp, qq) + 2):
                st.info("Không đủ quan sát.")
            else:
                Yv = joint[list(ev["Y"].columns)].to_numpy(float)
                Xv = joint[xs].to_numpy(float)
                s0 = max(int(pp), int(qq))
                Zr, Yt = var_design(Yv, int(pp), start=s0)
                Zu, _ = var_design(Yv, int(pp), Xv, int(qq), jmin, start=s0)
                _, Er = ols(Zr, Yt)
                _, Eu = ols(Zu, Yt)
                icr, icu = info_criteria(Zr, Er), info_criteria(Zu, Eu)
                nres = len(xs) * (int(qq) - jmin + 1) * Yv.shape[1]
                LR = max(0.0, (icu["Te"] - icu["c"]) * (icr["logdet"] - icu["logdet"]))
                pval = float(stats.chi2.sf(LR, nres)) if SCIPY_AVAILABLE else np.nan
                st.dataframe(pd.DataFrame([
                    {"Mô hình": "VAR", "Hệ số/phương trình": icr["c"], "AIC": icr["AIC"], "BIC": icr["BIC"]},
                    {"Mô hình": "VARX", "Hệ số/phương trình": icu["c"], "AIC": icu["AIC"], "BIC": icu["BIC"]},
                ]).style.format({"AIC": "{:.4f}", "BIC": "{:.4f}"}), hide_index=True, **STRETCH)
                b1, b2, b3 = st.columns(3)
                b1.metric("ΔAIC (VARX − VAR)", f"{icu['AIC'] - icr['AIC']:.4f}")
                b2.metric("ΔBIC (VARX − VAR)", f"{icu['BIC'] - icr['BIC']:.4f}")
                b3.metric("Kiểm định khối B_j (LR, p)", f"{pval:.4g}" if not np.isnan(pval) else "n/a")
                st.caption("ΔAIC/ΔBIC < 0: VARX được ưu tiên theo tiêu chí. Kiểm định khối cho toàn bộ B_j dùng LR có hiệu "
                           "chỉnh mẫu nhỏ (cùng kiểu Trang 4.6), tính trên sai phân và đoạn TRAIN.")
    st.markdown("- Chênh lệch sai số ngoài mẫu VARX so với VAR và kiểm định **Clark–West** (VAR là mô hình lồng trong VARX): "
                "hiển thị ở **Trang 16.1**.\n- Sai số chuẩn hệ số dùng ước lượng bền **HAC** (tự tương quan và phương sai đổi): "
                "tính ở bước ước lượng cuối của Trang 15/16, không tính trong trang giới thiệu.")
    st.markdown("**Tài liệu.** Warsono et al. (2019): VARX cho dữ liệu năng lượng; Sudarwanto et al. (2026): VARX cho "
                "doanh thu xuất khẩu Indonesia; Parra et al. (2025): phân tích Granger trong khung VARX.")

# ============================================================
# TRANG 9 — VARX-L
# ============================================================
with tabs[2]:
    st.subheader("Trang 9 — VARX-L (VARX có regularization cấu trúc)")
    st.markdown("**Mục tiêu.** Kiểm soát số tham số và chọn lọc/thu nhỏ các quan hệ không cần thiết của VARX bằng "
                "regularization có cấu trúc.")
    st.latex(r"\min_{\Theta}\; \mathcal{L}(\Theta) + \lambda\,\mathcal{P}(\Theta)")
    st.markdown("$\\mathcal{L}$: hàm mất mát bình phương; $\\mathcal{P}$: hàm phạt có cấu trúc.")

    kinds = ["Basic (lasso theo phần tử)", "Lag Group", "Own/Other", "Sparse Lag Group", "Sparse Own/Other"]
    kind = st.selectbox("Kiểu phạt (Nicholson et al., 2017)", kinds)
    desc = {
        "Basic (lasso theo phần tử)": "Phạt L1 từng hệ số riêng lẻ.",
        "Lag Group": "Nhóm theo độ trễ: có thể loại cả một độ trễ.",
        "Own/Other": "Tách độ trễ của chính chuỗi (đường chéo) và của chuỗi khác.",
        "Sparse Lag Group": "Lag Group cộng thưa trong từng nhóm.",
        "Sparse Own/Other": "Own/Other cộng thưa trong từng nhóm.",
    }
    st.caption(desc[kind])
    Kv, pv = min(int(K), 6), min(int(p), 5)
    G = penalty_groups(Kv, pv, kind)
    fig = go.Figure(go.Heatmap(z=G, colorscale="Turbo", showscale=False))
    fig.update_layout(title=f"Cấu trúc nhóm phạt (minh họa K={Kv}, p={pv}; mỗi màu = một nhóm)",
                      xaxis_title="Lag × biến (A_1 … A_p)", yaxis_title="Phương trình", height=260,
                      margin=dict(l=10, r=10, t=45, b=10))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, **STRETCH)
    st.caption("Đây là sơ đồ cấu trúc nhóm, không phải ma trận thưa ước lượng từ dữ liệu. Ma trận thưa theo từng lag "
               "được hiển thị sau khi huấn luyện (Trang 15/16).")

    st.markdown("**Số hệ số và tỷ lệ thưa**")
    v1, v2, v3 = st.columns(3)
    v1.metric("Tổng hệ số (dày)", f"{tot_lin:,}")
    infeasible = c_lin > n_eff
    v2.metric("N_eff", f"{n_eff:,}")
    v3.metric("c / N_eff", f"{c_lin / n_eff:.3f}" if n_eff > 0 else "∞")
    if infeasible:
        st.error("c > N_eff: ước lượng bình phương nhỏ nhất không khả thi → ghi **'không ước lượng được'** "
                 "cho Θ_OLS, không vẽ cột giả khi so sánh chuẩn Frobenius ‖Θ‖_F trước/sau regularization.")
    st.markdown("Tỷ lệ thưa = số hệ số bằng 0 / tổng hệ số (tính sau khi huấn luyện).")

    st.markdown("**Chọn λ.** Đường λ (path): số hệ số khác 0 theo λ. Chọn λ bằng validation theo thời gian "
                "(rolling-origin), **không** dùng k-fold ngẫu nhiên: điều kiện để k-fold hợp lệ với chuỗi thời gian "
                "chỉ được chứng minh cho mô hình tự hồi quy thuần với sai số không tự tương quan (Bergmeir et al., 2018).")
    st.markdown("**Ổn định lựa chọn.** Biến X được giữ lại kèm tần suất được chọn qua các fold "
                "(Meinshausen & Bühlmann, 2010). Hiển thị cả bán kính phổ của ma trận đồng hành với/không có phạt ổn định.")
    st.markdown("**Thuật toán.** Nicholson et al. (2017) dùng block coordinate descent (BCD); phiên bản PyTorch trong "
                "ứng dụng dùng **ISTA** như một sự điều chỉnh.")
    solver = st.radio("Solver đang dùng (ghi rõ trên giao diện)", ["ISTA (PyTorch)", "BCD (Nicholson et al., 2017)"], horizontal=True)
    st.session_state["varxl_solver"] = solver
    st.info("Lasso có xu hướng chọn một biến tùy ý trong nhóm biến tương quan cao (Zou & Hastie, 2005). Vì VIF cao đã "
            "ghi ở Trang 4.7 (luận văn ghi nhận VIF tới khoảng 8–9), nên dùng Group Lasso hoặc elastic net và xem "
            "tần suất chọn thay vì một lần chạy duy nhất.")

# ============================================================
# TRANG 10 — LSTM
# ============================================================
with tabs[3]:
    st.subheader("Trang 10 — LSTM")
    st.markdown("**Mục tiêu.** Kiểm tra khả năng học quan hệ phi tuyến theo thời gian; nằm giữa VARX-L và DeepVAR.")
    st.latex(r"\mathbf{h}_t = \mathrm{LSTM}\big(\mathbf{Y}_{t-p:t-1},\,\mathbf{X}_{t-p:t-1}\big),\qquad \hat{\mathbf{Y}}_t = f(\mathbf{h}_t)")
    l1, l2, l3, l4 = st.columns(4)
    n_h = l1.number_input("n_h (hidden)", 1, 2048, 64, key="lstm_nh")
    n_lay = l2.number_input("Số lớp", 1, 8, 1, key="lstm_layers")
    Hh = l3.number_input("Horizon H", 1, 90, 1, key="lstm_H")
    batch = l4.number_input("Batch", 1, 4096, 32, key="lstm_batch")
    n_in = int(K + M)
    tot_l, per_l = lstm_params(n_in, int(n_h), int(n_lay))
    head = dense(int(n_h), int(K * Hh))
    st.markdown(f"**Kích thước đầu vào** (batch, L, K+M) = ({batch}, {p}, {n_in}); $n_h$ = {n_h}; số lớp = {n_lay}.")
    st.markdown("**Số tham số một lớp LSTM của PyTorch:** $4 n_h (n_{in} + n_h + 2)$.")
    t1, t2, t3 = st.columns(3)
    t1.metric("LSTM", f"{tot_l:,}")
    t2.metric("Đầu ra FC (n_h → K·H)", f"{head:,}")
    t3.metric("Tổng", f"{tot_l + head:,}")
    ex_tot, _ = lstm_params(10, 64, 1)
    st.caption(f"Ví dụ minh họa: n_in = 10, n_h = 64 → 4·64·76 = {ex_tot:,}. Khi huấn luyện, tổng tham số được lấy từ mô hình "
               "thật (tổng p.numel()); loss theo epoch ở Trang 15.")
    st.warning("LSTM với MSE là **dự báo điểm** nên không có CRPS, trừ khi bổ sung đầu ra phân phối hoặc khoảng dự báo dựa "
               "trên phần dư. Trên bảng Trang 16.1 ghi **'n/a (dự báo điểm)'** thay vì để trống.")
    st.markdown("**Tài liệu.** Hochreiter & Schmidhuber (1997). Beck et al. (2025) ghi nhận hiệu năng của các mô hình học "
                "máy giảm nhiều hơn mô hình thống kê khi chuỗi biến động mạnh, nên LSTM phải chứng minh giá trị bằng so "
                "sánh với Naive và VARX-L, không mặc nhiên.")

# ============================================================
# TRANG 11 — DeepVAR
# ============================================================
with tabs[4]:
    st.subheader("Trang 11 — DeepVAR")
    st.markdown("**Mục tiêu.** Dự báo đa biến **theo xác suất**: mô hình hóa $P(\\mathbf{Y}_t \\mid \\mathbf{Y}_{1:t-1})$.")
    st.code("Historical Y → Neural temporal model → Distribution parameters\n"
            "             → Multivariate forecast → mean / quantiles / uncertainty", language="text")
    st.markdown("**Đầu ra phân phối.** Tham số hóa phân phối đa biến $\\mathcal{N}(\\mu_t, \\Sigma_t)$. "
                "Salinas et al. (2019): ma trận hiệp phương sai dạng hạng thấp cộng đường chéo, và phân phối biên qua "
                "copula Gaussian để xử lý khác biệt thang đo giữa các chuỗi.")
    st.latex(r"\mathrm{NLL} = -\sum_t \log \mathcal{N}(\mathbf{y}_t;\, \mu_t, \Sigma_t)")
    st.markdown("Huấn luyện trên TRAIN, early stopping trên VAL.")

    st.markdown("**Số tham số hiệp phương sai** (tính được): đầy đủ $K(K+1)/2$; hạng thấp $r$ + đường chéo $Kr + K$.")
    r_ = st.number_input("Hạng r", 1, 50, 2, key="dv_r")
    rows = []
    for kk, rr in [(2, 1), (12, 2), (int(K), int(r_))]:
        full, low = cov_params(kk, rr)
        rows.append({"K": kk, "r": rr, "Đầy đủ K(K+1)/2": full, "Hạng thấp K·r + K": low,
                     "Nên dùng": "Đầy đủ (Cholesky)" if low >= full else "Hạng thấp"})
    st.dataframe(pd.DataFrame(rows).drop_duplicates().reset_index(drop=True), hide_index=True, **STRETCH)
    st.caption("K = 2: đầy đủ = 3, hạng thấp r = 1 → 4, hạng thấp không tiết kiệm gì nên dùng ma trận đầy đủ "
               "(Σ = L·Lᵀ, đường chéo của L qua softplus để dương). K = 12: đầy đủ = 78; hạng thấp r = 2 → 36; "
               "hạng thấp chỉ có ý nghĩa khi K lớn, đúng bối cảnh của Salinas et al. (2019).")

    d1, d2, d3 = st.columns(3)
    dv_h = d1.number_input("n_h", 1, 2048, 64, key="dv_nh")
    dv_l = d2.number_input("Số lớp", 1, 8, 1, key="dv_layers")
    dv_kind = d3.radio("Covariance", ["Cholesky đầy đủ", "Hạng thấp + đường chéo"], key="dv_kind")
    tl, _ = lstm_params(int(K), int(dv_h), int(dv_l))
    hd = dist_head(int(dv_h), int(K), dv_kind, int(r_))
    e1, e2, e3 = st.columns(3)
    e1.metric("Mã hóa thời gian (LSTM)", f"{tl:,}")
    e2.metric("Đầu ra phân phối", f"{hd:,}")
    e3.metric("Tổng DeepVAR", f"{tl + hd:,}")
    st.error("**Không** gọi LSTM → Dense → MSE là DeepVAR: đó là LSTM dự báo điểm (Trang 10). DeepVAR phải có đầu ra phân phối "
             "đa biến và hàm mất mát NLL.")
    st.markdown("**Bằng chứng từ tài liệu.** Trong thí nghiệm của Salinas et al. (2019), CRPS và CRPS-Sum cải thiện trung "
                "bình khoảng 10% và 40% so với hai mô hình tốt nhì tương ứng, với 400 mẫu mỗi mô hình khi tính CRPS. "
                "Đây là kết quả trên các bộ dữ liệu của họ, **không tự áp dụng** cho dữ liệu cà phê. Beck et al. (2025) "
                "đưa DeepVAR vào so sánh và vẫn kết luận không phương pháp nào luôn thắng Naive trên chuỗi biến động "
                "mạnh, nên DeepVAR cũng cần kiểm chứng như mọi mô hình.")

# ============================================================
# TRANG 12 — DeepVARX
# ============================================================
with tabs[5]:
    st.subheader("Trang 12 — DeepVARX (mô hình đề xuất)")
    st.markdown("**Mô tả.** DeepVARX mở rộng DeepVAR bằng cách đưa các biến ngoại sinh vào quá trình biểu diễn và dự báo "
                "đa biến, rồi tham số hóa **phân phối** dự báo đa biến.")
    arch = st.radio("Kiến trúc được huấn luyện (cần chọn MỘT)",
                    ["A. Hai nhánh LSTM (nội sinh / ngoại sinh) + fusion — mô tả giao diện",
                     "B. Nhánh tuyến tính VARX-L + nhánh LSTM học phần dư, cổng BDS + cổng tin cậy g(h) — luận văn"],
                    key="dx_arch")
    st.session_state["deepvarx_architecture"] = "A" if arch.startswith("A") else "B"
    st.warning("Mô tả giao diện (hai nhánh LSTM + fusion) khác kiến trúc trong luận văn (nhánh tuyến tính VARX-L cộng nhánh "
               "LSTM học phần dư, nối bằng cổng BDS và cổng tin cậy giảm dần theo horizon). Cần chọn một kiến trúc duy "
               "nhất và để giao diện phản ánh đúng kiến trúc được huấn luyện.")

    x1, x2, x3, x4 = st.columns(4)
    H12 = x1.number_input("Horizon H", 1, 90, 7, key="dx_H")
    hE = x2.number_input("hidden_E", 1, 1024, 64, key="dx_hE")
    hX = x3.number_input("hidden_X", 1, 1024, 32, key="dx_hX")
    Mkf = x4.number_input("Số biến known-future", 0, 100, 0, key="dx_mkf",
                          help="Biến lịch/mùa vụ hoặc có dự báo sẵn tại thời điểm gốc.")
    dx_l = st.number_input("Số lớp LSTM mỗi nhánh", 1, 4, 1, key="dx_layers")
    dx_kind = st.radio("Covariance", ["Cholesky đầy đủ", "Hạng thấp + đường chéo"], key="dx_kind", horizontal=True)
    r12 = st.number_input("Hạng r (nếu hạng thấp)", 1, 50, 2, key="dx_r")

    if arch.startswith("A"):
        st.code("Endogenous branch:  Y(t−p:t−1) → LSTM_E → hidden_E ─┐\n"
                "Exogenous branch:   X          → LSTM_X → hidden_X ─┤\n"
                "                    hidden_E + hidden_X → Fusion → Distribution parameters\n"
                "                                        → Multivariate forecast", language="text")
        tE, _ = lstm_params(int(K), int(hE), int(dx_l))
        tX, _ = lstm_params(int(M + Mkf), int(hX), int(dx_l))
        fus = dense(int(hE + hX), int(hE + hX))
        hd = dist_head(int(hE + hX), int(K), dx_kind, int(r12))
        parts = {"Nhánh nội sinh (LSTM_E)": tE, "Nhánh ngoại sinh (LSTM_X)": tX, "Fusion (FC)": fus,
                 "Đầu ra phân phối": hd}
    else:
        c_dense = linear_params(int(K), int(M), int(p), int(q), jmin)[1]
        tR, _ = lstm_params(int(K + M + Mkf), int(hE), int(dx_l))
        hd = dist_head(int(hE), int(K), dx_kind, int(r12))
        parts = {"Nhánh tuyến tính VARX-L (dày, cận trên)": c_dense, "Nhánh LSTM học phần dư": tR,
                 "Cổng tin cậy g(h), h = 1..H": int(H12), "Đầu ra phân phối": hd}
        st.code("Nhánh tuyến tính VARX-L ─→ ŷ_lin ──────────────┐\n"
                "Phần dư → LSTM ─→ (cổng BDS bật/tắt) ─→ g(h)·ŷ_nl ┴→ phân phối dự báo", language="text")
        st.caption("Cổng BDS là kiểm định thống kê, không có tham số học; g(h) là một đường số học được cho h = 1..H.")
    pt = pd.DataFrame({"Thành phần": list(parts.keys()), "Số tham số": list(parts.values())})
    pt.loc[len(pt)] = ["Tổng", int(pt["Số tham số"].sum())]
    st.dataframe(pt, hide_index=True, **STRETCH)

    st.markdown("**Bản đồ khả dụng của biến ngoại sinh** (từ Tab D, Trang 1). Cửa sổ ngoại sinh chỉ được vươn tới "
                "$(t, t+H]$ cho known-future hoặc bản dự báo X có sẵn tại thời điểm gốc; biến past-only chỉ có cửa sổ "
                "kết thúc tại thời điểm gốc.")
    prov = st.session_state.get("provenance")
    if isinstance(prov, pd.DataFrame) and "Known at forecast time" in prov.columns:
        win = {"Past-only": "X(t−q : t)  — kết thúc tại thời điểm gốc",
               "Known-future": "X(t−q : t+H) — được dùng tương lai",
               "Forecast-only": "X(t−q : t) + bản dự báo X(t+1 : t+H) tại thời điểm gốc"}
        rows_m = [{"Biến": r["Variable"], "Known at forecast time": r["Known at forecast time"],
                   "Cửa sổ được phép": win.get(r["Known at forecast time"], "?"),
                   "Look-ahead risk": r.get("Look-ahead risk", "")}
                  for _, r in prov.iterrows() if str(r["Variable"]) in set(map(str, sel_vars)) or not sel_vars]
        st.dataframe(pd.DataFrame(rows_m), hide_index=True, **STRETCH)
        st.caption("Mâu thuẫn (ví dụ dùng X past-only trong cửa sổ tương lai) bị chặn ở Trang 13 (Leakage Audit). "
                   "Lim et al. (2021) cũng tách đầu vào quan sát được khỏi đầu vào biết trước.")
    else:
        st.info("Chưa có bảng Provenance (Tab D, Trang 1) để dựng bản đồ khả dụng.")

    st.markdown("**Bằng chứng hiển thị:** tổng số tham số và số tham số mỗi nhánh (bảng trên); nếu dùng cổng BDS: "
                "p-value BDS, ngưỡng α và quyết định kích hoạt/không kích hoạt nhánh phi tuyến; nếu dùng cổng tin cậy "
                "theo horizon: giá trị học được g(h) cho h = 1..H (một đường số); liên kết đến ablation (Trang 16.6).")
    st.markdown("**Tài liệu.** Salinas et al. (2019) cho đầu ra phân phối; Nicholson et al. (2017) cho nhánh tuyến tính có "
                "regularization; Lim et al. (2021) cho phân loại biến theo mức khả dụng.")
    st.success("**DeepVARX** là mô hình đề xuất chính của đề tài; hiệu năng được đánh giá ở Trang 16, không ở trang này.")

st.divider()
if st.button("Tiếp theo: Chuẩn bị & Thực nghiệm →", **STRETCH):
    st.switch_page("pages/5_Chuan_bi_thuc_nghiem.py")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"

# Cột meta (không dùng làm biến phân tích)
META_COLS = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"

st.set_page_config(page_title="Phân tích tương quan", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    proc = st.session_state.get("processed_data")
    raw = st.session_state.get("raw_data")
    df_check = proc if proc is not None else raw
    if df_check is not None:
        st.success(f"Dữ liệu: {df_check.shape[0]:,} × {df_check.shape[1]}")
    if st.session_state.get("endogenous_vars"):
        st.info("Endog: "+ ", ".join(st.session_state["endogenous_vars"][:3]))
    if st.session_state.get("exogenous_vars"):
        st.info(f"Exog: {len(st.session_state['exogenous_vars'])} biến")

st.caption(APP_HEADER)
st.title("Trang 2 — Phân tích tương quan & Lựa chọn biến")

# ── Kiểm tra dữ liệu ──────────────────────────────────────────────────────
proc = st.session_state.get("processed_data")
raw = st.session_state.get("raw_data")
df = proc if proc is not None else raw

if df is None:
    st.warning("Chưa có dữ liệu. Vui lòng tải dữ liệu ở Trang 1 trước.")
    if st.button("Đi đến Trang 1"):
        st.switch_page("pages/1_Tai_du_lieu.py")
    st.stop()

if proc is None:
    st.warning("Dữ liệu chưa được tiền xử lý. Kết quả phân tích có thể bị ảnh hưởng bởi missing/outlier.")

# Chọn cột số (loại bỏ meta)
num_all = [c for c in df.select_dtypes(include="number").columns if c not in META_COLS]
# Biến ứng viên (loại bỏ target)
candidate_cols = [c for c in num_all if c != TARGET_COL]

st.caption(f"Dữ liệu: **{df.shape[0]:,}** dòng | Biến số: **{len(num_all)}** | Target: **{TARGET_COL}** | Biến ứng viên: **{len(candidate_cols)}**")

if TARGET_COL not in df.columns:
    st.error(f"Không tìm thấy cột target `{TARGET_COL}`trong dữ liệu.")
    st.stop()

numeric_df = df[num_all].dropna()
st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# 4 TAB PHÂN TÍCH
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Pearson",
    "Spearman",
    "VIF",
    "CCF",
])

# TAB 1 — PEARSON ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Tương quan Pearson")
    st.markdown("Đo mức độ **tuyến tính** giữa 2 biến liên tục. Giá trị ∈ [−1, 1]. |r| > 0.7: mạnh · 0.4–0.7: trung bình · < 0.4: yếu.")

    selected_p = st.multiselect("Chọn cột phân tích Pearson", num_all,
                                default=num_all[:min(12, len(num_all))], key="p_cols")
    if len(selected_p) >= 2:
        corr_p = numeric_df[selected_p].corr(method="pearson")
        fig = go.Figure(go.Heatmap(
            z=corr_p.values, x=corr_p.columns, y=corr_p.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_p.values, 2), texttemplate="%{text}",
            colorbar=dict(title="r"),
        ))
        fig.update_layout(title="Ma trận Pearson", height=max(420, 55*len(selected_p)),
                          margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if TARGET_COL in selected_p:
            st.subheader(f"Tương quan với {TARGET_COL}")
            corr_target = corr_p[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
            fig2 = px.bar(corr_target.reset_index(), x="index", y=TARGET_COL,
                          title=f"Pearson r với {TARGET_COL}",
                          color=TARGET_COL, color_continuous_scale="RdBu", range_color=[-1,1])
            fig2.add_hline(y=0.4, line_dash="dash", line_color="green", annotation_text="0.4")
            fig2.add_hline(y=-0.4, line_dash="dash", line_color="green")
            fig2.update_layout(height=380, xaxis_title="Biến", yaxis_title="r", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.session_state["_pearson_target"] = corr_target.to_dict()
    else:
        st.info("Chọn ít nhất 2 cột.")

# TAB 2 — SPEARMAN ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Tương quan Spearman")
    st.markdown("Đo tương quan **đơn điệu** (rank-based). Phù hợp khi dữ liệu không chuẩn hoặc có outlier. |ρ| > 0.7: mạnh.")

    selected_s = st.multiselect("Chọn cột phân tích Spearman", num_all,
                                default=num_all[:min(12, len(num_all))], key="s_cols")
    if len(selected_s) >= 2:
        corr_s = numeric_df[selected_s].corr(method="spearman")
        fig = go.Figure(go.Heatmap(
            z=corr_s.values, x=corr_s.columns, y=corr_s.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr_s.values, 2), texttemplate="%{text}",
            colorbar=dict(title="rho"),
        ))
        fig.update_layout(title="Ma trận Spearman", height=max(420, 55*len(selected_s)),
                          margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if TARGET_COL in selected_s:
            st.subheader(f"Tương quan Spearman với {TARGET_COL}")
            corr_st = corr_s[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
            fig2 = px.bar(corr_st.reset_index(), x="index", y=TARGET_COL,
                          title=f"Spearman ρ với {TARGET_COL}",
                          color=TARGET_COL, color_continuous_scale="RdBu", range_color=[-1,1])
            fig2.add_hline(y=0.4, line_dash="dash", line_color="green")
            fig2.add_hline(y=-0.4, line_dash="dash", line_color="green")
            fig2.update_layout(height=380, xaxis_title="Biến", yaxis_title="rho", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
            st.session_state["_spearman_target"] = corr_st.to_dict()
    else:
        st.info("Chọn ít nhất 2 cột.")

# TAB 3 — VIF ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("VIF — Variance Inflation Factor")
    st.markdown("**VIF** đo đa cộng tuyến. VIF=1: không đa cộng tuyến · VIF<5: chấp nhận · VIF≥5: · VIF≥10: nghiêm trọng")

    selected_v = st.multiselect("Chọn biến tính VIF (nên loại bỏ target trước)",
                                candidate_cols, default=candidate_cols[:min(12, len(candidate_cols))], key="v_cols")
    if len(selected_v) >= 2:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            Xv = numeric_df[selected_v].dropna()
            X_const = np.column_stack([np.ones(len(Xv)), Xv.values])
            vif_vals = [variance_inflation_factor(X_const, i+1) for i in range(len(selected_v))]
            vif_df = pd.DataFrame({"Biến": selected_v, "VIF": np.round(vif_vals, 4)}).sort_values("VIF", ascending=False)

            def _c(v):
                if v >= 10: return "background-color:#fee2e2;color:#991b1b"
                if v >= 5: return "background-color:#fef9c3;color:#92400e"
                return "background-color:#dcfce7;color:#166534"
            st.dataframe(vif_df.style.applymap(_c, subset=["VIF"]).format({"VIF":"{:.4f}"}),
                         use_container_width=True, height=280)

            fig = px.bar(vif_df.sort_values("VIF"), x="VIF", y="Biến", orientation="h",
                         color="VIF", color_continuous_scale=["#22c55e","#eab308","#ef4444"],
                         title="VIF theo biến")
            fig.add_vline(x=5, line_dash="dash", line_color="#eab308", annotation_text="VIF=5")
            fig.add_vline(x=10, line_dash="dash", line_color="#ef4444", annotation_text="VIF=10")
            fig.update_layout(height=max(350, 32*len(selected_v)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.session_state["_vif_data"] = dict(zip(vif_df["Biến"], vif_df["VIF"]))
            n_h = int((vif_df["VIF"] >= 10).sum())
            n_m = int(((vif_df["VIF"] >= 5) & (vif_df["VIF"] < 10)).sum())
            if n_h > 0: st.error(f"{n_h} biến VIF ≥ 10")
            if n_m > 0: st.warning(f"{n_m} biến 5 ≤ VIF < 10")
            if n_h == 0 and n_m == 0: st.success("Không có đa cộng tuyến nghiêm trọng (VIF < 5 tất cả)")
        except ImportError:
            st.error("Cần cài: pip install statsmodels")
        except Exception as e:
            st.error(f"Lỗi VIF: {e}")
    else:
        st.info("Chọn ít nhất 2 biến.")

# TAB 4 — CCF ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("CCF — Cross-Correlation Function (Tương quan chéo)")
    st.markdown("Đo tương quan giữa biến ngoại sinh và **coffee_price_vnd_kg** ở các độ trễ khác nhau. Phát hiện lag nhân quả.")

    try:
        from statsmodels.tsa.stattools import ccf as _ccf

        c1, c2, c3 = st.columns(3)
        x_c = c1.selectbox("Biến X (nguyên nhân)", candidate_cols, key="ccf_x")
        n_lag = c3.slider("Số lag tối đa", 5, 60, 30, key="ccf_lags")

        sx = numeric_df[x_c].values
        sy = numeric_df[TARGET_COL].values
        min_len = min(len(sx), len(sy))
        ccf_v = _ccf(sx[:min_len], sy[:min_len], nlags=n_lag, alpha=None)
        lags = np.arange(len(ccf_v))
        conf = 1.96 / np.sqrt(min_len)

        fig = go.Figure()
        for lag, val in zip(lags, ccf_v):
            clr = "#6366f1"if abs(val) > conf else "#94a3b8"
            fig.add_trace(go.Scatter(x=[lag, lag], y=[0, val], mode="lines",
                                     line=dict(color=clr, width=3), showlegend=False))
        fig.add_trace(go.Scatter(x=lags, y=ccf_v, mode="markers",
                                 marker=dict(color=["#6366f1"if abs(v)>conf else "#94a3b8"for v in ccf_v], size=7),
                                 name="CCF"))
        fig.add_hline(y=conf, line_dash="dash", line_color="#f59e0b", annotation_text=f"+{conf:.3f} 95%CI")
        fig.add_hline(y=-conf, line_dash="dash", line_color="#f59e0b", annotation_text=f"-{conf:.3f} 95%CI")
        fig.update_layout(title=f"CCF: {x_c} → {TARGET_COL}", xaxis_title="Lag", yaxis_title="CCF",
                          height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        sig = [(int(l), float(v)) for l, v in zip(lags, ccf_v) if abs(v) > conf]
        if sig:
            best = sig[np.argmax([abs(v) for _, v in sig])]
            st.success(f"Lag có CCF mạnh nhất: **lag = {best[0]}** (CCF = {best[1]:.4f})")
    except ImportError:
        st.error("Cần cài: pip install statsmodels")
    except Exception as e:
        st.error(f"Lỗi CCF: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# BẢNG TỔNG HỢP
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Bảng tổng hợp chỉ số tương quan")
st.markdown("Mỗi hàng là 1 biến ứng viên — so sánh Pearson, Spearman, VIF, CCF (max lag) với target.")

if st.button("Tính lại bảng tổng hợp", type="secondary"):
    with st.spinner("Đang tính..."):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.tsa.stattools import ccf as _ccf2

            rows = []
            Xv_all = numeric_df[candidate_cols].dropna()
            X_const_all = np.column_stack([np.ones(len(Xv_all)), Xv_all.values])
            for i, col in enumerate(candidate_cols):
                pr = numeric_df[[col, TARGET_COL]].corr(method="pearson").loc[col, TARGET_COL]
                sp = numeric_df[[col, TARGET_COL]].corr(method="spearman").loc[col, TARGET_COL]
                try:
                    vif_val = variance_inflation_factor(X_const_all, i+1)
                except Exception:
                    vif_val = float("nan")
                try:
                    sx = Xv_all[col].values
                    sy = numeric_df[TARGET_COL].values[:len(sx)]
                    cv = _ccf2(sx, sy, nlags=30, alpha=None)
                    best_ccf = float(np.max(np.abs(cv)))
                    best_lag = int(np.argmax(np.abs(cv)))
                except Exception:
                    best_ccf, best_lag = float("nan"), -1
                rows.append({
                    "Biến": col,
                    "Pearson r": round(float(pr), 4),
                    "|Pearson|": round(abs(float(pr)), 4),
                    "Spearman rho": round(float(sp), 4),
                    "|Spearman|": round(abs(float(sp)), 4),
                    "VIF": round(float(vif_val), 2),
                    "CCF max": round(best_ccf, 4),
                    "CCF lag*": best_lag,
                })

            summary_df = pd.DataFrame(rows).sort_values("|Pearson|", ascending=False).reset_index(drop=True)
            st.session_state["corr_summary"] = summary_df
        except Exception as e:
            st.error(f"Lỗi: {e}")

if st.session_state.get("corr_summary") is not None:
    sdf = st.session_state["corr_summary"]
    st.dataframe(
        sdf.style.format({
            "Pearson r": "{:.4f}", "|Pearson|": "{:.4f}",
            "Spearman rho": "{:.4f}", "|Spearman|": "{:.4f}",
            "VIF": "{:.2f}", "CCF max": "{:.4f}",
        }).background_gradient(subset=["|Pearson|","|Spearman|"], cmap="YlOrRd"),
        use_container_width=True, height=420,
    )
else:
    st.info("Nhấn **Tính lại bảng tổng hợp** để xem so sánh tất cả biến.")

# ═══════════════════════════════════════════════════════════════════════════
# CHON BIEN DUA VAO KIEM DINH TRANG 3
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Chon bien dua vao kiem dinh chuan hoa thoi gian (Trang 3)")
st.markdown("""
Dua tren ket qua phan tich tuong quan, chon cac bien ung vien co tuong quan cao voi target
de dua vao kiem dinh tinh dung (ADF · PP · KPSS), dong lien ket (Johansen) va nhan qua (Granger) o Trang 3.

> **Luu y**: Day la buoc **loc bien ung vien** de giam khoi luong kiem dinh — chua phai chon Endo/Exog.
> Viec phan cong Endo / Exog chinh thuc se thuc hien o **Trang 5**.
""")

# Goi y tu bang tong hop neu co
corr_summary = st.session_state.get("corr_summary")
if corr_summary is not None:
    threshold = st.slider(
        "Loc tu dong: lay bien co |Pearson| >=",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="Dieu chinh nguong de loc danh sach bien goi y",
        key="corr_thresh_slider",
    )
    suggested = corr_summary[corr_summary["|Pearson|"] >= threshold]["Bien"].tolist()
    if TARGET_COL in num_all and TARGET_COL not in suggested:
        suggested = [TARGET_COL] + suggested
    st.caption(f"Goi y: **{len(suggested)}** bien co |Pearson| >= {threshold}")
else:
    threshold = 0.3
    suggested = num_all[:min(10, len(num_all))]
    st.info("Chua co bang tong hop. Hay nhan 'Tinh lai bang tong hop' o tren de co goi y tu dong.")

corr_prev = st.session_state.get("corr_selected_vars") or []
corr_prev = [c for c in corr_prev if c in num_all]
default_sel = corr_prev if corr_prev else suggested

selected_for_ts = st.multiselect(
    "Chon cac bien dua vao kiem dinh Trang 3",
    options=num_all,
    default=default_sel,
    key="corr_ts_vars",
    help="Chon TARGET + cac bien ngoai sinh co tuong quan cao.",
)

if len(selected_for_ts) >= 2:
    st.subheader("Heatmap tuong quan cua tap bien da chon")
    corr_sel = numeric_df[selected_for_ts].corr(method="pearson")
    fig_h = go.Figure(go.Heatmap(
        z=corr_sel.values, x=corr_sel.columns, y=corr_sel.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_sel.values, 2), texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig_h.update_layout(
        title="Pearson Heatmap — tap bien da chon cho Trang 3",
        height=max(380, 50 * len(selected_for_ts)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_h, use_container_width=True)

st.divider()
if st.button("Luu va chuyen sang Trang 3", type="primary", use_container_width=True):
    if not selected_for_ts:
        st.error("Chon it nhat 1 bien.")
    else:
        st.session_state["corr_selected_vars"] = selected_for_ts
        st.success(
            f"Da luu {len(selected_for_ts)} bien: "
            f"{', '.join(selected_for_ts[:5])}{'...' if len(selected_for_ts) > 5 else ''}"
        )
        st.switch_page("pages/3_Phan_tich_chuoi_thoi_gian.py")


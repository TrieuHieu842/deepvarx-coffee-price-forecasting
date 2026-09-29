import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
META_COLS = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"

st.set_page_config(page_title="Phân tích chuỗi thời gian", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    endo = st.session_state.get("endogenous_vars", [])
    exog = st.session_state.get("exogenous_vars", [])
    if endo: st.info("Endog: "+ ", ".join(endo[:3]))
    if exog: st.info(f"Exog: {len(exog)} biến")

st.caption(APP_HEADER)
st.title("Trang 3 — Phân tích chuỗi thời gian")
st.markdown("Kiểm định tính dừng (ADF · PP · KPSS), đồng liên kết (Johansen), và quan hệ nhân quả (Granger).")

# ── Lấy dữ liệu ───────────────────────────────────────────────────────────
proc = st.session_state.get("processed_data")
raw = st.session_state.get("raw_data")
df = proc if proc is not None else raw

if df is None:
    st.warning("Chưa có dữ liệu. Quay lại Trang 1.")
    if st.button("Trang 1"): st.switch_page("pages/1_Tai_du_lieu.py")
    st.stop()

# Lay bien kiem dinh tu Page 2 (corr_selected_vars), fallback toan bo so
corr_sel_vars = st.session_state.get("corr_selected_vars") or []
num_all = [c for c in df.select_dtypes(include="number").columns if c not in META_COLS]
analysis_vars = [c for c in corr_sel_vars if c in df.columns] if corr_sel_vars else num_all

if not corr_sel_vars:
    st.warning(
        "Chua chon bien o Trang 2. Su dung tat ca bien so. "
        "Hay quay lai Trang 2 de chon bien co tuong quan cao."
    )
else:
    st.info(f"Kiem dinh tren **{len(analysis_vars)}** bien duoc chon tu Trang 2: "
            f"{', '.join(analysis_vars[:6])}{'...' if len(analysis_vars)>6 else ''}")

# Tong hop theo ngay (mean across provinces)
date_col = st.session_state.get("date_col", "date")
group_col = st.session_state.get("group_col", "province")
if date_col in df.columns and group_col in df.columns:
    agg_df = df.groupby(date_col)[analysis_vars].mean().reset_index().set_index(date_col).sort_index()
elif date_col in df.columns:
    agg_df = df.set_index(date_col)[analysis_vars].sort_index()
else:
    agg_df = df[analysis_vars].copy()
agg_df = agg_df.dropna()


st.caption(f"Chuỗi phân tích: **{agg_df.shape[0]:,}** điểm thời gian | **{len(analysis_vars)}** biến")
st.divider()

tab1, tab2, tab3 = st.tabs(["Tính dừng (ADF · PP · KPSS)", "Johansen Cointegration", "Granger Causality"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TÍNH DỪNG
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Kiểm định tính dừng")
    st.markdown("""
    | Kiểm định | H₀ | Kết luận khi p < 0.05 |
    |-----------|----|-----------------------|
    | **ADF** (Augmented Dickey-Fuller) | Chuỗi có nghiệm đơn vị (không dừng) | Bác bỏ H₀ → chuỗi **dừng** I(0) |
    | **PP** (Phillips-Perron) | Chuỗi có nghiệm đơn vị | Bác bỏ H₀ → chuỗi **dừng** I(0) |
    | **KPSS** | Chuỗi **dừng** | Bác bỏ H₀ → chuỗi **không dừng** |
    """)

    sig_level = st.select_slider("Mức ý nghĩa α", [0.01, 0.05, 0.10], value=0.05)
    diff_mode = st.checkbox("Tự động sai phân chuỗi chưa dừng (I(1)) trước khi kiểm định PP & Johansen", value=False)
    run_stationarity = st.button("Chạy kiểm định tính dừng", type="primary")

    if run_stationarity:
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            from arch.unitroot import PhillipsPerron
            rows = []
            for col in analysis_vars:
                s = agg_df[col].dropna()
                # ADF
                adf_res = adfuller(s, autolag="AIC")
                adf_p = float(adf_res[1])
                # PP
                try:
                    pp_res = PhillipsPerron(s)
                    pp_p = float(pp_res.pvalue)
                except Exception:
                    pp_p = float("nan")
                # KPSS
                try:
                    kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
                except Exception:
                    kpss_p = float("nan")

                adf_stat = "Dừng "if adf_p < sig_level else "Không dừng "
                pp_stat = "Dừng "if pp_p < sig_level else "Không dừng "
                kpss_stat_ = "Không dừng "if kpss_p < sig_level else "Dừng "

                # Kết luận tổng hợp
                votes_stationary = sum([adf_p < sig_level, pp_p < sig_level, kpss_p >= sig_level])
                if votes_stationary >= 2:
                    conclusion = "I(0)"
                else:
                    conclusion = "I(1)"

                rows.append({
                    "Biến": col, "ADF p-value": round(adf_p, 4), "ADF": adf_stat,
                    "PP p-value": round(pp_p, 4), "PP": pp_stat,
                    "KPSS p-value": round(kpss_p, 4), "KPSS": kpss_stat_,
                    "Kết luận": conclusion,
                })

            stat_df = pd.DataFrame(rows)
            st.session_state["stationarity_df"] = stat_df

        except ImportError as e:
            st.error(f"Thiếu thư viện hoặc import không hợp lệ: {e}")
            st.code("python -m pip install pandas numpy plotly statsmodels arch")
        except Exception as e:
            st.error(f"Lỗi: {e}")

    if st.session_state.get("stationarity_df") is not None:
        sdf = st.session_state["stationarity_df"]
        def _color_concl(val):
            return "color:#22c55e;font-weight:bold"if val=="I(0)"else "color:#ef4444;font-weight:bold"
        st.dataframe(
            sdf.style.applymap(_color_concl, subset=["Kết luận"]).format(
                {"ADF p-value":"{:.4f}", "PP p-value":"{:.4f}", "KPSS p-value":"{:.4f}"}
            ),
            use_container_width=True, height=350,
        )
        n_i0 = int((sdf["Kết luận"]=="I(0)").sum())
        n_i1 = int((sdf["Kết luận"]=="I(1)").sum())
        c1,c2 = st.columns(2)
        c1.metric("Chuỗi dừng I(0)", n_i0)
        c2.metric("Chuỗi không dừng I(1)", n_i1)
        if n_i1 > 0:
            st.info(f"Có {n_i1} chuỗi I(1) — kiểm định Johansen để phát hiện đồng liên kết.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — JOHANSEN
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Johansen Cointegration Test")
    st.markdown("""
    **Điều kiện áp dụng**: Các chuỗi phải cùng bậc tích hợp I(1).
    Kiểm định Johansen xác định **số quan hệ đồng liên kết (r)** giữa các biến.
    - Nếu r > 0: tồn tại quan hệ dài hạn → cân nhắc mô hình VECM
    - Nếu r = 0: không đồng liên kết → sử dụng VAR với chuỗi sai phân
    """)

    sdf = st.session_state.get("stationarity_df")
    i1_vars = [c for c in analysis_vars if c in (sdf[sdf["Kết luận"]=="I(1)"]["Biến"].tolist() if sdf is not None else analysis_vars)]
    if not i1_vars:
        i1_vars = analysis_vars

    selected_j = st.multiselect("Chọn biến I(1) để kiểm định Johansen", i1_vars,
                                default=i1_vars[:min(6, len(i1_vars))], key="johansen_vars")
    det_order = st.selectbox("Deterministic order", [-1, 0, 1], index=1,
                              help="-1: không hằng số, 0: hằng số ngoài, 1: hằng số trong")
    k_ar = st.slider("Số lag (k_ar)", 1, 12, 2)

    if st.button("Chạy Johansen", type="primary"):
        if len(selected_j) < 2:
            st.warning("Chọn ít nhất 2 biến.")
        else:
            try:
                from statsmodels.tsa.vector_ar.vecm import coint_johansen
                data_j = agg_df[selected_j].dropna()
                result_j = coint_johansen(data_j, det_order=det_order, k_ar_diff=k_ar)

                trace_rows = []
                for i in range(len(selected_j)):
                    trace_rows.append({
                        "r ≤ i": i,
                        "Trace Statistic": round(float(result_j.lr1[i]), 4),
                        "Critical 90%": round(float(result_j.cvt[i, 0]), 4),
                        "Critical 95%": round(float(result_j.cvt[i, 1]), 4),
                        "Critical 99%": round(float(result_j.cvt[i, 2]), 4),
                        "Bác bỏ H₀ (95%)?": "Có"if result_j.lr1[i] > result_j.cvt[i, 1] else "Không",
                    })
                maxeig_rows = []
                for i in range(len(selected_j)):
                    maxeig_rows.append({
                        "r ≤ i": i,
                        "Max-Eig Statistic": round(float(result_j.lr2[i]), 4),
                        "Critical 90%": round(float(result_j.cvm[i, 0]), 4),
                        "Critical 95%": round(float(result_j.cvm[i, 1]), 4),
                        "Critical 99%": round(float(result_j.cvm[i, 2]), 4),
                        "Bác bỏ H₀ (95%)?": "Có"if result_j.lr2[i] > result_j.cvm[i, 1] else "Không",
                    })

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Trace Statistic**")
                    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)
                with col_b:
                    st.markdown("**Max-Eigenvalue Statistic**")
                    st.dataframe(pd.DataFrame(maxeig_rows), use_container_width=True)

                r_count = sum(1 for row in trace_rows if ""in row["Bác bỏ H₀ (95%)?"])
                if r_count > 0:
                    st.success(f"Phát hiện **{r_count}** quan hệ đồng liên kết — Cân nhắc mô hình VECM.")
                else:
                    st.info("Không phát hiện đồng liên kết — Sử dụng VAR với chuỗi sai phân.")
            except ImportError:
                st.error("Cần: pip install statsmodels")
            except Exception as e:
                st.error(f"Lỗi Johansen: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GRANGER
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Granger Causality Test")
    st.markdown("""
    **Granger causality**: X **Granger-causes** Y nếu dự báo Y dùng lịch sử X & Y tốt hơn chỉ dùng lịch sử Y.
    > **Lưu ý**: Đây là nhân quả thống kê, không phải nhân quả nhân tố. Khi các biến có đồng liên kết,
    > kết quả Granger trong VAR mức đơn có thể lệch — cần dùng VECM hoặc kết hợp với kiểm định Johansen.
    """)

    target_g = st.selectbox("Biến kết quả (Y — cần dự báo)", [TARGET_COL] + [c for c in analysis_vars if c!=TARGET_COL],
                            index=0, key="granger_target")
    causes_g = st.multiselect("Biến nguyên nhân (X) — Granger causes Y?",
                              [c for c in analysis_vars if c!=target_g],
                              default=[c for c in analysis_vars if c!=target_g][:min(8, len(analysis_vars)-1)],
                              key="granger_causes")
    max_lag_g = st.slider("Maximum lag", 1, 14, 7, key="granger_lag")
    sig_g = st.select_slider("Mức ý nghĩa α", [0.01, 0.05, 0.10], value=0.05, key="granger_sig")

    if st.button("Chạy Granger Causality", type="primary"):
        if not causes_g:
            st.warning("Chọn ít nhất 1 biến nguyên nhân.")
        else:
            try:
                from statsmodels.tsa.stattools import grangercausalitytests

                g_rows = []
                prog = st.progress(0)
                for ci, cause in enumerate(causes_g):
                    try:
                        data_g = agg_df[[target_g, cause]].dropna()
                        result_g = grangercausalitytests(data_g[[target_g, cause]], maxlag=max_lag_g, verbose=False)
                        for lag in range(1, max_lag_g+1):
                            p_f = float(result_g[lag][0]["ssr_ftest"][1])
                            g_rows.append({
                                "Nguyên nhân (X)": cause,
                                "Kết quả (Y)": target_g,
                                "Lag": lag,
                                "p-value (F-test)": round(p_f, 4),
                                "Kết luận": f"X→Y (α={sig_g})"if p_f < sig_g else "Không nhân quả",
                            })
                    except Exception as ex:
                        g_rows.append({"Nguyên nhân (X)": cause, "Kết quả (Y)": target_g,
                                       "Lag": -1, "p-value (F-test)": float("nan"), "Kết luận": f"Lỗi: {ex}"})
                    prog.progress((ci+1)/len(causes_g))

                g_df = pd.DataFrame(g_rows)
                st.session_state["granger_df"] = g_df

            except ImportError:
                st.error("Cần: pip install statsmodels")
            except Exception as e:
                st.error(f"Lỗi Granger: {e}")

    if st.session_state.get("granger_df") is not None:
        gdf = st.session_state["granger_df"]

        # Best lag per cause — bỏ qua các hàng lỗi (p-value = NaN)
        valid_gdf = gdf.dropna(subset=["p-value (F-test)"])
        if valid_gdf.empty:
            st.warning("Không có kết quả hợp lệ — tất cả biến đều bị lỗi khi kiểm định.")
            st.stop()
        best_idx = valid_gdf.groupby("Nguyên nhân (X)")["p-value (F-test)"].idxmin().dropna()
        best_rows = valid_gdf.loc[best_idx].copy()
        best_rows = best_rows.sort_values("p-value (F-test)")
        
        st.subheader("Kết quả tốt nhất theo lag (p-value nhỏ nhất)")
        def _c_granger(val):
            return "color:#22c55e;font-weight:bold"if ""in str(val) else "color:#ef4444"
        st.dataframe(
            best_rows.style.applymap(_c_granger, subset=["Kết luận"]).format({"p-value (F-test)":"{:.4f}"}),
            use_container_width=True, height=320,
        )

        sig_causes = best_rows[best_rows["p-value (F-test)"] < sig_g]["Nguyên nhân (X)"].tolist()
        if sig_causes:
            st.success(f"**{len(sig_causes)}** biến có nhân quả Granger → {target_g}: "+ ", ".join(sig_causes[:5]))
        else:
            st.info("Không có biến nào có nhân quả Granger có ý nghĩa.")

        with st.expander("Xem toàn bộ kết quả theo lag"):
            st.dataframe(gdf.style.applymap(_c_granger, subset=["Kết luận"]).format({"p-value (F-test)":"{:.4f}"}),
                         use_container_width=True, height=400)

st.divider()
if st.button("Tiếp theo: Giới thiệu mô hình →", use_container_width=True):
    st.switch_page("pages/4_Gioi_thieu_mo_hinh.py")

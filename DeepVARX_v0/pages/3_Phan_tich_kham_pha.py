import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.stats_tests import run_adf_test, compute_vif, cross_group_correlation, adf_summary_table
from utils.plotting import correlation_heatmap, acf_pacf_chart

st.set_page_config(page_title="Phân tích khám phá (EDA)", page_icon="📊", layout="wide")
st.title("Phân tích khám phá dữ liệu (EDA)")

df = st.session_state["processed_data"] if st.session_state["processed_data"] is not None \
    else st.session_state["raw_data"]

if df is None:
    st.warning("Chưa có dữ liệu. Vào trang **Tải dữ liệu** trước.")
    st.stop()

date_col = st.session_state["date_col"]
group_col = st.session_state["group_col"]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

tab1, tab2, tab3, tab4 = st.tabs([
    "Tương quan & VIF (đa cộng tuyến)",
    "Tính dừng (ADF)",
    "ACF / PACF",
    "Tương quan chéo giữa tỉnh (panel redundancy)",
])

# --- 1. Correlation + VIF ----------------------------------------------------
with tab1:
    st.subheader("Ma trận tương quan giữa các biến")
    sel_cols = st.multiselect("Chọn các biến", numeric_cols, default=numeric_cols[:min(8, len(numeric_cols))])
    if len(sel_cols) >= 2:
        corr = df[sel_cols].corr()
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)

        st.subheader("Variance Inflation Factor (VIF) — kiểm tra đa cộng tuyến")
        st.caption("VIF > 5 (hoặc > 10 tùy ngưỡng) cho thấy đa cộng tuyến cao — "
                   "đây là căn cứ để dùng Group Lasso trong VARX-L / DeepVARX.")
        exo_default = st.session_state.get("exogenous_vars") or sel_cols
        vif_cols = st.multiselect("Chọn biến ngoại sinh để tính VIF", sel_cols, default=exo_default[:min(6,len(exo_default))])
        if len(vif_cols) >= 2 and st.button("Tính VIF"):
            vif_df = compute_vif(df, vif_cols)
            st.dataframe(vif_df, use_container_width=True)
            high_vif = vif_df[vif_df["VIF"] > 5]
            if not high_vif.empty:
                st.warning(f"{len(high_vif)} biến có VIF > 5 — nên cân nhắc Group Lasso hoặc loại bớt biến.")
    else:
        st.info("Chọn ít nhất 2 biến để xem tương quan.")

# --- 2. ADF -------------------------------------------------------------------
with tab2:
    st.subheader("Kiểm định nghiệm đơn vị (Augmented Dickey-Fuller)")
    adf_cols = st.multiselect("Chọn biến để kiểm định ADF", numeric_cols, default=numeric_cols[:min(5,len(numeric_cols))])
    if adf_cols and st.button("Chạy kiểm định ADF"):
        summary = adf_summary_table(df, adf_cols)
        st.dataframe(summary, use_container_width=True)
        n_nonstationary = (summary["Kết luận"].astype(str).str.contains("Không dừng")).sum()
        if n_nonstationary > 0:
            st.info(f"{n_nonstationary}/{len(adf_cols)} biến không dừng — cân nhắc sai phân bậc 1 "
                    "(I(1)) trước khi đưa vào nhánh VARX-L.")

# --- 3. ACF/PACF ---------------------------------------------------------------
with tab3:
    st.subheader("Biểu đồ tự tương quan ACF / PACF")
    acf_col = st.selectbox("Chọn biến", numeric_cols)
    nlags = st.slider("Số lag", 5, 60, 30)
    fig1, fig2 = acf_pacf_chart(df[acf_col], nlags=nlags)
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig1, use_container_width=True)
    c2.plotly_chart(fig2, use_container_width=True)
    st.caption("PACF giúp xác định bậc trễ p hợp lý ban đầu cho VAR/VARX; "
               "PACF tắt dần chậm là dấu hiệu chuỗi chưa dừng.")

# --- 4. Cross-province correlation ---------------------------------------------
with tab4:
    st.subheader("Tương quan chéo giữa các tỉnh (đánh giá dư thừa panel)")
    if not group_col or not date_col:
        st.warning("Cần khai báo cột thời gian và cột nhóm/tỉnh ở trang Tải dữ liệu.")
    else:
        value_col = st.selectbox("Chọn biến giá để so sánh giữa các tỉnh", numeric_cols)
        if st.button("Tính tương quan chéo tỉnh"):
            cg = cross_group_correlation(df, date_col, group_col, value_col)
            st.plotly_chart(correlation_heatmap(cg, title=f"Tương quan chéo tỉnh: {value_col}"),
                             use_container_width=True)
            mask = ~np.eye(cg.shape[0], dtype=bool)
            avg_corr = cg.values[mask].mean()
            st.metric("Tương quan trung bình giữa các cặp tỉnh", f"{avg_corr:.4f}")
            if avg_corr > 0.95:
                st.warning(
                    "Tương quan gần như tuyệt đối giữa các tỉnh cùng loại cà phê — cho thấy dư thừa "
                    "thông tin (functional redundancy). Cân nhắc giảm số biến nội sinh xuống còn đại diện "
                    "theo loại cà phê (Robusta/Arabica) thay vì giữ nguyên 12 tỉnh, như đã trình bày trong "
                    "phần lý thuyết DeepVARX."
                )

import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import (
    SCALER_OPTIONS, MISSING_OPTIONS, fit_transform_column,
    handle_missing, winsorize_iqr, detect_outliers_iqr,
)

st.set_page_config(page_title="Tiền xử lý dữ liệu", layout="wide")
st.title("Tiền xử lý dữ liệu")

df = st.session_state["raw_data"]
if df is None:
    st.warning("Chưa có dữ liệu. Vào trang **Tải dữ liệu** trước.")
    st.stop()

date_col = st.session_state["date_col"]
group_col = st.session_state["group_col"]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Missing value", "2. Outlier", "3. Chuẩn hóa / Biến đổi", "4. Xem trước & Lưu"]
)

work_df = df.copy()

# --- 1. Missing value -------------------------------------------------------
with tab1:
    st.subheader("Xử lý giá trị thiếu")
    cols_missing = st.multiselect("Chọn cột cần xử lý missing", numeric_cols,
                                    default=[c for c in numeric_cols if df[c].isna().any()])
    method_label = st.selectbox("Phương pháp", list(MISSING_OPTIONS.keys()))
    method = MISSING_OPTIONS[method_label]

    if method == "median_group" and not group_col:
        st.error("Cần khai báo cột nhóm (tỉnh) ở trang Tải dữ liệu để dùng phương pháp này.")
    elif cols_missing and st.button("Áp dụng xử lý missing"):
        work_df = handle_missing(work_df, cols_missing, method, group_col)
        st.session_state["_step1_df"] = work_df
        st.success(f"Đã xử lý missing cho {len(cols_missing)} cột bằng phương pháp: {method_label}")
        st.dataframe(work_df[cols_missing].isna().sum().rename("Số missing còn lại"))

if "_step1_df" in st.session_state:
    work_df = st.session_state["_step1_df"]

# --- 2. Outlier --------------------------------------------------------------
with tab2:
    st.subheader("Phát hiện & xử lý outlier (IQR)")
    col_check = st.selectbox("Chọn biến để kiểm tra outlier", numeric_cols)
    k = st.slider("Hệ số IQR (k)", 1.0, 3.0, 1.5, 0.1)
    outlier_mask = detect_outliers_iqr(work_df[col_check], k)
    st.write(f"Số điểm outlier phát hiện: **{outlier_mask.sum()}** / {len(work_df)} "
             f"({outlier_mask.sum()/len(work_df)*100:.2f}%)")

    import plotly.express as px
    fig = px.box(work_df, y=col_check, title=f"Boxplot: {col_check}")
    st.plotly_chart(fig, use_container_width=True)

    cols_outlier = st.multiselect("Chọn các cột cần winsorize (kẹp outlier)", numeric_cols)
    if cols_outlier and st.button("Áp dụng winsorize"):
        work_df = winsorize_iqr(work_df, cols_outlier, k)
        st.session_state["_step2_df"] = work_df
        st.success(f"Đã kẹp outlier cho {len(cols_outlier)} cột với k={k}")

if "_step2_df" in st.session_state:
    work_df = st.session_state["_step2_df"]

# --- 3. Chuẩn hóa -------------------------------------------------------------
with tab3:
    st.subheader("Chọn phương pháp chuẩn hóa / biến đổi cho từng biến")
    st.caption(
        "⚠️ Nguyên tắc chống rò rỉ dữ liệu: scaler chỉ được fit trên tập **train**. "
        "Ở bước này, việc *fit* thực sự sẽ chạy trên toàn bộ dữ liệu hiện có để bạn xem trước hình dạng "
        "phân phối sau biến đổi; khi vào trang **Chia dữ liệu**, hệ thống sẽ fit lại đúng chuẩn "
        "chỉ trên tập train trước khi áp dụng cho val/test."
    )

    if "column_methods" not in st.session_state or not isinstance(st.session_state["column_methods"], dict):
        st.session_state["column_methods"] = {}

    n_cols_per_row = 2
    cols_list = numeric_cols
    for i in range(0, len(cols_list), n_cols_per_row):
        row_cols = st.columns(n_cols_per_row)
        for j, colname in enumerate(cols_list[i:i + n_cols_per_row]):
            with row_cols[j]:
                current = st.session_state["column_methods"].get(colname, "Min-Max (0-1)")
                choice = st.selectbox(
                    f"`{colname}`", list(SCALER_OPTIONS.keys()),
                    index=list(SCALER_OPTIONS.keys()).index(current) if current in SCALER_OPTIONS else 0,
                    key=f"scaler_{colname}",
                )
                st.session_state["column_methods"][colname] = choice

    st.markdown("**Gợi ý theo loại biến** (tham khảo mục 2.5 lý thuyết chuẩn hóa):")
    st.markdown("""
    - Giá nội sinh (Robusta/Arabica) → *Log-transform + Sai phân bậc 1* (chuỗi I(1), tăng trưởng nhân tính)
    - Biến vĩ mô (tỷ giá, giá dầu) → *Log-transform* hoặc *Z-score*
    - Biến thời tiết (mưa, nhiệt độ) → *Min-Max* (có biên vật lý tự nhiên)
    - Biến có nhiều outlier cực đoan → *Robust Scaler*
    """)

    if st.button("Xem trước phân phối sau biến đổi"):
        import plotly.express as px
        preview_col = st.selectbox("Chọn biến xem trước", numeric_cols, key="preview_select")
        method = SCALER_OPTIONS[st.session_state["column_methods"].get(preview_col, "Min-Max (0-1)")]
        transformed, _ = fit_transform_column(work_df[preview_col].dropna(), method)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(work_df, x=preview_col, title="Trước biến đổi"), use_container_width=True)
        with c2:
            st.plotly_chart(px.histogram(transformed, title="Sau biến đổi"), use_container_width=True)

# --- 4. Lưu -------------------------------------------------------------------
with tab4:
    st.subheader("Xem trước dữ liệu sau xử lý missing/outlier")
    st.dataframe(work_df.head(30), use_container_width=True)
    st.caption("Lưu ý: bước chuẩn hóa thực tế (fit-on-train) sẽ được áp dụng tại trang "
               "**Chia dữ liệu & chọn biến**, dựa trên lựa chọn phương pháp bạn đã chọn ở tab 3.")

    if st.button("Lưu làm dữ liệu đã xử lý (processed_data)", type="primary"):
        st.session_state["processed_data"] = work_df
        st.success("Đã lưu `processed_data`. Tiếp tục sang trang 'Phân tích khám phá' hoặc 'Chia dữ liệu'.")

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Tải dữ liệu", page_icon="📂", layout="wide")
st.title("Tải dữ liệu")

uploaded = st.file_uploader("Chọn file dữ liệu (CSV hoặc Excel)", type=["csv", "xlsx", "xls"])

if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.session_state["raw_data"] = df
        st.success(f"Đã nạp thành công: {df.shape[0]} dòng × {df.shape[1]} cột")
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")

df = st.session_state["raw_data"]

if df is None:
    st.info("Chưa có dữ liệu. Vui lòng upload file CSV/Excel phía trên ")
    st.stop()

# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Xem dữ liệu", "Mô tả thống kê", "Kiểu dữ liệu & Missing", "Cấu hình cột"])

with tab1:
    st.subheader("Xem trước dữ liệu")
    n_rows = st.slider("Số dòng hiển thị", 5, 200, 20)
    st.dataframe(df.head(n_rows), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", df.shape[1])
    c3.metric("Dung lượng bộ nhớ", f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

with tab2:
    st.subheader("Mô tả thống kê (numeric)")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        st.warning("Không có cột số nào trong dữ liệu.")
    else:
        st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.subheader("Mô tả thống kê (categorical)")
    cat_df = df.select_dtypes(exclude="number")
    if not cat_df.empty:
        st.dataframe(cat_df.describe().T, use_container_width=True)

with tab3:
    st.subheader("Kiểu dữ liệu từng cột")
    dtype_df = pd.DataFrame({
        "Cột": df.columns,
        "Kiểu dữ liệu": df.dtypes.astype(str).values,
        "Số giá trị thiếu": df.isna().sum().values,
        "% thiếu": (df.isna().sum().values / len(df) * 100).round(2),
        "Số giá trị duy nhất": df.nunique().values,
    })
    st.dataframe(dtype_df, use_container_width=True)

    missing_cols = dtype_df[dtype_df["Số giá trị thiếu"] > 0]
    if not missing_cols.empty:
        st.warning(f"Có {len(missing_cols)} cột chứa giá trị thiếu — xử lý ở trang "
                    f"**🧹 Tiền xử lý dữ liệu**.")
    else:
        st.success("Không có giá trị thiếu trong dữ liệu.")

with tab4:
    st.subheader("Khai báo cột đặc biệt (để dùng cho các trang sau)")
    all_cols = list(df.columns)

    date_col = st.selectbox(
        "Cột thời gian (date/time)",
        options=[None] + all_cols,
        index=(all_cols.index(st.session_state["date_col"]) + 1)
        if st.session_state["date_col"] in all_cols else 0,
    )
    group_col = st.selectbox(
        "Cột nhóm/tỉnh (panel group, nếu có)",
        options=[None] + all_cols,
        index=(all_cols.index(st.session_state["group_col"]) + 1)
        if st.session_state["group_col"] in all_cols else 0,
    )

    if st.button("💾 Lưu cấu hình cột"):
        st.session_state["date_col"] = date_col
        st.session_state["group_col"] = group_col
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                st.session_state["raw_data"] = df
            except Exception:
                st.warning("Không thể tự động parse cột thời gian sang datetime — kiểm tra lại định dạng.")
        st.success("Đã lưu cấu hình.")

    if date_col and group_col:
        st.subheader("Xem nhanh chuỗi thời gian theo nhóm")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            value_col = st.selectbox("Chọn biến để xem", numeric_cols)
            import plotly.express as px
            fig = px.line(df, x=date_col, y=value_col, color=group_col,
                          title=f"{value_col} theo thời gian, phân theo {group_col}")
            st.plotly_chart(fig, use_container_width=True)

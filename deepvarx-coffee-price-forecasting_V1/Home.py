import streamlit as st

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê thu mua Việt Nam"

st.set_page_config(page_title="DeepVARX - Home", page_icon="☕", layout="wide")

defaults = {
    "raw_data": None,
    "processed_data": None,
    "date_col": "date",
    "group_col": "province",
    "endogenous_vars": [],
    "exogenous_vars": [],
    "preprocess_done": False,
    "corr_summary": None,
    "split_config": {"train_ratio": 0.72, "val_ratio": 0.08, "test_ratio": 0.20},
    "scaler_type": "StandardScaler",
    "lookback_p": 7,
    "horizon_q": 1,
    "pq_results": {},
    "best_pq": {},
    "train_results": {},
    "model_results": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title(APP_HEADER)
st.markdown(
    "Ứng dụng nghiên cứu hỗ trợ toàn bộ quy trình: từ nạp dữ liệu → phân tích → "
    "huấn luyện → so sánh các mô hình **VAR · VARX · DeepVAR · DeepVARX** "
    "trên dữ liệu bảng giá cà phê xuất khẩu 9 tỉnh Việt Nam."
)
st.divider()
st.markdown("### Quy trình tổng thể")

pages = [
    ("1. Dữ liệu & Tiền xử lý",
     "Upload / tải sẵn, thống kê, missing, outlier, tự động tiền xử lý",
     "pages/1_Tai_du_lieu.py"),
    ("2. Phân tích tương quan",
     "Pearson · Spearman · VIF · CCF — lựa chọn biến Endog / Exog",
     "pages/2_Phan_tich_tuong_quan.py"),
    ("3. Phân tích chuỗi thời gian",
     "ADF · PP · KPSS · Johansen cointegration · Granger causality",
     "pages/3_Phan_tich_chuoi_thoi_gian.py"),
    ("4. Giới thiệu mô hình",
     "Lý thuyết & kiến trúc VAR / VARX / DeepVAR / DeepVARX",
     "pages/4_Gioi_thieu_mo_hinh.py"),
    ("5. Chuẩn bị & Thực nghiệm p,q",
     "Chuẩn hóa · Tạo cửa sổ · Train/Val/Test · Grid-search p,q",
     "pages/5_Chuan_bi_thuc_nghiem.py"),
    ("6. Huấn luyện mô hình",
     "Chọn mô hình · Xem cấu hình · Huấn luyện · Log loss",
     "pages/6_Huan_luyen.py"),
    ("7. Kết quả & So sánh",
     "MAE · RMSE · sMAPE · CV(RMSE) · Biểu đồ dự báo · So sánh mô hình",
     "pages/7_Ket_qua.py"),
]

cols = st.columns(4)
for i, (name, desc, path) in enumerate(pages):
    with cols[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(desc)
            if st.button("Mo trang ->", key=f"nav_{i}", use_container_width=True):
                st.switch_page(path)




with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    raw = st.session_state["raw_data"]
    proc = st.session_state["processed_data"]
    if raw is not None:
        st.success(f"Dữ liệu thô: {raw.shape[0]:,} × {raw.shape[1]}")
    else:
        st.warning("Chưa có dữ liệu")
    if proc is not None:
        st.success(f"Đã tiền xử lý: {proc.shape[0]:,} × {proc.shape[1]}")
    if st.session_state.get("endogenous_vars"):
        st.info("Endog: "+ ", ".join(st.session_state["endogenous_vars"][:2]))
    if st.session_state.get("exogenous_vars"):
        st.info(f"Exog: {len(st.session_state['exogenous_vars'])} biến")

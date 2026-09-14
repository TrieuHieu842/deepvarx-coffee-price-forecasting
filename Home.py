import streamlit as st

st.set_page_config(
    page_title="DeepVARX - Dự báo giá cà phê xuất khẩu",
    page_icon="☕",
    layout="wide",
)

# --- Khởi tạo session_state dùng chung toàn app ---
defaults = {
    "raw_data": None,
    "processed_data": None,
    "date_col": None,
    "group_col": None,
    "column_methods": {},      # {col_name: {"method": ..., "scaler": ...}}
    "endogenous_vars": [],
    "exogenous_vars": [],
    "split_config": {"train_ratio": 0.7, "val_ratio": 0.15},
    "train_df": None,
    "val_df": None,
    "test_df": None,
    "model_configs": {},        # {model_name: {hyperparams...}}
    "results": {},              # {model_name: {"metrics":..., "forecast_df":..., "loss_curve":...}}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam")

st.markdown("""
Ứng dụng hỗ trợ toàn bộ quy trình nghiên cứu: từ nạp dữ liệu, tiền xử lý,
phân tích khám phá, đến huấn luyện và so sánh 4 mô hình dự báo — **VAR, VARX,
LSTM, DeepVARX** — trên dữ liệu bảng (panel) giá cà phê xuất khẩu 12 tỉnh Việt Nam.

### Điều hướng (sidebar bên trái)
1. **📂 Tải dữ liệu** — upload file, xem tổng quan, mô tả thống kê
2. **🧹 Tiền xử lý dữ liệu** — xử lý missing/outlier, chọn phương pháp chuẩn hóa
3. **📊 Phân tích khám phá (EDA)** — tương quan, VIF, ADF, ACF/PACF, tương quan chéo tỉnh
4. **📖 Giới thiệu mô hình** — lý thuyết VAR / VARX / LSTM / DeepVARX
5. **✂️ Chia dữ liệu & chọn biến** — biến nội sinh/ngoại sinh, train/val/test, độ trễ
6. **🎯 Huấn luyện mô hình** — cấu hình & huấn luyện (khung giao diện, chờ nối code)
7. **📈 Kết quả & So sánh** — RMSE/MAE/MAPE, biểu đồ dự báo, ablation, so sánh theo tỉnh

---
👉 Bắt đầu từ trang **"📂 Tải dữ liệu"** ở sidebar bên trái.
""")

with st.sidebar:
    st.success("Chọn trang chức năng phía trên ⬆️")
    if st.session_state["raw_data"] is not None:
        st.caption(f"Dữ liệu đã nạp: {st.session_state['raw_data'].shape[0]} dòng × "
                    f"{st.session_state['raw_data'].shape[1]} cột")
    else:
        st.caption("Chưa có dữ liệu nào được nạp.")

import streamlit as st
from modules.ui import setup_page, STEPS

# Khởi tạo trang, để trống icon
setup_page("Trang chủ", icon="")

# ==========================================
# 1. TÊN ĐỀ TÀI
# ==========================================
st.markdown(
    "<h1 style='text-align: center; color: #2e7bcf;'>TÌM HIỂU MÔ HÌNH DEEPVARX VÀO DỰ BÁO GIÁ THU MUA CÀ PHÊ Ở VIỆT NAM</h1>", 
    unsafe_allow_html=True
)
st.divider()

#2. THÀNH VIÊN THỰC HIỆN
st.subheader("Thành viên thực hiện")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**1. Triệu Phúc Hiếu** - MSSV: 23110217")
with col2:
    st.markdown("**2. Bùi Quốc Hậu** - MSSV: 23110211")
st.caption("Chuyên ngành Hệ thống Thông tin - Trường Đại học Công nghệ Kỹ thuật TP. Hồ Chí Minh")
st.write("---")

# 3. GIỚI THIỆU TÓM TẮT
st.subheader("Giới thiệu đề tài")
st.markdown(
    """
    Hệ thống này được xây dựng nhằm hỗ trợ quá trình phân tích dữ liệu, lựa chọn biến và đưa ra dự báo giá thu mua cà phê tại thị trường Việt Nam. 
    Điểm cốt lõi của đề tài là ứng dụng chiến lược **Phễu lọc hai luồng** để sàng lọc và chốt bộ biến tối ưu nhất.
    
    Hệ thống cung cấp môi trường thử nghiệm toàn diện cho các mô hình chuỗi thời gian đa biến, bao gồm các mô hình thống kê truyền thống (VAR, VARX, VARX-L), 
    các mô hình học sâu (LSTM, DeepVAR) và mô hình đề xuất **DeepVARX**.
    """
)

# 4. CÁC THÔNG TIN GIỚI THIỆU KHÁC
st.subheader("Quy trình hệ thống")
c = st.columns(6)
cards = [
    ("1. Tải dữ liệu", "Phễu lọc chung: xử lý khoảng trống, tần suất giả, đa cộng tuyến."),
    ("2. Tuyến tính", "Các kiểm định: ADF/KPSS, CCF, Granger-HC1, Johansen."),
    ("3. Phi tuyến", "Các kiểm định: BDS, Mutual Information, XGBoost, Ljung-Box/VR."),
    ("4. Chốt biến", "Lập Evidence Matrix, chia 2 giỏ biến, xuất 3 file CSV."),
    ("5. Mô hình", "Thiết lập hồ sơ 6 mô hình, tinh chỉnh siêu tham số và huấn luyện."),
    ("6. Báo cáo", "Đánh giá Metric, DM test, phân tích phần dư và Attention Heatmap."),
]
for col, (t, body) in zip(c, cards):
    with col:
        with st.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(body)

st.subheader("Hướng dẫn sử dụng")
st.markdown(
    """
    1. Bắt đầu tại **Trang 1**: Chọn nguồn dữ liệu (dữ liệu mẫu, tải lên, hoặc từ thư mục `data/`), xác định cột thời gian và biến mục tiêu Y, sau đó bấm **Nạp dữ liệu**.
    2. Thực hiện tuần tự qua các trang. Nhấn nút **Chốt...** ở cuối mỗi bước (hoặc **Lưu & khởi tạo dữ liệu huấn luyện** tại Trang 4) để hệ thống lưu trữ trạng thái vào bộ nhớ tạm cho các bước tiếp theo.
    3. Nếu cần thực hiện lại một bước trước đó, dữ liệu của các bước phía sau sẽ tự động bị xóa để ngăn chặn việc sử dụng dữ liệu cũ không nhất quán.
    4. Nhằm tránh hiện tượng rò rỉ dữ liệu (data leakage), mọi kiểm định thống kê đều được thiết lập chỉ tính toán trên tập **TRAIN**.
    """
)

st.info(
    "Lưu ý cơ sở lý thuyết: Tập Cơ Sở bao gồm các biến vượt qua toàn bộ kiểm định tuyến tính. "
    "Tập Mở Rộng bao gồm Tập Cơ Sở kết hợp với các biến phi tuyến hoặc có tính logic nghiệp vụ mạnh. "
    "Một biến chỉ bị loại bỏ hoàn toàn khỏi DeepVARX sau khi đi qua bước Ablation (Attention)."
)

st.subheader("Trạng thái tiến trình hiện tại")
cols = st.columns(len(STEPS))
for col, (key, label) in zip(cols, STEPS):
    with col:
        # Thay thế emoji bằng Text trạng thái rõ ràng
        status_text = "Hoàn thành" if st.session_state.get(key) is not None else "Chưa thực hiện"
        st.metric(label.split(" · ")[0], status_text)
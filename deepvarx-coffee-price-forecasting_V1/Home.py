import streamlit as st
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá thu mua cà phê Việt Nam"

st.set_page_config(
    page_title="DeepVARX - Home",
    page_icon="☕",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    # Data
    "raw_data": None,
    "processed_data": None,

    # Data configuration
    "date_col": "date",
    "group_col": "province",
    "endogenous_vars": [],
    "exogenous_vars": [],

    # Preprocessing
    "preprocess_done": False,
    "corr_summary": None,

    # Experiment
    "split_config": {
        "train_ratio": 0.72,
        "val_ratio": 0.08,
        "test_ratio": 0.20,
    },
    "scaler_type": "StandardScaler",
    "lookback_p": 7,
    "horizon_q": 1,

    # Model
    "pq_results": {},
    "best_pq": {},
    "train_results": {},
    "model_results": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_current_data():
    """
    Ưu tiên dữ liệu đã được xử lý.
    Nếu chưa có thì sử dụng dữ liệu thô.
    """
    processed = st.session_state.get("processed_data")
    raw = st.session_state.get("raw_data")

    if processed is not None:
        return processed

    return raw


def get_data_scope():
    """
    Lấy thông tin phạm vi dữ liệu tự động từ dữ liệu
    đã được nạp/xử lý ở các trang trước.

    Không hard-code số liệu.
    """

    df = get_current_data()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "available": False,
            "n_rows": None,
            "n_cols": None,
            "date_min": None,
            "date_max": None,
            "frequency": None,
            "n_provinces": None,
            "n_endog": len(st.session_state.get("endogenous_vars", [])),
            "n_exog": len(st.session_state.get("exogenous_vars", [])),
        }

    date_col = st.session_state.get("date_col", "date")
    group_col = st.session_state.get("group_col", "province")

    # --------------------------------------------------------
    # Số dòng / số biến
    # --------------------------------------------------------

    n_rows = len(df)
    n_cols = len(df.columns)

    # --------------------------------------------------------
    # Khoảng thời gian
    # --------------------------------------------------------

    date_min = None
    date_max = None

    if date_col in df.columns:
        dates = pd.to_datetime(
            df[date_col],
            errors="coerce"
        ).dropna()

        if not dates.empty:
            date_min = dates.min()
            date_max = dates.max()

    # --------------------------------------------------------
    # Số tỉnh / nhóm
    # --------------------------------------------------------

    n_provinces = None

    if group_col in df.columns:
        n_provinces = (
            df[group_col]
            .dropna()
            .nunique()
        )

    # --------------------------------------------------------
    # Tần suất dữ liệu
    # --------------------------------------------------------

    frequency = "Không xác định"

    if date_col in df.columns:

        dates = pd.to_datetime(
            df[date_col],
            errors="coerce"
        ).dropna()

        if len(dates) >= 3:

            unique_dates = (
                dates
                .drop_duplicates()
                .sort_values()
            )

            diff_days = (
                unique_dates
                .diff()
                .dt.total_seconds()
                .div(86400)
                .dropna()
            )

            if not diff_days.empty:

                median_days = diff_days.median()

                if median_days <= 1.5:
                    frequency = "Hàng ngày"
                elif median_days <= 8:
                    frequency = "Hàng tuần"
                elif median_days <= 32:
                    frequency = "Hàng tháng"
                elif median_days <= 100:
                    frequency = "Hàng quý"
                else:
                    frequency = "Không xác định"

    return {
        "available": True,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "date_min": date_min,
        "date_max": date_max,
        "frequency": frequency,
        "n_provinces": n_provinces,
        "n_endog": len(
            st.session_state.get(
                "endogenous_vars",
                []
            )
        ),
        "n_exog": len(
            st.session_state.get(
                "exogenous_vars",
                []
            )
        ),
    }


def format_date(value):
    """Định dạng ngày để hiển thị trên Home."""

    if value is None:
        return "Chưa có dữ liệu"

    try:
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


# ============================================================
# HEADER
# ============================================================

st.title(APP_HEADER)

st.markdown(
    """
    **Đề tài:** Tìm hiểu mô hình DeepVARX và ứng dụng vào bài toán
    dự báo giá thu mua cà phê ở Việt Nam.
    """
)

st.markdown(
    """
    Hệ thống hỗ trợ quá trình nghiên cứu và thực nghiệm từ dữ liệu đầu vào,
    phân tích dữ liệu, phân tích chuỗi thời gian, lựa chọn biến, chuẩn bị dữ liệu,
    huấn luyện mô hình đến đánh giá và so sánh kết quả dự báo.
    """
)

st.divider()


# ============================================================
# 1. GIỚI THIỆU ĐỀ TÀI
# ============================================================

st.header("1. Giới thiệu đề tài")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Mục tiêu")

    st.markdown(
        """
        - Tìm hiểu cơ sở lý thuyết của mô hình **VAR, VARX, DeepVAR
          và DeepVARX**.
        - Phân tích các yếu tố có khả năng ảnh hưởng đến giá cà phê.
        - Xây dựng quy trình xử lý và chuẩn bị dữ liệu cho bài toán
          dự báo chuỗi thời gian đa biến.
        - Thực nghiệm và đánh giá khả năng dự báo của các mô hình.
        - So sánh kết quả giữa các phương pháp dự báo.
        """
    )


with col2:

    st.subheader("Phạm vi nghiên cứu")

    st.markdown(
        """
        - Đối tượng dự báo: **giá thu mua cà phê tại Việt Nam**.
        - Dữ liệu được tổ chức theo **thời gian và khu vực/tỉnh**.
        - Xem xét đồng thời các biến nội sinh và biến ngoại sinh.
        - Tập trung vào bài toán **dự báo chuỗi thời gian đa biến**.
        - Kết quả được đánh giá bằng các chỉ số sai số dự báo.
        """
    )


# ============================================================
# 2. PHẠM VI DỮ LIỆU
# ============================================================

st.header("2. Phạm vi dữ liệu")

st.caption(
    "Các thông tin dưới đây được lấy tự động từ dữ liệu đã được "
    "nạp/xử lý trong hệ thống, không nhập thủ công."
)

scope = get_data_scope()

if not scope["available"]:

    st.info(
        "Chưa có dữ liệu từ Trang 2. "
        "Hãy thực hiện bước nạp và phân tích dữ liệu trước "
        "để hệ thống tự động cập nhật phạm vi dữ liệu."
    )

    if st.button(
        "Đi đến Trang 2 →",
        use_container_width=False
    ):
        st.switch_page("pages/2_Phan_tich_tuong_quan.py")

else:

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Số quan sát",
            f"{scope['n_rows']:,}"
        )

    with c2:
        st.metric(
            "Số biến",
            f"{scope['n_cols']:,}"
        )

    with c3:
        st.metric(
            "Số tỉnh/khu vực",
            (
                f"{scope['n_provinces']:,}"
                if scope["n_provinces"] is not None
                else "—"
            )
        )

    with c4:
        st.metric(
            "Tần suất",
            scope["frequency"]
        )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    st.markdown("#### Khoảng thời gian dữ liệu")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Ngày bắt đầu",
            format_date(scope["date_min"])
        )

    with d2:
        st.metric(
            "Ngày kết thúc",
            format_date(scope["date_max"])
        )

    with d3:
        st.metric(
            "Biến nội sinh / ngoại sinh",
            f"{scope['n_endog']} / {scope['n_exog']}"
        )


# ============================================================
# 3. QUY TRÌNH NGHIÊN CỨU
# ============================================================

st.header("3. Quy trình nghiên cứu")

pages = [

    (
        "1. Dữ liệu & Tiền xử lý",
        "Nạp dữ liệu, kiểm tra cấu trúc, missing, outlier "
        "và thực hiện tiền xử lý.",
        "pages/1_Tai_du_lieu.py"
    ),

    (
        "2. Phân tích tương quan",
        "Pearson, Spearman, VIF, CCF và hỗ trợ lựa chọn "
        "biến nội sinh/ngoại sinh.",
        "pages/2_Phan_tich_tuong_quan.py"
    ),

    (
        "3. Phân tích chuỗi thời gian",
        "ADF, PP, KPSS, Johansen Cointegration "
        "và Granger Causality.",
        "pages/3_Phan_tich_chuoi_thoi_gian.py"
    ),

    (
        "4. Giới thiệu mô hình",
        "Trình bày lý thuyết và kiến trúc VAR, VARX, "
        "DeepVAR và DeepVARX.",
        "pages/4_Gioi_thieu_mo_hinh.py"
    ),

    (
        "5. Chuẩn bị & Thực nghiệm p,q",
        "Chuẩn hóa, tạo cửa sổ thời gian, chia dữ liệu "
        "Train/Validation/Test và tìm p,q.",
        "pages/5_Chuan_bi_thuc_nghiem.py"
    ),

    (
        "6. Huấn luyện mô hình",
        "Lựa chọn mô hình, cấu hình tham số, huấn luyện "
        "và theo dõi quá trình học.",
        "pages/6_Huan_luyen.py"
    ),

    (
        "7. Kết quả & So sánh",
        "Đánh giá MAE, RMSE, sMAPE, CV(RMSE), "
        "biểu đồ dự báo và so sánh mô hình.",
        "pages/7_Ket_qua.py"
    ),

    (
        "17. Sổ minh chứng",
        "Tập hợp các kết quả, biểu đồ, kiểm định và "
        "bằng chứng phục vụ việc truy vết kết luận.",
        "pages/17_So_minh_chung.py"
    ),
]


cols = st.columns(4)

for i, (name, desc, path) in enumerate(pages):

    with cols[i % 4]:

        with st.container(border=True):

            st.markdown(f"**{name}**")

            st.caption(desc)

            if st.button(
                "Mở trang →",
                key=f"nav_{i}",
                use_container_width=True
            ):
                st.switch_page(path)


# ============================================================
# 4. TRUY VẾT KẾT QUẢ
# ============================================================

st.divider()

col1, col2 = st.columns([3, 1])

with col1:

    st.subheader("Sổ minh chứng")

    st.markdown(
        """
        Các kết luận và kết quả thực nghiệm trong hệ thống được tổ chức
        theo hướng có thể **truy vết về dữ liệu, phép phân tích,
        kiểm định và kết quả thực nghiệm tương ứng**.
        """
    )


with col2:

    if st.button(
        "Mở Sổ minh chứng →",
        use_container_width=True
    ):
        st.switch_page("pages/17_So_minh_chung.py")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ☕ DeepVARX")

    st.caption(
        "Hệ thống dự báo giá thu mua cà phê Việt Nam"
    )

    st.divider()

    raw = st.session_state.get("raw_data")
    proc = st.session_state.get("processed_data")

    if raw is not None:

        st.success(
            f"Dữ liệu thô: "
            f"{raw.shape[0]:,} × {raw.shape[1]}"
        )

    else:

        st.warning("Chưa có dữ liệu thô")

    if proc is not None:

        st.success(
            f"Dữ liệu sau xử lý: "
            f"{proc.shape[0]:,} × {proc.shape[1]}"
        )

    if st.session_state.get("endogenous_vars"):

        st.info(
            "Endog: "
            + ", ".join(
                st.session_state["endogenous_vars"][:3]
            )
        )

    if st.session_state.get("exogenous_vars"):

        st.info(
            f"Exog: "
            f"{len(st.session_state['exogenous_vars'])} biến"
        )
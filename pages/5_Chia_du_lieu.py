import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessing import SCALER_OPTIONS, fit_transform_column, transform_with_fitted, time_based_split

st.set_page_config(page_title="Chia dữ liệu & chọn biến", layout="wide")
st.title("Chia dữ liệu & chọn biến")

df = st.session_state["processed_data"] if st.session_state["processed_data"] is not None \
    else st.session_state["raw_data"]

if df is None:
    st.warning("Chưa có dữ liệu. Vào trang **Tải dữ liệu** trước.")
    st.stop()

date_col = st.session_state["date_col"]
group_col = st.session_state["group_col"]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

if not date_col:
    st.error("Cần khai báo cột thời gian ở trang **Tải dữ liệu** trước khi chia dữ liệu theo thời gian.")
    st.stop()

st.subheader("1. Chọn biến nội sinh (endogenous) và ngoại sinh (exogenous)")
c1, c2 = st.columns(2)
with c1:
    endo = st.multiselect("Biến nội sinh (Y) — ví dụ: giá Robusta, Arabica",
                            numeric_cols, default=st.session_state["endogenous_vars"])
with c2:
    exo_options = [c for c in numeric_cols if c not in endo]
    exo = st.multiselect("Biến ngoại sinh (X) — ví dụ: tỷ giá, lượng mưa, giá dầu",
                           exo_options, default=[c for c in st.session_state["exogenous_vars"] if c in exo_options])

if group_col:
    st.info(f"Dữ liệu dạng panel theo cột **{group_col}**. Nếu tương quan chéo giữa các nhóm gần 1.0 "
            f"(xem trang EDA), cân nhắc chỉ chọn biến đại diện thay vì toàn bộ các nhóm.")

st.subheader("2. Bậc trễ (lag order)")
c3, c4 = st.columns(2)
with c3:
    p_lag = st.number_input("Bậc trễ nội sinh (p)", min_value=1, max_value=30, value=5)
with c4:
    q_lag = st.number_input("Bậc trễ ngoại sinh (q)", min_value=0, max_value=30, value=2)

st.subheader("3. Tỷ lệ chia Train / Validation / Test")
train_ratio = st.slider("Tỷ lệ Train", 0.5, 0.9, st.session_state["split_config"]["train_ratio"], 0.05)
val_ratio = st.slider("Tỷ lệ Validation", 0.05, 0.3, st.session_state["split_config"]["val_ratio"], 0.05)
test_ratio = round(1 - train_ratio - val_ratio, 2)
st.caption(f"Tỷ lệ Test suy ra: **{test_ratio:.2f}** — chia theo mốc thời gian (không shuffle), "
           f"đảm bảo test luôn là giai đoạn sau cùng, tránh rò rỉ dữ liệu tương lai.")
if test_ratio <= 0:
    st.error("Train + Val phải nhỏ hơn 1.0")
    st.stop()

st.subheader("4. Phương pháp chuẩn hóa áp dụng (lấy từ trang Tiền xử lý)")
methods_map = st.session_state.get("column_methods", {})
sel_vars = endo + exo
missing_methods = [v for v in sel_vars if v not in methods_map]
if missing_methods:
    st.warning(f"Các biến sau chưa được cấu hình phương pháp chuẩn hóa ở trang Tiền xử lý, "
               f"sẽ mặc định dùng Min-Max: {missing_methods}")
preview_table = pd.DataFrame({
    "Biến": sel_vars,
    "Loại": ["Nội sinh" if v in endo else "Ngoại sinh" for v in sel_vars],
    "Phương pháp chuẩn hóa": [methods_map.get(v, "Min-Max (0-1)") for v in sel_vars],
})
st.dataframe(preview_table, use_container_width=True)

if st.button("Thực hiện chia dữ liệu & chuẩn hóa (fit-on-train-only)", type="primary"):
    if not endo:
        st.error("Cần chọn ít nhất 1 biến nội sinh.")
        st.stop()

    train_df, val_df, test_df, cutoffs = time_based_split(df, date_col, train_ratio, val_ratio)

    fitted_scalers = {}
    train_out, val_out, test_out = train_df.copy(), val_df.copy(), test_df.copy()

    for v in sel_vars:
        method_label = methods_map.get(v, "Min-Max (0-1)")
        method = SCALER_OPTIONS[method_label]
        transformed_train, scaler = fit_transform_column(train_df[v], method)
        train_out[v] = transformed_train
        fitted_scalers[v] = {"method": method, "scaler": scaler}
        if len(val_df):
            val_out[v] = transform_with_fitted(val_df[v], method, scaler)
        if len(test_df):
            test_out[v] = transform_with_fitted(test_df[v], method, scaler)

    st.session_state["endogenous_vars"] = endo
    st.session_state["exogenous_vars"] = exo
    st.session_state["split_config"] = {"train_ratio": train_ratio, "val_ratio": val_ratio,
                                          "p_lag": p_lag, "q_lag": q_lag}
    st.session_state["train_df"] = train_out
    st.session_state["val_df"] = val_out
    st.session_state["test_df"] = test_out
    st.session_state["fitted_scalers"] = fitted_scalers

    st.success(
        f"Đã chia dữ liệu: Train={len(train_out)} dòng (đến {cutoffs['train_end']}), "
        f"Val={len(val_out)} dòng (đến {cutoffs['val_end']}), Test={len(test_out)} dòng. "
        f"Scaler đã fit CHỈ trên tập train."
    )
    st.dataframe(train_out[[date_col] + sel_vars].head(10), use_container_width=True)
    st.info("Tiếp tục sang trang **Huấn luyện mô hình**.")

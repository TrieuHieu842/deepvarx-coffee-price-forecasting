import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.plotting import forecast_vs_actual_chart, metric_comparison_bar, loss_curve_chart

st.set_page_config(page_title="Kết quả & So sánh", page_icon="📈", layout="wide")
st.title("Kết quả & So sánh mô hình")

results = st.session_state.get("results", {})
if not results:
    st.warning("Chưa có mô hình nào được huấn luyện. Vào trang **Huấn luyện mô hình** trước.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "Bảng so sánh RMSE/MAE/MAPE", "Biểu đồ dự báo vs thực tế",
    "Loss curve (LSTM/DeepVARX)", "So sánh theo tỉnh & Ablation",
])

# --- 1. Metrics table ----------------------------------------------------------
with tab1:
    rows = []
    for model_name, res in results.items():
        row = {"Mô hình": model_name}
        row.update(res["metrics"])
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    st.dataframe(metrics_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="lightgreen"),
                 use_container_width=True)

    metric_choice = st.selectbox("Chọn chỉ số để vẽ biểu đồ so sánh", ["RMSE", "MAE", "MAPE"])
    st.plotly_chart(metric_comparison_bar(metrics_df, metric_choice,
                                            title=f"So sánh {metric_choice} giữa các mô hình"),
                     use_container_width=True)

    best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Mô hình"]
    st.success(f"🏆 Mô hình có RMSE thấp nhất: **{best_model}**")
    st.caption("⚠️ Nhắc lại: kết quả hiện tại đến từ dữ liệu mô phỏng ở trang Huấn luyện (chưa nối "
               "code thật). Sau khi nối, ý nghĩa thống kê của mức vượt trội nên được kiểm tra bằng "
               "kiểm định Diebold-Mariano trước khi kết luận ở Ch.5.3.")

# --- 2. Forecast vs actual -------------------------------------------------------
with tab2:
    model_sel = st.selectbox("Chọn mô hình", list(results.keys()))
    res = results[model_sel]
    fig = forecast_vs_actual_chart(
        list(range(len(res["actual"]))), res["actual"], res["predicted"],
        title=f"{model_sel}: Dự báo vs Thực tế (tập test)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 3. Loss curve ------------------------------------------------------------
with tab3:
    models_with_loss = [m for m, r in results.items() if r.get("loss_curve")]
    if not models_with_loss:
        st.info("Chưa có mô hình nào (LSTM/DeepVARX) có loss curve được ghi lại.")
    else:
        model_sel2 = st.selectbox("Chọn mô hình", models_with_loss, key="loss_model_sel")
        lc = results[model_sel2]["loss_curve"]
        st.plotly_chart(loss_curve_chart(lc["train"], lc.get("val"),
                                           title=f"{model_sel2}: Loss theo epoch"),
                         use_container_width=True)
        if "bds_activated" in results[model_sel2]:
            st.caption(f"BDS test p-value = {results[model_sel2]['bds_pvalue']:.4f} → "
                       f"nhánh LSTM {'được kích hoạt' if results[model_sel2]['bds_activated'] else 'không kích hoạt'}.")

# --- 4. Province comparison + ablation ------------------------------------------
with tab4:
    st.subheader("So sánh hiệu suất theo tỉnh")
    group_col = st.session_state.get("group_col")
    if not group_col:
        st.info("Chưa khai báo cột tỉnh/nhóm ở trang Tải dữ liệu — không thể so sánh theo tỉnh. "
                "Khi nối code thật, hãy tính RMSE/MAE riêng cho từng tỉnh và hiển thị bảng/biểu đồ tại đây.")
    else:
        st.caption("Khung sẵn sàng — khi nối code huấn luyện thật theo từng tỉnh, lưu kết quả vào "
                   "`st.session_state['results'][model]['by_province']` (dict: tỉnh → {RMSE, MAE, MAPE}) "
                   "để bảng dưới đây tự động hiển thị.")
        for model_name, res in results.items():
            if "by_province" in res:
                st.markdown(f"**{model_name}**")
                st.dataframe(pd.DataFrame(res["by_province"]).T, use_container_width=True)

    st.divider()
    st.subheader("Ablation study — loại bỏ từng nhóm biến")
    st.caption(
        "Khung sẵn sàng cho việc so sánh hiệu suất khi bỏ nhóm biến thời tiết vs nhóm biến vĩ mô. "
        "Khi nối code huấn luyện thật, lưu mỗi cấu hình ablation vào "
        "`st.session_state['results']['DeepVARX_no_weather']`, `..._no_macro`, v.v. để bảng dưới "
        "tự tổng hợp so sánh."
    )
    ablation_models = [m for m in results.keys() if "no_" in m.lower() or "ablation" in m.lower()]
    if ablation_models:
        abl_rows = []
        for m in ablation_models:
            row = {"Cấu hình": m}
            row.update(results[m]["metrics"])
            abl_rows.append(row)
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True)
    else:
        st.info("Chưa có kết quả ablation nào được lưu.")

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui import setup_page

# Thiết lập cấu hình trang, không dùng icon rườm rà
setup_page("7 · Kiểm định Thống kê So sánh Mô hình", icon="")

st.title("Trang 7 · Kiểm định Thống kê So sánh Mô hình (Diebold–Mariano)")
st.caption(
    "So sánh độ chính xác dự báo giữa các mô hình trên tập kiểm thử 542 ngày (03/10/2024 – 30/10/2026) bằng các phép kiểm định thống kê chuẩn "
    "(Diebold–Mariano, Harvey-Leybourne-Newbold, Clark–West và Holm). Mục đích nhằm xác định xem việc một mô hình có sai số thấp hơn có thực sự phản ánh năng lực dự báo vượt trội hay chỉ do biến động ngẫu nhiên."
)

DATA_DIR = Path("statistical_evaluation/data/output")
INPUT_DIR = Path("statistical_evaluation/data/input")
SUMMARY_PATH = DATA_DIR / "dm_summary_results.csv"
MODEL_REP_PATH = DATA_DIR / "model_discovery_report.csv"
FORECAST_PATH = INPUT_DIR / "all_models_forecasts_test_542days.csv"

# Phân loại nhóm mô hình rõ ràng, từ ngữ dễ hiểu
MODEL_GROUPS = {
    # Nhóm học sâu có biến ngoại sinh
    "DeepVARX 3": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "TimeXer Dual-Stream (20 biến)"},
    "DeepVARX 2": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "TimeXer Dual-Stream (11 biến)"},
    "DeepVAR 2": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "DeepVAR (11 biến)"},
    "DeepVAR 3": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "DeepVAR (20 biến)"},
    "LSTM 2": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "LSTM (11 biến)"},
    "LSTM 3": {"group": "Học sâu có biến ngoại sinh", "color": "#1f77b4", "notes": "LSTM (20 biến)"},

    # Nhóm kinh tế lượng có biến ngoại sinh
    "VARX-L 2 (Ridge)": {"group": "Kinh tế lượng có biến ngoại sinh", "color": "#2ca02c", "notes": "VARX chuẩn hóa Ridge (11 biến)"},
    "VARX 2": {"group": "Kinh tế lượng có biến ngoại sinh", "color": "#2ca02c", "notes": "VARX (11 biến)"},
    "VARX 3": {"group": "Kinh tế lượng có biến ngoại sinh", "color": "#2ca02c", "notes": "VARX (20 biến)"},
    "VARX 1": {"group": "Kinh tế lượng có biến ngoại sinh", "color": "#2ca02c", "notes": "VARX (3 biến)"},

    # Nhóm mô hình chỉ dùng chuỗi giá quá khứ
    "DeepVARX 1": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "DeepVARX không biến ngoại sinh"},
    "DeepVAR 1": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "DeepVAR không biến ngoại sinh"},
    "LSTM 1": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "LSTM không biến ngoại sinh"},
    "VAR(5)": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "VAR 5 bậc trễ"},
    "VAR(2)": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "VAR 2 bậc trễ"},
    "VAR(1)": {"group": "Chỉ dùng chuỗi giá quá khứ", "color": "#7f7f7f", "notes": "VAR 1 bậc trễ"},
    "Naive (Random Walk)": {"group": "Tham chiếu cơ sở", "color": "#bcbd22", "notes": "Giá ngày hôm trước"},
}


def _get_files_mtime():
    """Lấy thời gian sửa đổi gần nhất của các tệp dữ liệu để tự động làm mới bộ nhớ đệm."""
    try:
        t1 = SUMMARY_PATH.stat().st_mtime if SUMMARY_PATH.exists() else 0
        t2 = MODEL_REP_PATH.stat().st_mtime if MODEL_REP_PATH.exists() else 0
        t3 = FORECAST_PATH.stat().st_mtime if FORECAST_PATH.exists() else 0
        return (t1, t2, t3)
    except Exception:
        return 0


@st.cache_data
def load_evaluation_data(_mtime):
    """Nạp dữ liệu kiểm định đã tính toán sẵn."""
    summary_df = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else None
    model_rep_df = pd.read_csv(MODEL_REP_PATH) if MODEL_REP_PATH.exists() else None
    fc_df = pd.read_csv(FORECAST_PATH) if FORECAST_PATH.exists() else None
    return summary_df, model_rep_df, fc_df


summary_df, model_rep_df, fc_df = load_evaluation_data(_get_files_mtime())

# Thanh trạng thái chung
with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 3])
    n_models = model_rep_df["model_name"].nunique() if model_rep_df is not None else 17
    n_pairs = len(summary_df) if summary_df is not None else 1148
    n_obs = 542
    n_provs = summary_df["province"].nunique() if (summary_df is not None and "province" in summary_df.columns) else 4

    c1.metric("Số mô hình", f"{n_models}")
    c2.metric("Số cặp so sánh", f"{n_pairs:,}")
    c3.metric("Số ngày kiểm thử", f"{n_obs} ngày")
    c4.metric("Khu vực", f"{n_provs} tỉnh")

    with c5:
        st.write("")
        if st.button("Chạy lại kiểm định DM", type="primary", use_container_width=True):
            with st.spinner("Đang tính toán lại các phép kiểm định..."):
                proc = subprocess.run([sys.executable, "statistical_evaluation/run_dm_test.py"], capture_output=True, text=True)
                if proc.returncode == 0:
                    st.cache_data.clear()
                    st.success("Đã hoàn thành và cập nhật kết quả mới.")
                    st.rerun()
                else:
                    st.error(f"Lỗi: {proc.stderr[:300]}")

if summary_df is None or model_rep_df is None:
    st.warning("Chưa có tệp kết quả kiểm định. Nhấn nút 'Chạy lại kiểm định DM' ở trên để sinh dữ liệu.")
    st.stop()

PROV_VN_NAMES = {
    "dak_lak": "Đắk Lắk",
    "lam_dong": "Lâm Đồng",
    "gia_lai": "Gia Lai",
    "dak_nong": "Đắk Nông",
}
avail_provinces = [p for p in ["dak_lak", "lam_dong", "gia_lai", "dak_nong"] if p in summary_df["province"].unique()]
if not avail_provinces:
    avail_provinces = ["dak_lak", "lam_dong", "gia_lai", "dak_nong"]

def fmt_p_val(p):
    if pd.isna(p) or p != p:
        return "–"
    if p < 0.0001:
        return "< 0.0001"
    return f"{p:.4f}"

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Tổng quan & Xếp hạng",
    "2. So sánh trực tiếp 2 mô hình",
    "3. Ma trận so sánh toàn bộ các cặp",
    "4. Kiểm tra mô hình lồng nhau (Clark–West)",
    "5. Đánh giá dự báo xác suất (CRPS)",
    "6. Xuất dữ liệu báo cáo",
])

# =============================================================================== TAB 1
with tab1:
    st.subheader("1. Tổng quan Năng lực và Xếp hạng Mô hình")
    st.caption("So sánh sai số dự báo của từng mô hình so với mô hình tham chiếu cơ sở Naive (Random Walk).")

    t1_c1, t1_c2 = st.columns(2)
    prov_choice = t1_c1.selectbox("Khu vực đánh giá", avail_provinces, format_func=lambda x: PROV_VN_NAMES.get(x, x), key="tab1_prov")
    loss_choice = t1_c2.selectbox("Hàm mất mát", ["mse", "mae"], format_func=lambda x: "Sai số bình phương (MSE)" if x == "mse" else "Sai số tuyệt đối (MAE)", key="tab1_loss")

    # Lọc so sánh với Naive
    sub_naive = summary_df[
        (summary_df["province"] == prov_choice) &
        (summary_df["loss_function"] == loss_choice) &
        (summary_df["model_b"] == "Naive (Random Walk)")
    ].copy()

    rep_sub = model_rep_df.copy()
    naive_rmse = rep_sub.loc[rep_sub["model_name"] == "Naive (Random Walk)", "test_rmse_vnd"].iloc[0]
    naive_mae = rep_sub.loc[rep_sub["model_name"] == "Naive (Random Walk)", "test_mae_vnd"].iloc[0]
    base_val = naive_rmse if loss_choice == "mse" else naive_mae

    t1_rows = []
    # Dòng Naive
    t1_rows.append({
        "Mô hình": "Naive (Random Walk)",
        "Nhóm mô hình": "Tham chiếu cơ sở",
        "RMSE (VNĐ/kg)": naive_rmse,
        "MAE (VNĐ/kg)": naive_mae,
        "Chênh lệch so với Naive": 0.0,
        "Mức giảm sai số (%)": 0.0,
        "Thống kê HLN": "–",
        "p-value gốc (HLN)": "–",
        "p-value hiệu chỉnh (Holm)": "–",
        "Kết luận kiểm định (α = 0.05)": "Mô hình cơ sở tham chiếu",
    })

    for _, row in sub_naive.iterrows():
        m_name = row["model_a"]
        m_info = rep_sub[rep_sub["model_name"] == m_name]
        m_rmse = m_info["test_rmse_vnd"].iloc[0] if not m_info.empty else np.nan
        m_mae = m_info["test_mae_vnd"].iloc[0] if not m_info.empty else np.nan
        cur_val = m_rmse if loss_choice == "mse" else m_mae
        diff_val = base_val - cur_val
        pct_gain = (diff_val / base_val) * 100

        verdict = "Chưa đủ bằng chứng khác biệt (p >= 0.05)"
        if row["is_statistically_significant"]:
            if row["better_model_statistical"] == m_name:
                sig_level = "p < 0.01" if row["holm_p_value"] < 0.01 else "p < 0.05"
                verdict = f"Sai số thấp hơn có ý nghĩa ({sig_level})"
            else:
                verdict = "Sai số cao hơn (p < 0.05)"

        g_info = MODEL_GROUPS.get(m_name, {"group": "Khác"})
        t1_rows.append({
            "Mô hình": m_name,
            "Nhóm mô hình": g_info["group"],
            "RMSE (VNĐ/kg)": m_rmse,
            "MAE (VNĐ/kg)": m_mae,
            "Chênh lệch so với Naive": diff_val,
            "Mức giảm sai số (%)": pct_gain,
            "Thống kê HLN": f"{row['hln_stat']:+.3f}",
            "p-value gốc (HLN)": fmt_p_val(row["hln_p_value"]),
            "p-value hiệu chỉnh (Holm)": fmt_p_val(row["holm_p_value"]),
            "Kết luận kiểm định (α = 0.05)": verdict,
        })

    t1_df = pd.DataFrame(t1_rows)
    # Sắp xếp theo mức giảm sai số từ cao xuống thấp
    t1_df = t1_df.sort_values("Mức giảm sai số (%)", ascending=False).reset_index(drop=True)
    t1_df.insert(0, "Hạng", [f"Hạng {i+1}" if r["Mô hình"] != "Naive (Random Walk)" else "Cơ sở" for i, r in t1_df.iterrows()])

    # Hiển thị bảng
    st.dataframe(
        t1_df.style.format({
            "RMSE (VNĐ/kg)": "{:,.1f}",
            "MAE (VNĐ/kg)": "{:,.1f}",
            "Chênh lệch so với Naive": "{:+,.1f}",
            "Mức giảm sai số (%)": "{:+.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Biểu đồ thanh ngang thể hiện mức giảm sai số %
    st.write("")
    plot_df = t1_df[t1_df["Mô hình"] != "Naive (Random Walk)"].sort_values("Mức giảm sai số (%)", ascending=True)

    fig_bar = px.bar(
        plot_df,
        x="Mức giảm sai số (%)",
        y="Mô hình",
        color="Nhóm mô hình",
        orientation="h",
        color_discrete_map={
            "Học sâu có biến ngoại sinh": "#1f77b4",
            "Kinh tế lượng có biến ngoại sinh": "#2ca02c",
            "Chỉ dùng chuỗi giá quá khứ": "#7f7f7f",
        },
        labels={"Mức giảm sai số (%)": "Mức giảm sai số so với Naive (%)", "Mô hình": "Mô hình", "Nhóm mô hình": "Nhóm mô hình"},
        height=480,
    )
    fig_bar.add_vline(x=0.0, line_dash="dash", line_color="black")
    fig_bar.update_layout(
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "Chú thích: Thanh màu xanh dương thể hiện các mô hình học sâu có biến ngoại sinh; màu xanh lá thể hiện mô hình kinh tế lượng có biến ngoại sinh; "
        "màu xám thể hiện các mô hình chỉ dùng chuỗi giá quá khứ (không có biến ngoại sinh)."
    )

# =============================================================================== TAB 2
with tab2:
    st.subheader("2. So sánh Trực tiếp Giữa 2 Mô hình")
    st.caption("Chọn 2 mô hình bất kỳ để kiểm tra xem mô hình nào có độ chính xác cao hơn và sự khác biệt đó có ý nghĩa thống kê hay không.")

    all_names = sorted(model_rep_df["model_name"].unique())
    c_m1, c_m2, c_prov, c_loss = st.columns([3, 3, 2, 2])

    m1_pick = c_m1.selectbox("Mô hình 1", all_names, index=all_names.index("DeepVARX 3") if "DeepVARX 3" in all_names else 0)
    remain_names = [m for m in all_names if m != m1_pick]
    m2_pick = c_m2.selectbox("Mô hình 2 (Đối chứng)", remain_names, index=remain_names.index("VAR(5)") if "VAR(5)" in remain_names else 0)
    prov_pick = c_prov.selectbox("Khu vực", avail_provinces, format_func=lambda x: PROV_VN_NAMES.get(x, x), key="tab2_prov")
    loss_pick = c_loss.selectbox("Hàm mất mát", ["mse", "mae"], format_func=lambda x: "Bình phương (MSE)" if x == "mse" else "Tuyệt đối (MAE)", key="tab2_loss")

    # Tìm cặp trong dữ liệu kiểm định
    pair_match = summary_df[
        (summary_df["province"] == prov_pick) &
        (summary_df["loss_function"] == loss_pick) &
        (((summary_df["model_a"] == m1_pick) & (summary_df["model_b"] == m2_pick)) |
         ((summary_df["model_a"] == m2_pick) & (summary_df["model_b"] == m1_pick)))
    ]

    info1 = model_rep_df[model_rep_df["model_name"] == m1_pick].iloc[0]
    info2 = model_rep_df[model_rep_df["model_name"] == m2_pick].iloc[0]

    with st.container(border=True):
        col1, col_center, col2 = st.columns([3, 4, 3])

        with col1:
            st.markdown(f"**Mô hình 1: {m1_pick}**")
            st.caption(f"Nhóm: {MODEL_GROUPS.get(m1_pick, {}).get('group', 'Khác')}")
            st.metric("Sai số RMSE", f"{info1['test_rmse_vnd']:,.1f} VNĐ/kg")
            st.metric("Sai số MAE", f"{info1['test_mae_vnd']:,.1f} VNĐ/kg")

        with col2:
            st.markdown(f"**Mô hình 2: {m2_pick}**")
            st.caption(f"Nhóm: {MODEL_GROUPS.get(m2_pick, {}).get('group', 'Khác')}")
            st.metric("Sai số RMSE", f"{info2['test_rmse_vnd']:,.1f} VNĐ/kg")
            st.metric("Sai số MAE", f"{info2['test_mae_vnd']:,.1f} VNĐ/kg")

        with col_center:
            st.markdown("**Kết quả kiểm định Diebold–Mariano (HLN 1997)**")
            if not pair_match.empty:
                p_row = pair_match.iloc[0]
                is_m1_a = (p_row["model_a"] == m1_pick)
                hln_stat = p_row["hln_stat"] if is_m1_a else -p_row["hln_stat"]
                raw_p = p_row["hln_p_value"]
                holm_p = p_row["holm_p_value"]
                n_obs = int(p_row["n_observations"]) if "n_observations" in p_row else 542
                df_deg = n_obs - 1

                rmse_diff = info2["test_rmse_vnd"] - info1["test_rmse_vnd"]
                mae_diff = info2["test_mae_vnd"] - info1["test_mae_vnd"]
                diff_show = rmse_diff if loss_pick == "mse" else mae_diff
                loss_name_str = "RMSE" if loss_pick == "mse" else "MAE"

                # Xác định mô hình có sai số thực nghiệm thấp hơn trên tập test
                if abs(diff_show) < 1e-4:
                    better_sample = "Equal"
                elif diff_show > 0:
                    better_sample = m1_pick
                else:
                    better_sample = m2_pick

                # Hiển thị số liệu thống kê đầy đủ, minh bạch
                s_col1, s_col2 = st.columns(2)
                s_col1.metric("Thống kê HLN (1997)", f"{hln_stat:+.3f}", help=f"Phân phối Student-t với bậc tự do df = {df_deg} (T = {n_obs} ngày).")
                s_col2.metric("p-value gốc (Cặp trực tiếp)", fmt_p_val(raw_p), help=f"p-value 2 phía trực tiếp từ t({df_deg}), chưa hiệu chỉnh đa kiểm định.")

                st.markdown(f"**p-value hiệu chỉnh Holm:** `{fmt_p_val(holm_p)}` *(Kiểm soát FWER cho 136 cặp kiểm định)*")

                # Diễn giải kết luận khoa học chính xác
                if better_sample == m1_pick:
                    pct = (rmse_diff / info2["test_rmse_vnd"]) * 100
                    lead_txt = f"**{m1_pick} có sai số thực tế thấp hơn {m2_pick}** ({abs(rmse_diff):,.1f} VNĐ/kg, giảm {abs(pct):.1f}% RMSE)."
                elif better_sample == m2_pick:
                    pct = (-rmse_diff / info1["test_rmse_vnd"]) * 100
                    lead_txt = f"**{m2_pick} có sai số thực tế thấp hơn {m1_pick}** ({abs(rmse_diff):,.1f} VNĐ/kg, giảm {abs(pct):.1f}% RMSE)."
                else:
                    lead_txt = f"**{m1_pick} và {m2_pick} có sai số thực tế bằng nhau.**"

                if holm_p < 0.05:
                    st.success(
                        f"{lead_txt}\n\n"
                        f"✓ **Bác bỏ H₀ ở mức ý nghĩa 5% (α = 0.05)** cả trong kiểm định cặp trực tiếp (`p_raw = {fmt_p_val(raw_p)}`) "
                        f"lẫn sau khi kiểm soát FWER trên 136 cặp (`p_Holm = {fmt_p_val(holm_p)}`). "
                        f"Có đầy đủ cơ sở thống kê khẳng định năng lực dự báo của **{better_sample}** vượt trội hơn."
                    )
                elif raw_p < 0.05 and holm_p < 0.10:
                    st.warning(
                        f"{lead_txt}\n\n"
                        f"• **Kiểm định cặp trực tiếp:** `p_raw = {fmt_p_val(raw_p)} < 0.01` — Bác bỏ H₀ ở mức ý nghĩa 1% và 5% khi kiểm tra độc lập giữa 2 mô hình.\n\n"
                        f"• **Kiểm soát đa kiểm định (Holm 136 cặp):** `p_Holm = {fmt_p_val(holm_p)}` — Bác bỏ H₀ ở mức ý nghĩa 10% (α = 0.10), nhưng chưa đủ bằng chứng ở mức khắt khe 5% (α = 0.05).\n\n"
                        f"🔍 **Diễn giải học thuật:** Mặc dù {better_sample} có sai số thấp hơn và vượt trội trong kiểm định cặp trực tiếp, "
                        f"khi áp dụng quy tắc Holm kiểm soát Family-Wise Error Rate cho toàn bộ 136 cặp kiểm định, bạn chưa đủ cơ sở thống kê để khẳng định vượt trội ở mức ý nghĩa 5% (nhưng có ý nghĩa ở mức 10%)."
                    )
                elif raw_p < 0.05 and holm_p >= 0.10:
                    st.info(
                        f"{lead_txt}\n\n"
                        f"• **Kiểm định cặp trực tiếp:** `p_raw = {fmt_p_val(raw_p)} < 0.05` (có ý nghĩa thống kê khi so sánh độc lập).\n\n"
                        f"• **Kiểm soát đa kiểm định (Holm 136 cặp):** `p_Holm = {fmt_p_val(holm_p)} ≥ 0.10` — Không còn giữ được ý nghĩa thống kê sau khi hiệu chỉnh bảo thủ cho toàn bộ họ 136 cặp.\n\n"
                        f"🔍 **Diễn giải học thuật:** Chưa đủ cơ sở để khẳng định sự vượt trội khi xét trong tổng thể toàn bộ các cặp mô hình."
                    )
                else:
                    st.info(
                        f"{lead_txt}\n\n"
                        f"• **Kết luận:** Chưa đủ bằng chứng thống kê để bác bỏ H₀ ở mức ý nghĩa 5% (`p_raw = {fmt_p_val(raw_p)} ≥ 0.05`). "
                        f"Chênh lệch sai số thực nghiệm có thể chỉ do biến động ngẫu nhiên."
                    )

                st.caption(
                    "📌 **Nguyên tắc kiểm định giả thuyết:** H₀: Hai mô hình có độ chính xác dự báo tương đương; "
                    "H₁: Hai mô hình có độ chính xác dự báo khác nhau. Khi p ≥ 0.05, về mặt thống kê ta chỉ kết luận "
                    "'chưa đủ bằng chứng để khẳng định có sự khác biệt', tuyệt đối không diễn giải thành 'hai mô hình có độ chính xác bằng nhau'."
                )
            else:
                st.info("Chưa có dữ liệu kiểm định cho cặp mô hình này.")

    # Biểu đồ đường dự báo thực tế
    if fc_df is not None:
        sub_fa = fc_df[(fc_df["model"] == m1_pick) & (fc_df["province"] == prov_pick)].sort_values("target_date")
        sub_fb = fc_df[(fc_df["model"] == m2_pick) & (fc_df["province"] == prov_pick)].sort_values("target_date")
        if not sub_fa.empty and not sub_fb.empty:
            fig_lines = go.Figure()
            fig_lines.add_trace(go.Scatter(x=pd.to_datetime(sub_fa["target_date"]), y=sub_fa["actual_price"], name="Giá thực tế", line=dict(color="black", width=2)))
            fig_lines.add_trace(go.Scatter(x=pd.to_datetime(sub_fa["target_date"]), y=sub_fa["predicted_price"], name=f"Dự báo {m1_pick}", line=dict(color="#1f77b4", width=1.5)))
            fig_lines.add_trace(go.Scatter(x=pd.to_datetime(sub_fb["target_date"]), y=sub_fb["predicted_price"], name=f"Dự báo {m2_pick}", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
            fig_lines.update_layout(
                title=f"Đường dự báo đối chiếu: {m1_pick} và {m2_pick} ({prov_pick.title()})",
                yaxis_title="VNĐ/kg",
                height=400,
                margin=dict(t=40, b=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig_lines, use_container_width=True)

# =============================================================================== TAB 3
with tab3:
    st.subheader("3. Ma trận So sánh Toàn bộ các Cặp Mô hình")
    st.caption(
        "Bảng tổng hợp mức ý nghĩa thống kê giữa mọi cặp mô hình. Tam giác dưới thể hiện giá trị thống kê HLN, tam giác trên thể hiện p-value hiệu chỉnh. "
        "Cách đọc: p-value < 0.05 (có dấu **) nghĩa là sự khác biệt có ý nghĩa thống kê ở mức 95%."
    )

    m3_col1, m3_col2, m3_col3, m3_col4 = st.columns([2, 1.5, 1.5, 2.5])
    mat_type = m3_col1.radio("Nhóm mô hình hiển thị", ["6 mô hình tiêu biểu", "Toàn bộ 17 mô hình"], horizontal=True)
    t3_prov = m3_col2.selectbox("Khu vực", avail_provinces, format_func=lambda x: PROV_VN_NAMES.get(x, x), key="tab3_prov")
    t3_loss = m3_col3.selectbox("Hàm mất mát", ["mse", "mae"], key="tab3_loss")
    p_type_choice = m3_col4.radio("Tam giác trên hiển thị", ["p-value hiệu chỉnh (Holm)", "p-value gốc (HLN)"], horizontal=True)
    is_holm = p_type_choice.startswith("p-value hiệu chỉnh")

    key_models = ["Naive (Random Walk)", "VAR(5)", "VARX 2", "LSTM 2", "DeepVAR 2", "DeepVARX 3"]
    mat_models = key_models if mat_type.startswith("6") else sorted(model_rep_df["model_name"].unique())

    mat_stat = pd.DataFrame(index=mat_models, columns=mat_models, data="")
    mat_p_num = pd.DataFrame(index=mat_models, columns=mat_models, data=np.nan)

    for i, ma in enumerate(mat_models):
        for j, mb in enumerate(mat_models):
            if i == j:
                mat_stat.loc[ma, mb] = "–"
                mat_p_num.loc[ma, mb] = 1.0
            else:
                pair = summary_df[
                    (summary_df["province"] == t3_prov) &
                    (summary_df["loss_function"] == t3_loss) &
                    (((summary_df["model_a"] == ma) & (summary_df["model_b"] == mb)) |
                     ((summary_df["model_a"] == mb) & (summary_df["model_b"] == ma)))
                ]
                if not pair.empty:
                    rp = pair.iloc[0]
                    use_p = rp["holm_p_value"] if is_holm else rp["hln_p_value"]
                    if i > j:
                        s_val = rp["hln_stat"] if rp["model_a"] == ma else -rp["hln_stat"]
                        mat_stat.loc[ma, mb] = f"{s_val:+.2f}"
                        mat_p_num.loc[ma, mb] = use_p
                    else:
                        star = " ***" if use_p < 0.01 else (" **" if use_p < 0.05 else (" *" if use_p < 0.1 else ""))
                        mat_stat.loc[ma, mb] = fmt_p_val(use_p) + star
                        mat_p_num.loc[ma, mb] = use_p

    st.dataframe(mat_stat, use_container_width=True)
    p_label_str = "p-value hiệu chỉnh Holm" if is_holm else "p-value gốc (HLN)"

    fig_heatmap = px.imshow(
        mat_p_num.values,
        x=mat_models,
        y=mat_models,
        color_continuous_scale="Blues_r",
        labels=dict(x="Mô hình B", y="Mô hình A", color="p-value"),
        title=f"Bản đồ nhiệt {p_label_str} ({PROV_VN_NAMES.get(t3_prov, t3_prov)})",
    )
    fig_heatmap.update_layout(height=480, margin=dict(t=40, b=80))
    st.plotly_chart(fig_heatmap, use_container_width=True)

# =============================================================================== TAB 4
with tab4:
    st.subheader("4. Kiểm định Mô hình Lồng nhau (Clark–West)")
    st.caption(
        "Khi so sánh một mô hình lớn (được bổ sung thêm biến ngoại sinh) với mô hình con (chỉ dùng giá quá khứ), "
        "kiểm định DM thông thường dễ bị sai lệch do mô hình lớn phải ước lượng nhiều tham số hơn. "
        "Kiểm định Clark–West (2007) hiệu chỉnh phần sai lệch này để đánh giá chính xác các biến bổ sung có thực sự giúp tăng độ chính xác hay không."
    )

    t4_col1, t4_col2 = st.columns(2)
    t4_prov = t4_col1.selectbox("Khu vực", avail_provinces, format_func=lambda x: PROV_VN_NAMES.get(x, x), key="tab4_prov")
    t4_loss = t4_col2.selectbox("Hàm mất mát", ["mse", "mae"], key="tab4_loss")

    nested_sub = summary_df[
        (summary_df["is_nested"] == True) &
        (summary_df["province"] == t4_prov) &
        (summary_df["loss_function"] == t4_loss)
    ].copy()

    if not nested_sub.empty:
        nested_df = pd.DataFrame({
            "Mô hình cơ sở": nested_sub["nested_restricted_model"],
            "Mô hình bổ sung biến": nested_sub["nested_unrestricted_model"],
            "Thống kê Clark–West": nested_sub["clark_west_stat"],
            "p-value (1 phía)": nested_sub["clark_west_p_value"].apply(fmt_p_val),
            "Kết luận": nested_sub["clark_west_p_value"].apply(lambda p: "Biến bổ sung giúp giảm sai số (p < 0.01)" if p < 0.01 else ("Biến bổ sung giúp giảm sai số (p < 0.05)" if p < 0.05 else "Chưa có ý nghĩa thống kê")),
        })
        st.dataframe(
            nested_df.style.format({"Thống kê Clark–West": "{:+.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Tổng số cặp mô hình lồng nhau được kiểm định: {len(nested_df)} cặp.")
    else:
        st.info("Không có dữ liệu cho các cặp mô hình lồng nhau theo bộ lọc hiện tại.")

# =============================================================================== TAB 5
with tab5:
    st.subheader("5. Đánh giá Dự báo Xác suất (CRPS)")
    st.caption(
        "Chỉ số Continuous Ranked Probability Score (CRPS) đánh giá độ chính xác của toàn bộ khoảng dự báo (thay vì chỉ đo 1 điểm giá trung bình). "
        "Điểm CRPS càng thấp, mô hình dự báo khoảng biến động giá càng chuẩn xác. Áp dụng cho các mô hình học sâu xác suất DeepVAR và DeepVARX."
    )

    t5_prov = st.selectbox("Khu vực", avail_provinces, format_func=lambda x: PROV_VN_NAMES.get(x, x), key="tab5_prov")

    crps_sub = summary_df[
        (summary_df["loss_function"] == "crps") &
        (summary_df["province"] == t5_prov)
    ].copy()

    if not crps_sub.empty:
        crps_disp = pd.DataFrame({
            "Mô hình A": crps_sub["model_a"],
            "Mô hình B": crps_sub["model_b"],
            "CRPS trung bình A": crps_sub["mean_loss_a"],
            "CRPS trung bình B": crps_sub["mean_loss_b"],
            "Chênh lệch CRPS": crps_sub["mean_loss_diff"],
            "Thống kê HLN": crps_sub["hln_stat"],
            "p-value gốc (HLN)": crps_sub["hln_p_value"].apply(fmt_p_val),
            "p-value hiệu chỉnh (Holm)": crps_sub["holm_p_value"].apply(fmt_p_val),
            "Mô hình tốt hơn về xác suất": crps_sub["better_model_statistical"].apply(lambda x: x if (x != "Equal" and not str(x).startswith("None")) else "Chưa đủ bằng chứng khác biệt (p >= 0.05)"),
        })

        st.dataframe(
            crps_disp.style.format({
                "CRPS trung bình A": "{:,.1f}",
                "CRPS trung bình B": "{:,.1f}",
                "Chênh lệch CRPS": "{:+.1f}",
                "Thống kê HLN": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Chưa có kết quả kiểm định CRPS cho khu vực này.")

# =============================================================================== TAB 6
with tab6:
    st.subheader("6. Tải xuống Dữ liệu Kết quả")
    st.caption("Tải các tệp dữ liệu chi tiết phục vụ lưu trữ, kiểm toán và đưa vào phụ lục luận văn:")

    d1, d2, d3 = st.columns(3)

    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "rb") as f:
            n_p_lbl = f"{len(summary_df):,}" if summary_df is not None else "1,148"
            d1.download_button(f"Tải dm_summary_results.csv ({n_p_lbl} cặp)", data=f.read(), file_name="dm_summary_results.csv", mime="text/csv", use_container_width=True)

    if MODEL_REP_PATH.exists():
        with open(MODEL_REP_PATH, "rb") as f:
            n_m_lbl = f"{n_models}" if model_rep_df is not None else "17"
            d2.download_button(f"Tải model_discovery_report.csv ({n_m_lbl} mô hình)", data=f.read(), file_name="model_discovery_report.csv", mime="text/csv", use_container_width=True)

    if FORECAST_PATH.exists():
        with open(FORECAST_PATH, "rb") as f:
            d3.download_button("Tải all_models_forecasts_test_542days.csv", data=f.read(), file_name="all_models_forecasts_test_542days.csv", mime="text/csv", use_container_width=True)

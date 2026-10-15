import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_utils import difference
from modules.feature_selection import xgb_importance
from modules.stat_tests import ar_residuals, bds_residuals, bds_with_exog, ljung_box, variance_ratio
from modules.ui import fmt_p, require, setup_page, step_box

setup_page("3 · Khám phá phi tuyến")
st.title("Trang 3 · Khám phá phi tuyến (Luồng 2)")
st.caption("Nới lỏng giả định tuyến tính để tìm đặc trưng mà mạng nơ-ron có thể khai thác. Dùng **XGBoost** làm thước đo "
           "tầm quan trọng. Kết quả được lưu làm bằng chứng cho **Trang 4**.")
require(["df_clean", "d", "evidence_lin"], "Cần hoàn tất **Trang 1** và **Trang 2** (chốt bậc d, mở trang để lưu bằng chứng) trước.")

df = st.session_state["df_clean"]
panel = st.session_state["df_clean_panel"]
pm = st.session_state["panel_meta"]
target = st.session_state["target"]
xcols = st.session_state["x_cols"]
train_n = st.session_state["train_n"]
train_end = st.session_state["train_end"]
d = st.session_state["d"]
d_eff = max(d, 1)
gr_ok = st.session_state["evidence_lin"].set_index("Biến")["Granger đạt"]

tab1, tab2, tab3 = st.tabs(["① Động lực phi tuyến (BDS)", "② Tầm quan trọng XGBoost", "③ Bước ngẫu nhiên (Ljung-Box / VR)"])


@st.cache_data(show_spinner="Đang chạy BDS theo từng biến…")
def run_bds_vars(dtr: pd.DataFrame, target_: str, xcols_: list, p: int, max_dim: int) -> pd.DataFrame:
    rows, y = [], dtr[target_].to_numpy()
    for c in xcols_:
        try:
            pv = bds_with_exog(y, dtr[c].to_numpy(), p, max_dim)
        except ImportError:
            pv = np.nan
        rows.append({"Biến": c, "p BDS (phần dư Y~Y,X)": pv})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Đang huấn luyện XGBoost…")
def run_xgb(panel_, date_col_, entity_col_, target_, xcols_, d_, train_end_, n_lags_, scope_, proxy_df_):
    if scope_ == "proxy":
        frames = [difference(proxy_df_[[target_] + xcols_], d_).iloc[:int((proxy_df_.index <= train_end_).sum()) - d_]]
    else:
        frames = []
        for _, g in panel_.groupby(entity_col_, observed=True, sort=False):
            g = g.set_index(date_col_).sort_index()[[target_] + xcols_]
            frames.append(difference(g[g.index <= train_end_], d_))
    return xgb_importance(frames, target_, xcols_, n_lags_)


# =============================================================================== TAB 1
with tab1:
    step_box("Kiểm định <b>BDS</b> trên phần dư của mô hình tuyến tính (TRAIN, sai phân, tỉnh đại diện). \n Bác bỏ i.i.d. ⇒ còn "
             "phụ thuộc phi tuyến mà mô hình tuyến tính bỏ sót. \n (a) phần dư AR của Y; \n (b) <b>theo từng biến</b>: phần dư của Y~(Y, X<sub>i</sub>).")
    c1, c2, c3 = st.columns(3)
    ar_max = c1.number_input("Bậc AR tối đa", 1, 30, 10)
    max_dim = c2.number_input("Chiều nhúng tối đa", 3, 8, 5)
    a_bds = c3.select_slider("Mức ý nghĩa", [0.01, 0.05, 0.10], value=0.05)
    dtr = difference(df[[target] + xcols], d_eff).iloc[:train_n - d_eff]
    resid, p_ar = ar_residuals(dtr[target].to_numpy(), int(ar_max))
    st.markdown(f"**(a) Phần dư AR({p_ar}) của Δ{d_eff}·{target}**, n = {len(resid):,}")
    try:
        bt = bds_residuals(resid, int(max_dim))
        st.dataframe(bt.style.format({"Thống kê BDS": "{:.3f}", "p-value": "{:.4f}"}), width="stretch", hide_index=True)
        if (bt["p-value"] < a_bds).any():
            st.warning("BDS **bác bỏ** i.i.d.: mô hình tuyến tính đã bỏ sót thông tin phi tuyến.")
        else:
            st.info("BDS **không** bác bỏ i.i.d.: chưa thấy bằng chứng phi tuyến mạnh trong phần dư.")
    except ImportError:
        st.error("Thiếu `statsmodels` — chạy `pip install statsmodels` để dùng BDS.")
    st.markdown("**(b) BDS theo từng biến X** (p đã hiệu chỉnh Bonferroni theo số chiều)")
    bds_var = run_bds_vars(dtr, target, xcols, max(1, min(int(p_ar), 8)), min(int(max_dim), 4))
    st.dataframe(bds_var.assign(**{"Phi tuyến": np.where(bds_var.iloc[:, 1] < a_bds, "✅ Có", "–")})
                 .style.format({"p BDS (phần dư Y~Y,X)": "{:.4f}"}), width="stretch", hide_index=True)

# =============================================================================== TAB 2
with tab2:
    step_box("<b>XGBoost</b> dự báo y<sub>t</sub> từ các trễ của X và Y. \n Đo mức đóng góp bằng <b>gain</b>. Học trên 80% đầu của "
             "TRAIN, kiểm chứng bằng R² trên 20% cuối. Chạy ở <b>tỉnh đại diện</b> và trên <b>toàn bộ các tỉnh</b> để kiểm tra độ nhất quán.")
    m1, m2 = st.columns(2)
    n_lags = m1.number_input("Số lag", 1, 30, 5)
    top_n = m2.number_input("Top-n coi là quan trọng", 1, max(1, len(xcols)), min(3, len(xcols)))
    try:
        imp_p, info_p = run_xgb(panel, pm["date_col"], pm["entity_col"], target, xcols, d_eff, train_end, int(n_lags),
                                "proxy", df)
        imp_a, info_a = run_xgb(panel, pm["date_col"], pm["entity_col"], target, xcols, d_eff, train_end, int(n_lags),
                                "panel", df)
        xgb_ok = True
    except ImportError:
        xgb_ok = False
        st.error("Chưa cài `xgboost` — chạy `pip install xgboost`. Đây là thước đo duy nhất của hệ thống nên không có phương án thay thế.")
    if xgb_ok:
        k1, k2 = st.columns(2)
        k1.metric("R² kiểm chứng — tỉnh đại diện", f"{info_p['r2_val']:.3f}")
        k2.metric("R² kiểm chứng — toàn panel", f"{info_a['r2_val']:.3f}")
        if max(info_p["r2_val"], info_a["r2_val"]) < 0.02:
            st.warning("R² kiểm chứng gần 0: mô hình cây gần như không dự báo được ΔY nên thứ hạng importance chỉ mang tính "
                       "tham khảo (rất dễ là nhiễu). Cần thận trọng khi “cứu” biến.")
        evx = imp_p.rename(columns={"Importance (gain)": "Importance (tỉnh đại diện)", "Hạng": "Hạng (đại diện)",
                                    "Lag mạnh nhất": "Lag XGB"}).merge(
            imp_a[["Biến", "Importance (gain)", "Hạng"]].rename(columns={"Importance (gain)": "Importance (toàn panel)",
                                                                         "Hạng": "Hạng (toàn panel)"}), on="Biến")
        evx["Granger đạt"] = evx["Biến"].map(gr_ok).map(lambda v: "✅" if v else "–")
        st.dataframe(evx.style.format({"Importance (tỉnh đại diện)": "{:.4f}", "Importance (toàn panel)": "{:.4f}"}),
                     width="stretch", hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=evx["Biến"], x=evx["Importance (tỉnh đại diện)"], orientation="h", name="Tỉnh đại diện"))
        fig.add_trace(go.Bar(y=evx["Biến"], x=evx["Importance (toàn panel)"], orientation="h", name="Toàn panel"))
        fig.update_layout(barmode="group", height=max(320, 60 * len(evx)), margin=dict(t=30, b=10),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h"),
                          title="XGBoost importance (tổng gain các lag)")
        st.plotly_chart(fig, width="stretch")
        rescue = [v for v, r in zip(evx["Biến"], evx["Hạng (đại diện)"]) if r <= top_n and not gr_ok.get(v, False)]
        if rescue:
            st.info("**Biến có thể “cứu”** (trượt Granger nhưng nằm Top-n XGBoost): " + ", ".join(rescue))

# =============================================================================== TAB 3
with tab3:
    step_box("Giá có thuần túy là bước ngẫu nhiên hay còn cấu trúc phụ thuộc? <b>Ljung-Box</b> (tự tương quan của lợi suất) và "
             "<b>Variance Ratio</b> (Lo-MacKinlay, z* bền).")
    price = df[target].iloc[:train_n]
    if (price > 0).all():
        ret, ret_name = np.log(price).diff().dropna().to_numpy(), "lợi suất log"
    else:
        ret, ret_name = price.diff().dropna().to_numpy(), "sai phân bậc 1"
    st.caption(f"Tính trên {ret_name} của {target} (TRAIN, tỉnh đại diện), n = {len(ret):,}.")
    lb, vr = ljung_box(ret, [5, 10, 20]), variance_ratio(ret, [2, 5, 10, 20])
    a, b = st.columns(2)
    a.markdown("**Ljung-Box**")
    a.dataframe(lb.style.format({"Q (Ljung-Box)": "{:.2f}", "p-value": "{:.4f}"}), width="stretch", hide_index=True)
    b.markdown("**Variance Ratio**")
    b.dataframe(vr.style.format({"VR(q)": "{:.3f}", "z*": "{:.2f}", "p-value": "{:.4f}"}), width="stretch", hide_index=True)
    if (lb["p-value"] < 0.05).any() or (vr["p-value"] < 0.05).any():
        st.warning("Có bằng chứng **cấu trúc phụ thuộc** (không phải bước ngẫu nhiên thuần túy).")
    else:
        st.info("Không bác bỏ bước ngẫu nhiên: giá khó dự báo hơn naive; kỳ vọng cải thiện nên thận trọng.")

# =============================================================================== LƯU BẰNG CHỨNG
st.divider()
if xgb_ok:
    st.session_state["evidence_nl"] = evx.merge(bds_var, on="Biến", how="left")
    st.success(f"Đã lưu bằng chứng phi tuyến (XGBoost + BDS) cho {len(evx)} biến. Tiếp tục **Trang 4 · Chốt biến & Gói dữ liệu**.")
else:
    st.warning("Chưa lưu được bằng chứng vì thiếu xgboost.")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_utils import difference
from modules.stat_tests import (adf_kpss, ccf_prewhitened, granger_hc1, integration_order, johansen_rank,
                                summarize_ccf, summarize_granger)
from modules.ui import fmt_p, require, reset_downstream, setup_page, step_box

setup_page("2 · Kiểm định tuyến tính")
st.title("Trang 2 · Kiểm định tuyến tính (Luồng 1)")
st.caption("Dữ liệu đầu vào: chuỗi của **tỉnh đại diện**. Thu thập bằng chứng tuyến tính (bậc d, CCF, Granger, Johansen → VAR hay VECM). "
           "Việc chọn biến vào Tập Cơ Sở được chốt ở **Trang 4**.")
require(["df_clean"], "Chưa có dữ liệu sạch. Hãy hoàn tất **Trang 1** và bấm *Chốt dữ liệu sạch*.")

df = st.session_state["df_clean"]
target = st.session_state["target"]
xcols = st.session_state["x_cols"]
train_n = st.session_state["train_n"]

with st.expander("Tham số kiểm định", expanded=False):
    p1, p2, p3, p4 = st.columns(4)
    alpha = p1.select_slider("Mức ý nghĩa α", [0.01, 0.05, 0.10], value=0.05)
    max_lag_ccf = p2.number_input("Số lag tối đa — CCF", 3, 60, 20)
    max_lag_gc = p3.number_input("Số lag tối đa — Granger", 1, 30, 10)
    ar_max = p4.number_input("Bậc AR tối đa (tiền trắng hoá)", 1, 30, 10)


# --------------------------------------------------------------------------- hàm tính (cache)
@st.cache_data(show_spinner="Đang chạy ADF/KPSS…")
def run_stationarity(train_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for c in train_df.columns:
        s = train_df[c]
        r0, r1 = adf_kpss(s, alpha), adf_kpss(s.diff().dropna(), alpha)
        rows.append({"Biến": c, "ADF p (d=0)": r0["adf_p"], "KPSS p (d=0)": r0["kpss_p"], "Kết luận d=0": r0["verdict"],
                     "ADF p (d=1)": r1["adf_p"], "KPSS p (d=1)": r1["kpss_p"], "Kết luận d=1": r1["verdict"],
                     "d đề xuất": integration_order(s, 2, alpha)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Đang tính CCF tiền trắng hoá…")
def run_ccf(dtr: pd.DataFrame, target: str, xcols: list, max_lag: int, alpha: float, ar_max: int):
    rows, tabs = [], {}
    for c in xcols:
        tab, p = ccf_prewhitened(dtr[c].to_numpy(), dtr[target].to_numpy(), max_lag, alpha, ar_max)
        s = summarize_ccf(tab)
        tabs[c] = tab
        rows.append({"Biến": c, "AR(p) lọc": p, "Lag dẫn trước tốt nhất": s["best_lag"], "CCF tại lag đó": s["ccf"],
                     "Ngưỡng Bonferroni": s["threshold"], "Số lag vượt ngưỡng": s["n_sig"],
                     "CCF đạt": s["pass"]})
    return pd.DataFrame(rows), tabs


@st.cache_data(show_spinner="Đang chạy Granger (HC1 + Bonferroni)…")
def run_granger(dtr: pd.DataFrame, target: str, xcols: list, max_lag: int, alpha: float):
    rows, tabs = [], {}
    y = dtr[target].to_numpy()
    for c in xcols:
        x = dtr[c].to_numpy()
        fw, bw = granger_hc1(y, x, max_lag), granger_hc1(x, y, max_lag)
        sf, sb = summarize_granger(fw, alpha), summarize_granger(bw, alpha)
        tabs[c] = (fw, bw)
        rows.append({"Biến": c, "Lag tốt nhất": sf["best_lag"], "F": sf["F"], "p thô": sf["p_raw"],
                     "p Bonferroni": sf["p_bonf"], "Granger đạt": sf["pass"],
                     "p Bonf. (Y→X)": sb["p_bonf"], "Phản hồi 2 chiều": sb["pass"]})
    return pd.DataFrame(rows), tabs


tab1, tab2, tab3 = st.tabs(["Bậc tích hợp & Tính dừng", "Độ trễ dẫn trước (CCF)", "Granger & Đồng liên kết"])

# =============================================================================== TAB 1
with tab1:
    step_box("Chạy song song <b>ADF</b> (H0: có nghiệm đơn vị) và <b>KPSS</b> (H0: dừng) trên đoạn TRAIN.\n " 
             "Chuỗi “Dừng” khi ADF bác bỏ và KPSS không bác bỏ \n. <b>Bắt buộc chốt một bậc d chung</b> cho toàn hệ thống.")
    lv = df.iloc[:train_n]
    stn = run_stationarity(lv, alpha)
    show = stn.copy()
    for col in ["ADF p (d=0)", "KPSS p (d=0)", "ADF p (d=1)", "KPSS p (d=1)"]:
        show[col] = show[col].map(fmt_p)
    show.insert(1, "Vai trò", ["Y" if b == target else "X" for b in show["Biến"]])
    st.dataframe(show, width="stretch", hide_index=True)

    rec = int(stn["d đề xuất"].max())
    c1, c2 = st.columns([1, 3])
    d_pick = c1.number_input("Bậc sai phân d (chung)", 0, 2, rec)
    c2.info(f"Đề xuất: **d = {rec}** (lớn nhất trong các biến). \n VARX/VARX-L bắt buộc dùng dữ liệu đã sai phân bậc d; "
            "luồng 2 (Deep) được phép dùng dữ liệu gốc.")
    if d_pick == 0:
        st.warning("d = 0: các kiểm định CCF/Granger trên mức giá gốc dễ cho hồi quy giả nếu chuỗi có nghiệm đơn vị.")
    if st.button("🔒 Chốt bậc d", type="primary"):
        reset_downstream("df_clean")
        st.session_state["d"] = int(d_pick)
        st.success(f"Đã chốt d = {int(d_pick)}.")
    if st.session_state.get("d") is not None:
        st.caption(f"d đang chốt: **{st.session_state['d']}**")

d = st.session_state.get("d")
if d is None:
    for t in (tab2, tab3):
        with t:
            st.info("Hãy **chốt bậc d** trước.")
    st.stop()

dfd = difference(df, d)
dtr = dfd.iloc[:train_n - d]

# =============================================================================== TAB 2
with tab2:
    step_box(f"Lọc <b>cả X và Y</b> qua cùng một bộ lọc AR khớp trên X (tiền trắng hoá) rồi tính CCF trên dữ liệu "
             f"sai phân bậc d = {d} (TRAIN). Lag k &gt; 0: X<sub>t-k</sub> dẫn trước Y<sub>t</sub>. "
             "Ngưỡng hiệu chuẩn Bonferroni theo số lag.")
    ccf_tab, ccf_all = run_ccf(dtr, target, xcols, int(max_lag_ccf), alpha, int(ar_max))
    view = ccf_tab.copy()
    view["CCF đạt"] = np.where(view["CCF đạt"], "✅", "–")
    st.dataframe(view.style.format({"CCF tại lag đó": "{:.3f}", "Ngưỡng Bonferroni": "{:.3f}"}),
                 width="stretch", hide_index=True)

    pick = st.selectbox("Vẽ CCF cho biến", xcols)
    tb = ccf_all[pick]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=tb["lag"], y=tb["ccf"], marker_color=np.where(tb["significant"], "#C0392B", "#95A5A6"),
                         name="CCF"))
    thr = float(tb["threshold"].iloc[0])
    for s in (thr, -thr):
        fig.add_hline(y=s, line_dash="dash", line_color="#2C3E50")
    fig.update_layout(height=340, margin=dict(t=30, b=10), xaxis_title="Lag k (k>0: X dẫn trước Y)",
                      yaxis_title="CCF (pre-whitened)", title=f"CCF: {pick} → {target}", showlegend=False)
    st.plotly_chart(fig, width="stretch")

# =============================================================================== TAB 3
with tab3:
    step_box("<b>Granger</b>: Wald F-test với sai số chuẩn <b>HC1</b>, p-value hiệu chỉnh <b>Bonferroni</b> theo số lag. "
             "Cũng kiểm tra chiều ngược Y→X để phát hiện <b>phản hồi hai chiều</b>. <b>Johansen</b> xác định có "
             "dùng VECM hay không.")
    gr_tab, gr_all = run_granger(dtr, target, xcols, int(max_lag_gc), alpha)
    gv = gr_tab.copy()
    gv["Granger đạt"] = np.where(gv["Granger đạt"], "✅", "–")
    gv["Phản hồi 2 chiều"] = np.where(gv["Phản hồi 2 chiều"], "⚠️ Có", "–")
    for col in ["p thô", "p Bonferroni", "p Bonf. (Y→X)"]:
        gv[col] = gv[col].map(fmt_p)
    st.dataframe(gv.style.format({"F": "{:.2f}"}), width="stretch", hide_index=True)
    st.caption("Lưu ý: Granger chỉ đo quan hệ dự báo **tuyến tính**. Không bác bỏ H0 chưa đủ để loại biến vĩnh viễn — "
               "biến có logic nghiệp vụ mạnh sẽ được xét lại ở Luồng 2 (Trang 3).")
    with st.expander("Chi tiết p-value theo từng lag"):
        gp = st.selectbox("Biến", xcols, key="gc_pick")
        fw, bw = gr_all[gp]
        a, b = st.columns(2)
        a.markdown(f"**{gp} → {target}**")
        a.dataframe(fw, width="stretch", hide_index=True)
        b.markdown(f"**{target} → {gp}**")
        b.dataframe(bw, width="stretch", hide_index=True)

    st.subheader("Đồng liên kết Johansen & quyết định VAR / VECM")
    vecm_hint, johansen_r, base_model, rank_sys = False, {}, "VAR", 0
    if d != 1:
        st.info("Johansen áp dụng khi các chuỗi là I(1) (d = 1). Hiện d đã chốt khác 1 nên mô hình cơ sở là **VAR trên sai phân**.")
    else:
        k_diff = st.number_input("k_ar_diff", 1, 10, 2)

        @st.cache_data(show_spinner="Đang chạy Johansen…")
        def run_johansen(lv_tr_: pd.DataFrame, target_: str, xcols_: list, sys_vars_: list, k_: int):
            pair = []
            for c in xcols_:
                r = johansen_rank(lv_tr_[[target_, c]].to_numpy(), 0, k_)
                pair.append({"Cặp (Y, X)": c, "Trace stat (r=0)": r["trace"][0], "Trace crit 95%": r["trace_crit95"][0],
                             "Hạng r (trace)": r["rank_trace"], "Hạng r (max-eig)": r["rank_maxeig"]})
            system = johansen_rank(lv_tr_[[target_] + sys_vars_].to_numpy(), 0, k_) if sys_vars_ else None
            return pd.DataFrame(pair), system

        try:
            sys_vars = [c for c, ok in zip(xcols, gr_tab["Granger đạt"]) if ok][:6]
            jt, system = run_johansen(df.iloc[:train_n], target, xcols, sys_vars, int(k_diff))
            johansen_r = dict(zip(jt["Cặp (Y, X)"], jt["Hạng r (trace)"]))
            st.markdown("**Theo từng cặp (Y, X)**")
            st.dataframe(jt.style.format({"Trace stat (r=0)": "{:.2f}", "Trace crit 95%": "{:.2f}"}), width="stretch",
                         hide_index=True)
            if system is not None:
                rank_sys = int(system["rank_trace"])
                st.markdown(f"**Hệ biến Y + {len(sys_vars)} biến qua Granger** ({', '.join(sys_vars)}): hạng r (trace) = **{rank_sys}**")
            else:
                rank_sys = int(max(johansen_r.values(), default=0))
                st.caption("Chưa có biến nào qua Granger → quyết định dựa trên kết quả theo cặp.")
            vecm_hint = rank_sys > 0
            base_model = "VECM" if vecm_hint else "VAR"
        except ImportError:
            st.error("Thiếu thư viện `statsmodels` — chạy `pip install statsmodels` để dùng Johansen.")
    (st.warning if base_model == "VECM" else st.success)(
        f"Quyết định động: mô hình cơ sở = **{base_model}**"
        + (" (có đồng liên kết, r > 0 → hiệu chỉnh sai số/ECT)." if base_model == "VECM" else " trên sai phân (r = 0)."))
    st.session_state["base_model"], st.session_state["johansen_rank"] = base_model, rank_sys


# =============================================================================== NHẤT QUÁN GIỮA CÁC TỈNH
@st.cache_data(show_spinner="Đang chạy Granger cho từng tỉnh…")
def run_granger_panel(clean: pd.DataFrame, date_col_: str, entity_col_: str, target_: str, xcols_: list, d_: int,
                      train_end_, max_lag_: int, alpha_: float) -> pd.DataFrame:
    rows = {}
    for e, g in clean.groupby(entity_col_, observed=True, sort=False):
        f = difference(g.set_index(date_col_).sort_index()[[target_] + xcols_], d_)
        f = f[f.index <= train_end_]
        rows[str(e)] = {c: summarize_granger(granger_hc1(f[target_].to_numpy(), f[c].to_numpy(), max_lag_), alpha_)["pass"]
                        for c in xcols_}
    return pd.DataFrame(rows)


st.divider()
st.subheader("Nhất quán giữa các tỉnh")
panel = st.session_state["df_clean_panel"]
pm = st.session_state["panel_meta"]
gp = run_granger_panel(panel, pm["date_col"], pm["entity_col"], target, xcols, int(d), st.session_state["train_end"],
                       int(max_lag_gc), alpha)
cons = gp.sum(axis=1)
cons_view = gp.replace({True: "✅", False: "–"})
cons_view.insert(0, "Số tỉnh đạt", cons.astype(int).astype(str) + f"/{gp.shape[1]}")
st.dataframe(cons_view, width="stretch")

# =============================================================================== BẰNG CHỨNG
st.divider()
st.subheader("Bằng chứng tuyến tính (tự động lưu cho Trang 4)")
ev = pd.DataFrame({
    "Biến": xcols,
    "Trễ CCF": ccf_tab["Lag dẫn trước tốt nhất"].to_numpy(),
    "CCF đạt": ccf_tab["CCF đạt"].to_numpy(),
    "p Granger (Bonferroni)": gr_tab["p Bonferroni"].to_numpy(),
    "Granger đạt": gr_tab["Granger đạt"].to_numpy(),
    "Phản hồi 2 chiều": gr_tab["Phản hồi 2 chiều"].to_numpy(),
    "r Johansen": [johansen_r.get(c, np.nan) for c in xcols],
    "Số tỉnh Granger đạt": cons.reindex(xcols).to_numpy(),
    "Tổng số tỉnh": gp.shape[1],
})
st.session_state["evidence_lin"] = ev
st.session_state["vecm_hint"] = bool(vecm_hint)
disp = ev.copy()
for c in ["CCF đạt", "Granger đạt"]:
    disp[c] = np.where(disp[c], "✅", "–")
disp["Phản hồi 2 chiều"] = np.where(disp["Phản hồi 2 chiều"], "⚠️", "")
disp["p Granger (Bonferroni)"] = disp["p Granger (Bonferroni)"].map(fmt_p)
st.dataframe(disp, width="stretch", hide_index=True)
st.success(f"Đã lưu bằng chứng cho {len(xcols)} biến (d = {d}, mô hình cơ sở: {st.session_state['base_model']}). "
           "Tiếp tục **Trang 3 · Khám phá phi tuyến**.")

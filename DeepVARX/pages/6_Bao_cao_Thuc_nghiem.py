import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from modules.attention_utils import attention_verdict
from modules.data_utils import metrics
from modules.model_specs import ORDER
from modules.stat_tests import dm_from_loss_diff, ljung_box
from modules.ui import fmt_p, require, setup_page

setup_page("6 · Báo cáo thực nghiệm")
st.title("Trang 6 · Báo cáo thực nghiệm & khám phá chuyên sâu")
st.caption("Mọi sai số tính trên **mức giá gốc (VNĐ/kg)**, dự báo 1 bước trên đoạn TEST chung. Kết quả được báo cáo **theo từng tỉnh** "
           "và **trung bình các tỉnh**.")
require(["results"], "Chưa có mô hình nào được huấn luyện. Hãy huấn luyện ở **Trang 5**.")
res = st.session_state["results"]
if not res:
    st.warning("Chưa có mô hình nào được huấn luyện. Hãy huấn luyện ở **Trang 5**.")
    st.stop()

target = st.session_state["target"]
names = [m for m in ORDER if m in res]
missing = [m for m in ORDER if m not in res]
if missing:
    st.info("Chưa huấn luyện: " + ", ".join(missing) + " (không xuất hiện trong các so sánh).")
provs = sorted(res[names[0]]["pred"]["province"].unique())
P = {m: res[m]["pred"].assign(e=lambda x: x["y_true"] - x["y_pred"]) for m in names}


def avg_metrics(m):
    return res[m]["metrics"]


tab1, tab2, tab3, tab4 = st.tabs(["Hiệu suất dự báo", "Đường dự báo", "Phân tích phần dư", "Hộp đen DeepVARX (Attention)"])

# =============================================================================== TAB 1
with tab1:
    naive_by = pd.DataFrame([{"province": e, **metrics(g["y_true"], g["y_prev"])} for e, g in P[names[0]].groupby("province")])
    rows = [{"Mô hình": m, "Phạm vi": res[m]["scope"], **avg_metrics(m)} for m in names]
    rows.append({"Mô hình": "Naive (tham chiếu)", "Phạm vi": "-", **{k: float(naive_by[k].mean()) for k in ("RMSE", "MAE", "MAPE (%)")}})
    tbl = pd.DataFrame(rows)
    if "VAR" in res:
        base = res["VAR"]["metrics"]["RMSE"]
        tbl["Mức cải thiện so với VAR (%)"] = (base - tbl["RMSE"]) / base * 100
    else:
        tbl["Mức cải thiện so với VAR (%)"] = np.nan
        st.info("Huấn luyện **VAR** ở Trang 5 để có cột “Mức cải thiện so với VAR”.")
    st.markdown(f"**Trung bình {len(provs)} tỉnh**")
    st.dataframe(tbl.style.format({"RMSE": "{:,.1f}", "MAE": "{:,.1f}", "MAPE (%)": "{:.2f}", "Mức cải thiện so với VAR (%)": "{:+.1f}%"},
                                  na_rep="–").highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color="#d5f5e3"),
                 width="stretch", hide_index=True)
    if "DeepVARX" in res and "VAR" in res:
        imp = float(tbl.loc[tbl["Mô hình"] == "DeepVARX", "Mức cải thiện so với VAR (%)"].iloc[0])
        (st.success if imp > 0 else st.warning)(f"DeepVARX {'giảm' if imp > 0 else 'tăng'} **{abs(imp):.1f}%** RMSE trung bình so với VAR.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[avg_metrics(m)["RMSE"] for m in names], name="RMSE"))
    fig.add_trace(go.Bar(x=names, y=[avg_metrics(m)["MAE"] for m in names], name="MAE"))
    fig.update_layout(barmode="group", height=340, margin=dict(t=40, b=10), yaxis_title="VNĐ/kg",
                      title="Sai số trung bình các tỉnh — từ tuyến tính đến học sâu", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Chi tiết theo từng tỉnh"):
        piv = pd.concat({m: res[m]["by_prov"].set_index("province")["RMSE"] for m in names}, axis=1)
        piv["Naive"] = naive_by.set_index("province")["RMSE"]
        st.markdown("**RMSE (VNĐ/kg)**")
        st.dataframe(piv.style.format("{:,.1f}").highlight_min(axis=1, subset=names, color="#d5f5e3"), width="stretch")
        pm_ = pd.concat({m: res[m]["by_prov"].set_index("province")["MAPE (%)"] for m in names}, axis=1)
        st.markdown("**MAPE (%)**")
        st.dataframe(pm_.style.format("{:.2f}").highlight_min(axis=1, color="#d5f5e3"), width="stretch")
        if "VAR" in res:
            gain = (piv[names].rsub(piv["VAR"], axis=0)).div(piv["VAR"], axis=0) * 100
            st.markdown("**Mức cải thiện RMSE so với VAR (%)**")
            st.dataframe(gain.style.format("{:+.1f}%"), width="stretch")

# =============================================================================== TAB 2
with tab2:
    scope = st.selectbox("Phạm vi", ["Trung bình các tỉnh"] + provs)
    st.markdown("**Chọn đường hiển thị**")
    cols = st.columns(len(names) + 1)
    show_actual = cols[0].checkbox("Thực tế", value=True, key="show_actual")
    show = {m: c.checkbox(m, value=m in ("VARX", "DeepVARX"), key=f"show_{m}") for c, m in zip(cols[1:], names)}

    def series(m, col):
        f = P[m] if scope.startswith("Trung") else P[m][P[m]["province"] == scope]
        return f.groupby("date")[col].mean()

    fig = go.Figure()
    if show_actual:
        a = series(names[0], "y_true")
        fig.add_trace(go.Scatter(x=a.index, y=a, name="Thực tế", line=dict(color="black", width=2)))
    for m in names:
        if show[m]:
            s_ = series(m, "y_pred")
            fig.add_trace(go.Scatter(x=s_.index, y=s_, name=m, line=dict(width=1.4)))
    fig.update_layout(height=430, margin=dict(t=30, b=10), yaxis_title=f"{target} (VNĐ/kg)", legend=dict(orientation="h"),
                      hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Directional Accuracy** — đúng hướng tăng/giảm so với phiên trước")
    da = {}
    for m in names:
        f = P[m]
        ok = np.sign(f["y_pred"] - f["y_prev"]) == np.sign(f["y_true"] - f["y_prev"])
        da[m] = ok.groupby(f["province"]).mean() * 100
    dadf = pd.DataFrame(da)
    st.dataframe(pd.DataFrame({"Mô hình": names, "DA trung bình (%)": [dadf[m].mean() for m in names]})
                 .style.format({"DA trung bình (%)": "{:.1f}"}).highlight_max(subset=["DA trung bình (%)"], color="#d5f5e3"),
                 width="stretch", hide_index=True)
    with st.expander("Directional Accuracy theo từng tỉnh"):
        st.dataframe(dadf.style.format("{:.1f}").highlight_max(axis=1, color="#d5f5e3"), width="stretch")
    st.caption("Sai số tiền có thể lớn nhưng đúng hướng vẫn hữu ích cho giao dịch. 50% ≈ đoán ngẫu nhiên.")

# =============================================================================== TAB 3
with tab3:
    sc3 = st.selectbox("Phạm vi phần dư", ["Tất cả tỉnh (gộp)"] + provs, key="sc3")
    pick = st.multiselect("So sánh phần dư của", names, default=[m for m in ("VARX", "DeepVARX") if m in names])

    def resid_of(m):
        f = P[m] if sc3.startswith("Tất") else P[m][P[m]["province"] == sc3]
        return f["e"].to_numpy()

    fig = go.Figure()
    for m in pick:
        e = resid_of(m)
        if e.std() > 0:
            xs = np.linspace(e.min() - e.std(), e.max() + e.std(), 300)
            fig.add_trace(go.Scatter(x=xs, y=stats.gaussian_kde(e)(xs), name=m))
    if pick:
        e0 = resid_of(pick[0])
        xs = np.linspace(e0.min() - e0.std(), e0.max() + e0.std(), 300)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, 0, e0.std()), name=f"Chuẩn N(0,σ) của {pick[0]}",
                                 line=dict(dash="dot", color="gray")))
    fig.add_vline(x=0, line_dash="dash")
    fig.update_layout(height=340, margin=dict(t=30, b=10), xaxis_title="Phần dư (VNĐ/kg)", yaxis_title="Mật độ",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    drows = []
    for m in names:
        e = resid_of(m)
        drows.append({"Mô hình": m, "Bias (TB)": e.mean(), "Độ lệch chuẩn": e.std(), "Skewness": stats.skew(e),
                      "Excess kurtosis": stats.kurtosis(e), "p Jarque-Bera": float(stats.jarque_bera(e).pvalue),
                      "p Ljung-Box(10)": float(ljung_box(e, [10])["p-value"].iloc[0])})
    st.dataframe(pd.DataFrame(drows).style.format({"Bias (TB)": "{:,.1f}", "Độ lệch chuẩn": "{:,.1f}", "Skewness": "{:.2f}",
                                                   "Excess kurtosis": "{:.2f}", "p Jarque-Bera": "{:.4f}",
                                                   "p Ljung-Box(10)": "{:.4f}"}), width="stretch", hide_index=True)
    if sc3.startswith("Tất"):
        st.caption("Gộp nhiều tỉnh: phần dư chồng nhiều chuỗi nên Ljung-Box chỉ mang tính tham khảo — chọn từng tỉnh để kiểm tra tự tương quan.")

    st.divider()
    st.markdown("**Kiểm định Diebold-Mariano** (dự báo 1 bước, hiệu chỉnh Harvey-Leybourne-Newbold)")
    if len(names) < 2:
        st.info("Cần ít nhất 2 mô hình đã huấn luyện để chạy DM test.")
    else:
        k1, k2, k3 = st.columns(3)
        A = k1.selectbox("Mô hình A", names, index=names.index("DeepVARX") if "DeepVARX" in names else len(names) - 1)
        others = [m for m in names if m != A]
        B = k2.selectbox("Mô hình B (đối chứng)", others, index=others.index("VARX") if "VARX" in others else 0)
        power = 2 if k3.radio("Hàm tổn thất", ["Bình phương (MSE)", "Tuyệt đối (MAE)"], horizontal=True).startswith("Bình") else 1

        def diffs(a, b):
            mg = P[a][["date", "province", "e"]].merge(P[b][["date", "province", "e"]], on=["date", "province"], suffixes=("_a", "_b"))
            mg["d"] = mg["e_b"].abs() ** power - mg["e_a"].abs() ** power
            return mg

        mg = diffs(A, B)
        r = dm_from_loss_diff(mg.groupby("date")["d"].mean().to_numpy())
        ma, mb, mc = st.columns(3)
        ma.metric("Thống kê DM (trung bình các tỉnh)", f"{r['stat']:.3f}" if r["stat"] == r["stat"] else "–")
        mb.metric("p-value", fmt_p(r["p"]))
        mc.metric("RMSE TB: A vs B", f"{res[A]['metrics']['RMSE']:,.0f} vs {res[B]['metrics']['RMSE']:,.0f}")
        if r["p"] != r["p"]:
            st.warning("Không đủ dữ liệu để chạy DM test.")
        elif r["p"] < 0.05 and r["mean_diff"] > 0:
            st.success(f"**{A} vượt trội hơn {B} có ý nghĩa thống kê ở mức tin cậy 95%** (p = {fmt_p(r['p'])}).")
        elif r["p"] < 0.05:
            st.error(f"**{A} kém hơn {B} có ý nghĩa thống kê ở mức tin cậy 95%** (p = {fmt_p(r['p'])}).")
        else:
            st.info(f"Chưa đủ bằng chứng thống kê để khẳng định {A} và {B} khác nhau (p = {fmt_p(r['p'])} ≥ 0.05) — "
                    "chênh lệch sai số có thể chỉ do ngẫu nhiên.")
        st.caption("DM chạy trên chênh lệch tổn thất **trung bình theo ngày giữa các tỉnh**, để không đếm 9 chuỗi tương quan chéo "
                   "như 9 quan sát độc lập (nếu gộp thẳng, p-value sẽ nhỏ giả tạo).")
        prow = []
        for e, g in mg.groupby("province"):
            q = dm_from_loss_diff(g["d"].to_numpy())
            prow.append({"Tỉnh": e, "DM": q["stat"], "p-value": q["p"],
                         "Kết luận": f"{A} tốt hơn ✅" if q["p"] < 0.05 and q["mean_diff"] > 0
                         else (f"{A} kém hơn ❌" if q["p"] < 0.05 else "Không khác biệt")})
        pdm = pd.DataFrame(prow)
        st.markdown(f"**Theo từng tỉnh** — {A} tốt hơn có ý nghĩa ở **{int(pdm['Kết luận'].str.contains('tốt hơn').sum())}/{len(pdm)}** tỉnh")
        st.dataframe(pdm.style.format({"DM": "{:.3f}", "p-value": "{:.4f}"}), width="stretch", hide_index=True)

# =============================================================================== TAB 4
MONTHS = {i: f"tháng {i}" for i in range(1, 13)}
with tab4:
    dv = res.get("DeepVARX")
    if dv is None or dv.get("attn") is None:
        st.info("Huấn luyện **DeepVARX** ở Trang 5 để xem Attention Heatmap.")
        st.stop()
    at = dv["attn"]
    cols_all = at["cols"]
    exog = [c for c in cols_all if c != target]
    if not exog:
        st.info("DeepVARX này chỉ có chuỗi Y (không có biến ngoại sinh).")
        st.stop()
    t1, t2 = st.columns(2)
    pv = t1.selectbox("Tỉnh", sorted(set(at["prov"])))
    smooth = t2.select_slider("Làm mượt hiển thị (số phiên)", [1, 5, 10, 20], value=5)
    msk = at["prov"] == pv
    share = pd.DataFrame(at["test"][msk] / len(cols_all), index=pd.DatetimeIndex(at["dates"][msk]), columns=cols_all).sort_index()
    view = share[exog].rolling(smooth, min_periods=1).mean() * 100
    hm = go.Figure(go.Heatmap(z=view.T.values, x=view.index, y=exog, colorscale="YlOrBr", colorbar=dict(title="% chú ý")))
    hm.update_layout(height=max(280, 70 * len(exog)), margin=dict(t=40, b=10), xaxis_title="Thời gian (TEST)",
                     title=f"Attention Heatmap — {pv}")
    st.plotly_chart(hm, width="stretch")
    st.caption("Giá trị = % trọng số gate biến (trung bình các bước trễ của mỗi cửa sổ). Phần còn lại thuộc chuỗi giá Y.")

    st.markdown("**Diễn giải tự động theo mùa vụ**")
    monthly = share.groupby(share.index.month).agg(["mean", "count"])
    sentences = []
    for v in exog:
        mm, cnt = monthly[(v, "mean")], monthly[(v, "count")]
        ok = cnt[cnt >= 5].index
        if len(ok) < 2:
            continue
        top = mm[ok].idxmax()
        top_val = float(mm[top])
        other = float(share.loc[share.index.month != top, v].mean())
        if top_val >= 1.5 * other and top_val - other >= 0.03:
            sentences.append((top_val / max(other, 1e-9), f"Vào **{MONTHS[top]}**, DeepVARX dồn **{top_val:.0%}** sự chú ý vào **{v}**, "
                                                          f"trong khi các tháng khác trung bình chỉ {other:.0%}."))
    if sentences:
        for _, s in sorted(sentences, reverse=True)[:4]:
            st.markdown("- " + s)
        st.caption("Đây là **gợi ý** rằng mạng điều chỉnh mức chú ý theo thời điểm, không phải bằng chứng nhân quả. Đoạn TEST chỉ phủ "
                   "một số tháng; cần đối chiếu với DM test và Ablation, và xem tỉnh khác có cùng xu hướng không.")
    else:
        st.info("Mức chú ý phân bố khá đều theo tháng trong đoạn TEST của tỉnh này (hoặc TEST quá ngắn để so sánh các tháng).")

    with st.expander("Gate theo epoch, theo độ trễ và kết luận Ablation"):
        if at.get("epoch") is not None:
            ae = at["epoch"]
            h2 = go.Figure(go.Heatmap(z=ae.T.values, x=ae.index, y=ae.columns, colorscale="YlOrBr"))
            h2.update_layout(height=max(260, 45 * len(ae.columns)), margin=dict(t=30, b=10), xaxis_title="Epoch",
                             title="Variable Attention theo epoch (validation, gate tương đối ≈ 1 là trung bình)")
            st.plotly_chart(h2, width="stretch")
            vd = attention_verdict(ae, 0.1)
            vd = vd[vd["Biến"] != target]
            st.dataframe(vd.style.format({c: "{:.3f}" for c in vd.columns if c.startswith("Gate")}), width="stretch", hide_index=True)
        lag = at["lag"]
        h3 = go.Figure(go.Heatmap(z=lag.T, x=[f"t-{lag.shape[0] - i}" for i in range(lag.shape[0])], y=cols_all, colorscale="YlOrBr"))
        h3.update_layout(height=max(260, 45 * len(cols_all)), margin=dict(t=30, b=10), title="Gate biến theo độ trễ (TB toàn TEST)")
        st.plotly_chart(h3, width="stretch")

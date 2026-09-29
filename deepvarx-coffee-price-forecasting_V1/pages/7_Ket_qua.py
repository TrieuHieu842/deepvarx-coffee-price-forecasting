import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"

st.set_page_config(page_title="Kết quả & So sánh", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    results = st.session_state.get("model_results", {})
    if results:
        st.success(f"{len(results)} mô hình đã huấn luyện")
        for m, r in results.items():
            st.caption(f"{m}: RMSE={r['test_metrics'].get('RMSE',0):.4f}")

st.caption(APP_HEADER)
st.title("Trang 7 — Kết quả & So sánh mô hình")

results = st.session_state.get("model_results", {})

if not results:
    st.warning("Chưa có kết quả. Vui lòng huấn luyện ít nhất 1 mô hình ở Trang 6.")
    if st.button("Đi đến Trang 6"): st.switch_page("pages/6_Huan_luyen.py")
    st.stop()

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# A. BẢNG METRICS
# ═══════════════════════════════════════════════════════════════════════════
st.header("A. Metrics đánh giá")
metric_rows = []
for mname, r in results.items():
    tm = r.get("test_metrics", {})
    vm = r.get("val_metrics", {})
    metric_rows.append({
        "Mô hình": mname,
        "p": r.get("p", "-"),
        "h": r.get("q", "-"),
        "MAE (Test)": round(tm.get("MAE", 0), 4),
        "RMSE (Test)": round(tm.get("RMSE", 0), 4),
        "sMAPE % (Test)": round(tm.get("sMAPE (%)", 0), 2),
        "CV(RMSE)": round(tm.get("CV(RMSE)", 0), 4),
        "RMSE (Val)": round(vm.get("RMSE", 0), 4),
        "Train time (s)": round(r.get("train_time", 0), 2),
    })

metric_df = pd.DataFrame(metric_rows).sort_values("RMSE (Test)")

# Highlight best
def _hl_best(s):
    is_min = s == s.min()
    return ["background-color:#dcfce7;font-weight:bold"if v else ""for v in is_min]

st.dataframe(
    metric_df.style.apply(_hl_best, subset=["RMSE (Test)","MAE (Test)","sMAPE % (Test)"])
             .format({
                 "MAE (Test)":"{:.4f}", "RMSE (Test)":"{:.4f}",
                 "sMAPE % (Test)":"{:.2f}%", "CV(RMSE)":"{:.4f}",
                 "RMSE (Val)":"{:.4f}", "Train time (s)":"{:.2f}",
             }),
    use_container_width=True, height=200,
)

best_model = metric_df.iloc[0]["Mô hình"]
best_rmse = metric_df.iloc[0]["RMSE (Test)"]
st.success(f"Mô hình tốt nhất: **{best_model}** — RMSE Test = **{best_rmse:.4f}**")

# ═══════════════════════════════════════════════════════════════════════════
# B. BIỂU ĐỒ DỰ BÁO
# ═══════════════════════════════════════════════════════════════════════════
st.divider()
st.header("B. Biểu đồ dự báo — Actual vs Forecast")

model_sel = st.selectbox("Chọn mô hình hiển thị", list(results.keys()))
r_sel = results[model_sel]

endo_names = r_sel.get("endo_names", [])
pred_test = np.array(r_sel.get("pred_test", []))
act_test = np.array(r_sel.get("actual_test", []))

if len(pred_test) > 0 and len(act_test) > 0:
    var_idx_sel = st.selectbox("Chọn biến nội sinh hiển thị",
                               range(len(endo_names)),
                               format_func=lambda i: endo_names[i] if endo_names else f"Var {i}")

    T = min(len(pred_test), len(act_test))
    t_axis = list(range(T))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=act_test[:T, var_idx_sel],
                             name="Actual", line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=t_axis, y=pred_test[:T, var_idx_sel],
                             name=f"Forecast ({model_sel})",
                             line=dict(color="#f97316", width=2, dash="dash")))
    fig.update_layout(
        title=f"Actual vs Forecast — {endo_names[var_idx_sel] if endo_names else 'Var'} — {model_sel}",
        xaxis_title="Bước thời gian (Test set)", yaxis_title="Giá trị",
        height=420, margin=dict(l=10,r=10,t=50,b=10),
        legend=dict(x=0, y=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    with st.expander("Biểu đồ Residuals"):
        resid = act_test[:T, var_idx_sel] - pred_test[:T, var_idx_sel]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=t_axis, y=resid, mode="lines+markers",
                                   line=dict(color="#8b5cf6"), name="Residual"))
        fig_r.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_r.update_layout(title="Residuals (Actual − Forecast)", height=280,
                            margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_r, use_container_width=True)
else:
    st.info("Chưa có dữ liệu dự báo cho mô hình này.")

# ═══════════════════════════════════════════════════════════════════════════
# D. SO SÁNH MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════
if len(results) > 1:
    st.divider()
    st.header("D. So sánh mô hình")

    metrics_to_compare = ["RMSE (Test)", "MAE (Test)", "sMAPE % (Test)", "CV(RMSE)", "Train time (s)"]
    metric_comp = st.selectbox("Metric so sánh", metrics_to_compare)
    comp_df = metric_df[["Mô hình", metric_comp]].sort_values(metric_comp)

    fig_comp = px.bar(comp_df, x="Mô hình", y=metric_comp,
                      color="Mô hình", text_auto=True,
                      title=f"So sánh {metric_comp} — Test Set",
                      color_discrete_sequence=px.colors.qualitative.Set2)
    fig_comp.update_layout(height=380, showlegend=False,
                           margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Radar chart
    st.subheader("Radar Chart — so sánh đa chiều (chuẩn hóa)")
    radar_metrics = ["MAE (Test)","RMSE (Test)","sMAPE % (Test)","CV(RMSE)"]
    radar_df = metric_df[["Mô hình"] + radar_metrics].copy()
    for col in radar_metrics:
        col_max = radar_df[col].max()
        if col_max > 0:
            radar_df[col] = radar_df[col] / col_max # 0-1 normalized (lower=better)

    fig_radar = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, (_, row) in enumerate(radar_df.iterrows()):
        vals = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]
        cats = radar_metrics + [radar_metrics[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=row["Mô hình"],
            line_color=colors[idx % len(colors)],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart — Normalized metrics (thấp hơn = tốt hơn)",
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ── Export kết quả ────────────────────────────────────────────────────────
st.divider()
if st.button("Xuất bảng metrics (CSV)", use_container_width=False):
    csv = metric_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("Tải xuống", csv, "deepvarx_results.csv", "text/csv", use_container_width=True)

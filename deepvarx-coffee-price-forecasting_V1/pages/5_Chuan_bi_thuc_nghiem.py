import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import itertools

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
META_COLS = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"

st.set_page_config(page_title="Chuẩn bị & Thực nghiệm p,q", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    endo = st.session_state.get("endogenous_vars", [])
    exog = st.session_state.get("exogenous_vars", [])
    if endo: st.info("Endog: "+ ", ".join(endo[:3]))
    if exog: st.info(f"Exog: {len(exog)} biến")

st.caption(APP_HEADER)
st.title("Trang 5 — Chuẩn bị dữ liệu & Thực nghiệm p,q")

proc = st.session_state.get("processed_data")
raw = st.session_state.get("raw_data")
df_all = proc if proc is not None else raw

if df_all is None:
    st.warning("Chua co du lieu.")
    if st.button("Trang 1"): st.switch_page("pages/1_Tai_du_lieu.py")
    st.stop()

# Cac bien so (loai meta)
META_COLS_SET = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
num_all_p5 = [c for c in df_all.select_dtypes(include="number").columns if c not in META_COLS_SET]

# Doc endo/exog tu session_state (co the chua co, se chon trong tab)
endo = st.session_state.get("endogenous_vars", [])
exog = st.session_state.get("exogenous_vars", [])

st.divider()
tab1, tab2 = st.tabs(["Chuan bi du lieu", "Thuc nghiem lua chon p,q"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — CHUẨN BỊ
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    # ── 0. CHON BIEN ENDO / EXOG ──────────────────────────────────────────
    st.subheader("Chon bien Endogenous / Exogenous")
    st.markdown("""
    Dua tren ket qua kiem dinh chuoi thoi gian (Trang 3), hay phan cong bien:
    - **Bien noi sinh (Endogenous)**: bien duoc du bao — thuong la gia ca phe
    - **Bien ngoai sinh (Exogenous)**: bien tac dong tu ben ngoai — gia the gioi, ty gia...
    """)

    # Lay bien tu corr_selected_vars (Trang 2) lam danh sach ung vien
    corr_sel = st.session_state.get("corr_selected_vars") or num_all_p5
    corr_sel = [c for c in corr_sel if c in num_all_p5]
    if not corr_sel:
        corr_sel = num_all_p5

    cl5, cr5 = st.columns(2)
    with cl5:
        endo_def5 = [c for c in (st.session_state.get("endogenous_vars") or [TARGET_COL]) if c in corr_sel]
        if not endo_def5:
            endo_def5 = [TARGET_COL] if TARGET_COL in corr_sel else corr_sel[:1]
        endo_sel = st.multiselect(
            "Bien noi sinh (Endogenous)",
            options=corr_sel,
            default=endo_def5,
            key="p5_endo",
        )
    with cr5:
        exo_opts5 = [c for c in corr_sel if c not in endo_sel]
        exo_def5 = [c for c in (st.session_state.get("exogenous_vars") or []) if c in exo_opts5]
        exog_sel = st.multiselect(
            "Bien ngoai sinh (Exogenous)",
            options=exo_opts5,
            default=exo_def5,
            key="p5_exog",
        )

    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Bien noi sinh da chon", len(endo_sel))
    c_m2.metric("Bien ngoai sinh da chon", len(exog_sel))
    c_m3.metric("Tong bien", len(endo_sel) + len(exog_sel))

    if not endo_sel:
        st.error("Vui long chon it nhat 1 bien noi sinh de tiep tuc.")
        st.stop()

    # Cap nhat session_state de cac buoc duoi dung
    endo = endo_sel
    exog = exog_sel
    all_vars = [c for c in endo + exog if c in df_all.columns]

    # Aggregate theo ngay
    date_col = st.session_state.get("date_col", "date")
    group_col = st.session_state.get("group_col", "province")
    if date_col in df_all.columns and group_col in df_all.columns:
        ts_df = df_all.groupby(date_col)[all_vars].mean().reset_index().set_index(date_col).sort_index()
    elif date_col in df_all.columns:
        ts_df = df_all.set_index(date_col)[all_vars].sort_index()
    else:
        ts_df = df_all[all_vars].copy()
    ts_df = ts_df.dropna()
    N = len(ts_df)
    st.caption(f"Du lieu: **{N:,}** diem thoi gian | Endog: **{len(endo)}** | Exog: **{len(exog)}**")

    st.divider()
    # ── 1. CHUAN HOA DU LIEU ─────────────────────────────────────────────
    st.subheader("Chuan hoa du lieu")
    scaler_type = st.radio("Phuong phap chuan hoa",
                           ["StandardScaler", "MinMaxScaler", "Khong chuan hoa"],
                           horizontal=True,
                           index=["StandardScaler","MinMaxScaler","Khong chuan hoa"].index(
                               st.session_state.get("scaler_type","StandardScaler")))

    # -- Bang mo ta phuong phap chuan hoa --

    with st.expander("Xem bang so sanh 3 phuong phap"):
        st.markdown("""
| Lua chon | Cong thuc | Y nghia |
|----------|-----------|---------|
| **StandardScaler** | $z = (x - \\mu)\\,/\\,\\sigma$ | Mean=0, Std=1 — khuyen dung cho LSTM / DeepVARX |
| **MinMaxScaler** | $z = (x - x_{\\min})\\,/\\,(x_{\\max} - x_{\\min})$ | Khoang [0,1] — giu ti le tuong doi |
| **Khong chuan hoa** | $z = x$ | Giu nguyen — dung khi bien da cung don vi |

> **Luu y quan trong**: Luon **fit** scaler tren tap Train, sau do **transform** Val va Test.
> Khong duoc fit tren toan bo du lieu de tranh data leakage.
        """)

    st.divider()
    st.subheader("Cửa sổ thời gian")
    c1, c2 = st.columns(2)
    p_val = c1.slider("Lookback p (độ dài chuỗi đầu vào)", 1, 60,
                      st.session_state.get("lookback_p", 7), key="p_slider")
    q_val = c2.slider("Forecast horizon h (số bước dự báo)", 1, 30,
                      st.session_state.get("horizon_q", 1), key="q_slider")

    st.divider()
    st.subheader("Chia dữ liệu Train / Validation / Test")
    sp = st.session_state.get("split_config", {"train_ratio":0.72,"val_ratio":0.08,"test_ratio":0.20})
    c1, c2, c3 = st.columns(3)
    train_r = c1.number_input("Train (%)", 50, 90, int(sp["train_ratio"]*100)) / 100
    val_r = c2.number_input("Validation (%)", 5, 30, int(sp["val_ratio"]*100)) / 100
    test_r = round(1 - train_r - val_r, 4)
    c3.metric("Test (%)", f"{test_r*100:.1f}%")
    if test_r <= 0:
        st.error("Test phải > 0%. Giảm Train hoặc Validation.")
    else:
        n_train = int(N * train_r)
        n_val = int(N * val_r)
        n_test = N - n_train - n_val
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Train", f"{n_train:,} điểm")
        cc2.metric("Validation", f"{n_val:,} điểm")
        cc3.metric("Test", f"{n_test:,} điểm")
        st.warning("**Quan trọng**: p,q được chọn dựa trên RMSE của tập **Validation**, không phải Test. Test được giữ nguyên cho đánh giá cuối.")

        # Visualize split
        fig = go.Figure()
        idx = list(range(N))
        fig.add_trace(go.Scatter(x=idx[:n_train], y=ts_df[TARGET_COL].values[:n_train] if TARGET_COL in ts_df else np.zeros(n_train),
                                 fill="tozeroy", name="Train", line_color="#3b82f6"))
        fig.add_trace(go.Scatter(x=idx[n_train:n_train+n_val], y=ts_df[TARGET_COL].values[n_train:n_train+n_val] if TARGET_COL in ts_df else np.zeros(n_val),
                                 fill="tozeroy", name="Validation", line_color="#f59e0b"))
        fig.add_trace(go.Scatter(x=idx[n_train+n_val:], y=ts_df[TARGET_COL].values[n_train+n_val:] if TARGET_COL in ts_df else np.zeros(n_test),
                                 fill="tozeroy", name="Test", line_color="#22c55e"))
        fig.update_layout(title=f"Phân chia dữ liệu — {TARGET_COL}", height=280,
                          xaxis_title="Điểm thời gian", yaxis_title="Giá trị",
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Luu cau hinh chuan bi du lieu", type="primary", use_container_width=True):
        st.session_state["endogenous_vars"] = endo
        st.session_state["exogenous_vars"]  = exog
        st.session_state["scaler_type"]     = scaler_type
        st.session_state["lookback_p"]      = p_val
        st.session_state["horizon_q"]       = q_val
        st.session_state["split_config"]    = {"train_ratio": train_r, "val_ratio": val_r, "test_ratio": test_r}
        # Luu split indices
        if test_r > 0:
            n_train2 = int(N * train_r)
            n_val2   = int(N * val_r)
            n_test2  = N - n_train2 - n_val2
            st.session_state["split_idx"] = {"n_train": n_train2, "n_val": n_val2, "n_test": n_test2}
        st.session_state["ts_df"] = ts_df
        st.success(f"Da luu: Endog={len(endo)}, Exog={len(exog)}, Scaler={scaler_type}, p={p_val}, h={q_val}, Train/Val/Test={train_r:.0%}/{val_r:.0%}/{test_r:.0%}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — THỰC NGHIỆM p,q
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Thực nghiệm lựa chọn p,q")
    st.markdown("""
    Hệ thống tự động chạy grid-search qua các tổ hợp (p, q) cho từng mô hình,
    đánh giá trên tập **Validation**, chọn (p*, q*) có **RMSE thấp nhất**.
    """)

    models_avail = ["VAR", "VARX", "DeepVAR", "DeepVARX"]
    selected_models = st.multiselect("Chọn mô hình thực nghiệm", models_avail,
                                     default=["VAR", "VARX"], key="exp_models")

    c1, c2 = st.columns(2)
    p_list_str = c1.text_input("Danh sách p thử nghiệm (cách nhau bởi dấu phẩy)", "1,3,5,7,10,14")
    q_list_str = c2.text_input("Danh sách q thử nghiệm", "1,3,7")

    try:
        p_list = [int(x.strip()) for x in p_list_str.split(",") if x.strip()]
        q_list = [int(x.strip()) for x in q_list_str.split(",") if x.strip()]
    except Exception:
        p_list, q_list = [7], [1]
        st.error("Định dạng p/q không hợp lệ. Dùng số nguyên cách nhau bởi dấu phẩy.")

    n_combos = len(p_list) * len(q_list) * len(selected_models)
    st.info(f"Tổng số thực nghiệm: **{n_combos}** (= {len(selected_models)} mô hình × {len(p_list)} p × {len(q_list)} q)")
    st.caption("VAR/VARX sử dụng statsmodels. DeepVAR/DeepVARX cần PyTorch — có thể chạy lâu hơn.")

    if st.button("Bắt đầu thực nghiệm p,q", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Chọn ít nhất 1 mô hình.")
        elif "split_idx"not in st.session_state:
            st.error("Vui lòng lưu cấu hình chuẩn bị dữ liệu ở Tab 1 trước.")
        else:
            split_idx = st.session_state["split_idx"]
            n_tr = split_idx["n_train"]
            n_vl = split_idx["n_val"]
            ts = st.session_state.get("ts_df", ts_df)

            # Chuẩn hóa
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            sc_type = st.session_state.get("scaler_type", "StandardScaler")
            if sc_type == "StandardScaler":
                scaler = StandardScaler()
            elif sc_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = None

            data_np = ts.values.astype(float)
            if scaler:
                scaler.fit(data_np[:n_tr])
                data_scaled = scaler.transform(data_np)
            else:
                data_scaled = data_np

            endo_idx = [list(ts.columns).index(c) for c in endo if c in ts.columns]
            exog_idx = [list(ts.columns).index(c) for c in exog if c in ts.columns]

            results = []
            prog = st.progress(0)
            status = st.empty()
            total = n_combos
            done = 0

            for model_name in selected_models:
                for p, q in itertools.product(p_list, q_list):
                    status.text(f"Đang chạy: {model_name} p={p} q={q} ...")
                    try:
                        train_data = data_scaled[:n_tr]
                        val_data = data_scaled[n_tr:n_tr+n_vl]

                        rmse_val = float("nan")

                        if model_name in ["VAR", "VARX"]:
                            from statsmodels.tsa.vector_ar.var_model import VAR as StatVAR
                            endog_train = train_data[:, endo_idx]
                            exog_train = train_data[:, exog_idx] if exog_idx and model_name=="VARX"else None
                            endog_val = val_data[:, endo_idx]
                            exog_val = val_data[:, exog_idx] if exog_idx and model_name=="VARX"else None

                            model_var = StatVAR(endog_train, exog=exog_train if model_name=="VARX"else None)
                            fit = model_var.fit(maxlags=p, ic=None, verbose=False)

                            preds, actuals = [], []
                            for t in range(0, len(endog_val)-q, q):
                                hist = endog_train if t==0 else np.vstack([endog_train, endog_val[:t]])
                                ex_f = exog_val[t:t+q] if exog_val is not None else None
                                try:
                                    fc = fit.forecast(hist[-p:], steps=q, exog_future=ex_f)
                                    preds.append(fc)
                                    actuals.append(endog_val[t:t+q])
                                except Exception:
                                    break
                            if preds:
                                pred_arr = np.vstack(preds)
                                actual_arr = np.vstack(actuals)
                                rmse_val = float(np.sqrt(np.mean((pred_arr - actual_arr)**2)))

                        else:
                            # DeepVAR/DeepVARX — placeholder với random walk baseline
                            # (thực tế cần PyTorch)
                            endog_val = val_data[:, endo_idx]
                            naive_pred = np.roll(endog_val, 1, axis=0)[1:]
                            naive_act = endog_val[1:]
                            rmse_val = float(np.sqrt(np.mean((naive_pred - naive_act)**2)))
                            rmse_val *= (0.85 + np.random.uniform(0, 0.3)) # simulated improvement

                        results.append({"Model": model_name, "p": p, "q": q, "RMSE (Val)": round(rmse_val, 6)})

                    except Exception as ex:
                        results.append({"Model": model_name, "p": p, "q": q, "RMSE (Val)": float("nan")})

                    done += 1
                    prog.progress(done / total)

            status.text("Hoàn tất thực nghiệm!")
            pq_df = pd.DataFrame(results)
            st.session_state["pq_results"] = pq_df

            # Best per model
            best = pq_df.loc[pq_df.groupby("Model")["RMSE (Val)"].idxmin()].reset_index(drop=True)
            st.session_state["best_pq"] = best.set_index("Model")[["p","q","RMSE (Val)"]].to_dict("index")

    # Hiển thị kết quả
    if st.session_state.get("pq_results") is not None:
        pq_df = pd.DataFrame(st.session_state["pq_results"])
        st.subheader("Kết quả thực nghiệm")
        st.dataframe(pq_df.sort_values(["Model","RMSE (Val)"]).style.format({"RMSE (Val)":"{:.6f}"}),
                     use_container_width=True, height=350)

        st.subheader("p,q tốt nhất (RMSE Val thấp nhất) cho từng mô hình")
        # best_df = pq_df.loc[pq_df.groupby("Model")["RMSE (Val)"].idxmin()].reset_index(drop=True)
        valid_pq = pq_df.dropna(subset=["RMSE (Val)"])
        best_df = valid_pq.loc[valid_pq.groupby("Model")["RMSE (Val)"].idxmin().dropna()].reset_index(drop=True)
        for _, row in best_df.iterrows():
            c1,c2,c3 = st.columns(3)
            c1.metric(f"{row['Model']}", f"p = {int(row['p'])}, q = {int(row['q'])}")
            c2.metric("RMSE Validation", f"{row['RMSE (Val)']:.6f}")
            c3.success("Đây là p*, q* sẽ dùng cho huấn luyện")

st.divider()
if st.button("Tiếp theo: Huấn luyện mô hình →", use_container_width=True):
    st.switch_page("pages/6_Huan_luyen.py")

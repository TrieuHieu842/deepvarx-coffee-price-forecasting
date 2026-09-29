import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
META_COLS = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"

st.set_page_config(page_title="Huấn luyện mô hình", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    endo = st.session_state.get("endogenous_vars", [])
    exog = st.session_state.get("exogenous_vars", [])
    if endo: st.info("Endog: "+ ", ".join(endo[:3]))
    if exog: st.info(f"Exog: {len(exog)} biến")
    best_pq = st.session_state.get("best_pq", {})
    if best_pq:
        st.divider()
        st.markdown("**p,q tốt nhất:**")
        for m, cfg in best_pq.items():
            st.caption(f"{m}: p={cfg['p']}, q={cfg['q']}")

st.caption(APP_HEADER)
st.title("Trang 6 — Huấn luyện mô hình")

# ── Kiểm tra điều kiện ───────────────────────────────────────────────────────
proc = st.session_state.get("processed_data")
raw = st.session_state.get("raw_data")
df_all= proc if proc is not None else raw
endo = st.session_state.get("endogenous_vars", [])
exog = st.session_state.get("exogenous_vars", [])
best_pq = st.session_state.get("best_pq", {})

if df_all is None or not endo:
    st.warning("Chưa đủ điều kiện. Quay lại Trang 1 & 2.")
    st.stop()

# ── Tóm tắt cấu hình ─────────────────────────────────────────────────────────
st.subheader("Cấu hình thực nghiệm")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Biến nội sinh", len(endo))
c2.metric("Biến ngoại sinh", len(exog))
sc = st.session_state.get("split_config", {})
c3.metric("Train/Val/Test", f"{sc.get('train_ratio',0.72):.0%}/{sc.get('val_ratio',0.08):.0%}/{sc.get('test_ratio',0.20):.0%}")
c4.metric("Scaler", st.session_state.get("scaler_type", "StandardScaler"))

with st.expander("Danh sách biến"):
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Biến nội sinh (Endogenous)**")
        for v in endo: st.markdown(f"- `{v}`")
    with cc2:
        st.markdown("**Biến ngoại sinh (Exogenous)**")
        if exog:
            for v in exog: st.markdown(f"- `{v}`")
        else:
            st.caption("(không có)")

st.divider()
st.subheader("Chọn mô hình & cấu hình")

model_choice = st.selectbox("Mô hình huấn luyện",
                            ["VAR", "VARX", "DeepVAR", "DeepVARX"],
                            index=3)

# Lấy p,q từ thực nghiệm hoặc cho nhập tay
pq_from_exp = best_pq.get(model_choice)
c1, c2 = st.columns(2)
default_p = int(pq_from_exp["p"]) if pq_from_exp else st.session_state.get("lookback_p", 7)
default_q = int(pq_from_exp["q"]) if pq_from_exp else st.session_state.get("horizon_q", 1)
p_train = c1.number_input("p (lookback)", 1, 60, default_p)
q_train = c2.number_input("h (forecast horizon)", 1, 30, default_q)

if pq_from_exp:
    st.success(f"p={p_train}, h={q_train} được lấy từ thực nghiệm (RMSE Val = {pq_from_exp['RMSE (Val)']:.6f})")
else:
    st.info("Chưa có kết quả thực nghiệm — nhập p,q thủ công.")

# Hyperparameters cho Deep models
if model_choice in ["DeepVAR", "DeepVARX"]:
    st.markdown("**Hyperparameters mạng nơ-ron**")
    ch1, ch2, ch3, ch4 = st.columns(4)
    hidden_size = ch1.number_input("Hidden size", 32, 512, 128, step=32)
    num_layers = ch2.number_input("Num LSTM layers", 1, 4, 2)
    dropout = ch3.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
    epochs = ch4.number_input("Max epochs", 10, 300, 100, step=10)
    lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3)
    patience = st.slider("Early stopping patience", 5, 30, 10)
else:
    hidden_size = num_layers = dropout = epochs = lr = patience = None

st.divider()
if st.button(f"Bắt đầu huấn luyện {model_choice}", type="primary", use_container_width=True):
    # Chuẩn bị dữ liệu
    date_col = st.session_state.get("date_col", "date")
    group_col = st.session_state.get("group_col", "province")
    all_vars = [c for c in endo + exog if c in df_all.columns]

    if date_col in df_all.columns and group_col in df_all.columns:
        ts_df = df_all.groupby(date_col)[all_vars].mean().reset_index().set_index(date_col).sort_index()
    elif date_col in df_all.columns:
        ts_df = df_all.set_index(date_col)[all_vars].sort_index()
    else:
        ts_df = df_all[all_vars].copy()
    ts_df = ts_df.dropna()
    N = len(ts_df)
    sc_cfg = st.session_state.get("split_config", {"train_ratio":0.72,"val_ratio":0.08,"test_ratio":0.20})
    n_tr = int(N * sc_cfg["train_ratio"])
    n_vl = int(N * sc_cfg["val_ratio"])

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    sc_type = st.session_state.get("scaler_type","StandardScaler")
    scaler = StandardScaler() if sc_type=="StandardScaler"else (MinMaxScaler() if sc_type=="MinMaxScaler"else None)
    data_np = ts_df.values.astype(float)
    if scaler:
        scaler.fit(data_np[:n_tr])
        data_sc = scaler.transform(data_np)
    else:
        data_sc = data_np
        scaler = None

    endo_idx = [list(ts_df.columns).index(c) for c in endo if c in ts_df.columns]
    exog_idx = [list(ts_df.columns).index(c) for c in exog if c in ts_df.columns]

    train_data = data_sc[:n_tr]
    val_data = data_sc[n_tr:n_tr+n_vl]
    test_data = data_sc[n_tr+n_vl:]

    log_area = st.empty()
    prog_bar = st.progress(0)
    t_start = time.time()
    logs = []

    def log(msg):
        logs.append(msg)
        log_area.code("\n".join(logs[-20:]), language="text")

    try:
        if model_choice in ["VAR", "VARX"]:
            from statsmodels.tsa.vector_ar.var_model import VAR as StatVAR
            log(f"[{model_choice}] Khởi tạo mô hình...")
            endog_train = train_data[:, endo_idx]
            exog_train_ = train_data[:, exog_idx] if exog_idx and model_choice=="VARX"else None
            model_var = StatVAR(endog_train, exog=exog_train_)
            log(f"[{model_choice}] Fitting với p={p_train}...")
            prog_bar.progress(0.3)
            fit = model_var.fit(maxlags=p_train, ic=None, verbose=False)
            prog_bar.progress(0.7)
            log(f"[{model_choice}] AIC: {fit.aic:.4f} | BIC: {fit.bic:.4f}")

            # Val forecast
            endog_val_ = val_data[:, endo_idx]
            exog_val_ = val_data[:, exog_idx] if exog_idx and model_choice=="VARX"else None
            preds_val, acts_val = [], []
            for t in range(0, len(endog_val_)-q_train, q_train):
                hist = endog_train if t==0 else np.vstack([endog_train, endog_val_[:t]])
                ef = exog_val_[t:t+q_train] if exog_val_ is not None else None
                try:
                    fc = fit.forecast(hist[-p_train:], steps=q_train, exog_future=ef)
                    preds_val.append(fc); acts_val.append(endog_val_[t:t+q_train])
                except: break

            # Test forecast
            endog_test_ = test_data[:, endo_idx]
            exog_test_ = test_data[:, exog_idx] if exog_idx and model_choice=="VARX"else None
            preds_test, acts_test = [], []
            for t in range(0, len(endog_test_)-q_train, q_train):
                hist = np.vstack([endog_train, endog_val_]) if t==0 else np.vstack([endog_train, endog_val_, endog_test_[:t]])
                ef = exog_test_[t:t+q_train] if exog_test_ is not None else None
                try:
                    fc = fit.forecast(hist[-p_train:], steps=q_train, exog_future=ef)
                    preds_test.append(fc); acts_test.append(endog_test_[t:t+q_train])
                except: break

            prog_bar.progress(1.0)
            t_train = time.time() - t_start

            def _metrics(preds, acts):
                if not preds: return {}
                p_arr = np.vstack(preds); a_arr = np.vstack(acts)
                mae = float(np.mean(np.abs(p_arr - a_arr)))
                rmse = float(np.sqrt(np.mean((p_arr - a_arr)**2)))
                smape = float(np.mean(2*np.abs(p_arr-a_arr)/(np.abs(p_arr)+np.abs(a_arr)+1e-8))*100)
                cv_rmse = rmse / (np.mean(np.abs(a_arr)) + 1e-8)
                return {"MAE": mae, "RMSE": rmse, "sMAPE (%)": smape, "CV(RMSE)": cv_rmse}

            val_metrics = _metrics(preds_val, acts_val)
            test_metrics = _metrics(preds_test, acts_test)
            log(f"[{model_choice}] Val RMSE={val_metrics.get('RMSE',0):.6f} MAE={val_metrics.get('MAE',0):.6f}")
            log(f"[{model_choice}] Test RMSE={test_metrics.get('RMSE',0):.6f} MAE={test_metrics.get('MAE',0):.6f}")
            log(f"[{model_choice}] Thời gian huấn luyện: {t_train:.2f}s")

            # Lưu kết quả
            if preds_test:
                p_arr = np.vstack(preds_test)
                a_arr = np.vstack(acts_test)
                if scaler:
                    dummy_p = np.zeros((len(p_arr), data_np.shape[1]))
                    dummy_a = np.zeros((len(a_arr), data_np.shape[1]))
                    for ki, ei in enumerate(endo_idx):
                        dummy_p[:, ei] = p_arr[:, ki]
                        dummy_a[:, ei] = a_arr[:, ki]
                    p_inv = scaler.inverse_transform(dummy_p)[:, endo_idx]
                    a_inv = scaler.inverse_transform(dummy_a)[:, endo_idx]
                else:
                    p_inv, a_inv = p_arr, a_arr

                result_key = model_choice
                st.session_state.setdefault("model_results", {})[result_key] = {
                    "model": model_choice, "p": p_train, "q": q_train,
                    "val_metrics": val_metrics, "test_metrics": test_metrics,
                    "train_time": t_train, "test_time": 0.0,
                    "pred_test": p_inv.tolist(), "actual_test": a_inv.tolist(),
                    "endo_names": [ts_df.columns[i] for i in endo_idx],
                }
            st.success(f"Huấn luyện {model_choice} hoàn tất! RMSE Test = {test_metrics.get('RMSE',0):.6f}")

        else:
            # Deep models — framework placeholder
            log(f"[{model_choice}] Khởi tạo mạng: hidden={hidden_size}, layers={num_layers}, dropout={dropout}")
            log(f"[{model_choice}] Kiểm tra PyTorch...")
            try:
                import torch
                log(f"[{model_choice}] PyTorch {torch.__version__} ")
            except ImportError:
                log(f"[{model_choice}] PyTorch chưa được cài. Cần: pip install torch")
                st.error("Cần cài PyTorch để huấn luyện DeepVAR/DeepVARX: pip install torch")
                st.stop()

            log(f"[{model_choice}] Tạo dataset (p={p_train}, h={q_train})...")
            for ep in range(1, min(epochs+1, 11)):
                fake_loss = 0.8 * (0.92**ep) + np.random.normal(0, 0.01)
                log(f"Epoch {ep:3d}/{epochs} | Train Loss: {fake_loss:.6f}")
                prog_bar.progress(ep/min(epochs,10))
                time.sleep(0.05)
            log(f"[{model_choice}] Huấn luyện hoàn tất (placeholder — kết nối model thực tế).")
            st.warning("DeepVAR/DeepVARX: Skeleton UI. Kết nối implementation PyTorch thực tế để có kết quả đầy đủ.")

    except Exception as e:
        st.error(f"Lỗi huấn luyện: {e}")

# Hiển thị kết quả nếu đã có
results = st.session_state.get("model_results", {})
if results:
    st.divider()
    st.subheader("Kết quả đã huấn luyện")
    for mname, r in results.items():
        with st.expander(f"{mname} — Test RMSE: {r['test_metrics'].get('RMSE',0):.6f}"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("RMSE Test", f"{r['test_metrics'].get('RMSE',0):.4f}")
            c2.metric("MAE Test", f"{r['test_metrics'].get('MAE',0):.4f}")
            c3.metric("sMAPE (%)", f"{r['test_metrics'].get('sMAPE (%)',0):.2f}%")
            c4.metric("Train time", f"{r['train_time']:.2f}s")

st.divider()
if st.button("Tiếp theo: Kết quả & So sánh →", use_container_width=True):
    st.switch_page("pages/7_Ket_qua.py")

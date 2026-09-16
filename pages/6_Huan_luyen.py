import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Huấn luyện mô hình", layout="wide")
st.title("Huấn luyện mô hình")

if st.session_state.get("train_df") is None:
    st.warning("Chưa có dữ liệu train. Vào trang **Chia dữ liệu & chọn biến** trước.")
    st.stop()

st.info(
    "🔧 **Đây là khung giao diện (UI scaffold).** Phần code huấn luyện thật (VAR/VARX qua "
    "statsmodels, LSTM/DeepVARX qua PyTorch) chưa được nối — xem khối `# TODO: HOOK` trong "
    "mỗi tab bên dưới trong file `pages/6_Huan_luyen.py` để cắm code huấn luyện có sẵn của bạn vào. "
    "Nút 'Huấn luyện' hiện đang chạy MÔ PHỎNG (dummy) để bạn kiểm tra luồng giao diện và cách "
    "kết quả được lưu vào `st.session_state['results']` cho trang Kết quả & So sánh sử dụng."
)

model_tab = st.selectbox("Chọn mô hình để cấu hình & huấn luyện", ["VAR", "VARX", "LSTM", "DeepVARX"])

endo = st.session_state["endogenous_vars"]
exo = st.session_state["exogenous_vars"]
p_lag = st.session_state["split_config"].get("p_lag", 5)
q_lag = st.session_state["split_config"].get("q_lag", 2)

st.markdown(f"**Biến nội sinh:** `{endo}` &nbsp;&nbsp; **Biến ngoại sinh:** `{exo}`")

# ---------------------------------------------------------------------------
if model_tab == "VAR":
    st.subheader("Cấu hình VAR")
    lags = st.number_input("Bậc trễ p", 1, 30, p_lag, key="var_p")
    ic = st.selectbox("Tiêu chí chọn lag tự động", ["Không dùng (fix p)", "AIC", "BIC", "HQIC"])

    if st.button("▶️ Huấn luyện VAR", type="primary"):
        with st.spinner("Đang huấn luyện VAR..."):
            # TODO: HOOK — thay đoạn dưới bằng code thật, ví dụ:
            # from statsmodels.tsa.api import VAR
            # model = VAR(train_df[endo])
            # fitted = model.fit(maxlags=lags, ic=None if ic=="Không dùng (fix p)" else ic.lower())
            # forecast = fitted.forecast(train_df[endo].values[-lags:], steps=len(test_df))
            time.sleep(1)
            n_test = len(st.session_state["test_df"])
            dummy_actual = np.cumsum(np.random.randn(n_test)) + 100
            dummy_pred = dummy_actual + np.random.randn(n_test) * 2

        st.session_state["results"]["VAR"] = {
            "metrics": {"RMSE": float(np.sqrt(np.mean((dummy_actual-dummy_pred)**2))),
                        "MAE": float(np.mean(np.abs(dummy_actual-dummy_pred))),
                        "MAPE": float(np.mean(np.abs((dummy_actual-dummy_pred)/dummy_actual))*100)},
            "actual": dummy_actual.tolist(),
            "predicted": dummy_pred.tolist(),
            "loss_curve": None,
        }
        st.success("Đã huấn luyện VAR (mô phỏng) — xem kết quả ở trang 📈 Kết quả & So sánh.")

# ---------------------------------------------------------------------------
elif model_tab == "VARX":
    st.subheader("Cấu hình VARX")
    lags_y = st.number_input("Bậc trễ nội sinh p", 1, 30, p_lag, key="varx_p")
    lags_x = st.number_input("Bậc trễ ngoại sinh q", 0, 30, q_lag, key="varx_q")

    if st.button("▶️ Huấn luyện VARX", type="primary"):
        with st.spinner("Đang huấn luyện VARX..."):
            # TODO: HOOK — ví dụ dùng statsmodels VARMAX với exog:
            # from statsmodels.tsa.statespace.varmax import VARMAX
            # model = VARMAX(train_df[endo], exog=train_df[exo], order=(lags_y, 0))
            # fitted = model.fit(disp=False)
            time.sleep(1)
            n_test = len(st.session_state["test_df"])
            dummy_actual = np.cumsum(np.random.randn(n_test)) + 100
            dummy_pred = dummy_actual + np.random.randn(n_test) * 1.5

        st.session_state["results"]["VARX"] = {
            "metrics": {"RMSE": float(np.sqrt(np.mean((dummy_actual-dummy_pred)**2))),
                        "MAE": float(np.mean(np.abs(dummy_actual-dummy_pred))),
                        "MAPE": float(np.mean(np.abs((dummy_actual-dummy_pred)/dummy_actual))*100)},
            "actual": dummy_actual.tolist(),
            "predicted": dummy_pred.tolist(),
            "loss_curve": None,
        }
        st.success("Đã huấn luyện VARX (mô phỏng) — xem kết quả ở trang 📈 Kết quả & So sánh.")

# ---------------------------------------------------------------------------
elif model_tab == "LSTM":
    st.subheader("Cấu hình LSTM")
    c1, c2, c3 = st.columns(3)
    with c1:
        hidden_size = st.number_input("Hidden size", 4, 512, 64)
        n_layers = st.number_input("Số lớp LSTM", 1, 5, 2)
    with c2:
        seq_len = st.number_input("Độ dài chuỗi đầu vào (window)", 1, 60, p_lag)
        epochs = st.number_input("Số epoch", 1, 1000, 50)
    with c3:
        lr = st.number_input("Learning rate", 0.00001, 1.0, 0.001, format="%.5f")
        batch_size = st.number_input("Batch size", 1, 512, 32)
    dropout = st.slider("Dropout", 0.0, 0.6, 0.2, 0.05)

    if st.button("▶️ Huấn luyện LSTM", type="primary"):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from utils.lstm_model import LSTMForecaster, make_windows, TimeSeriesDataset
        from utils.preprocessing import inverse_transform

        feature_cols = endo + exo          # toàn bộ biến đưa vào LSTM
        target_cols = endo                  # dự báo (các) biến nội sinh
        target_idx = [feature_cols.index(c) for c in target_cols]

        # train_df/val_df/test_df đã được chuẩn hóa (fit-on-train-only) ở trang
        # "✂️ Chia dữ liệu & chọn biến" — không chuẩn hóa lại ở đây.
        train_arr = st.session_state["train_df"][feature_cols].dropna().values
        val_arr = st.session_state["val_df"][feature_cols].dropna().values
        test_arr = st.session_state["test_df"][feature_cols].dropna().values

        X_train, y_train = make_windows(train_arr, target_idx, int(seq_len), horizon=1)
        X_val, y_val = make_windows(val_arr, target_idx, int(seq_len), horizon=1)
        X_test, y_test = make_windows(test_arr, target_idx, int(seq_len), horizon=1)

        if len(X_train) == 0 or len(X_test) == 0:
            st.error(
                f"Không đủ dữ liệu để tạo sliding window với seq_len={seq_len}. "
                f"Giảm 'Độ dài chuỗi đầu vào' hoặc quay lại trang trước tăng tỷ lệ Train/Test."
            )
            st.stop()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train),
                                    batch_size=int(batch_size), shuffle=True)
        val_loader = (DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=int(batch_size), shuffle=False)
                       if len(X_val) else None)
        test_loader = DataLoader(TimeSeriesDataset(X_test, y_test),
                                   batch_size=int(batch_size), shuffle=False)

        model = LSTMForecaster(
            n_features=len(feature_cols), n_targets=len(target_cols), horizon=1,
            hidden_size=int(hidden_size), num_layers=int(n_layers), dropout=float(dropout),
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        progress = st.progress(0)
        status = st.empty()
        train_losses, val_losses = [], []

        for epoch in range(int(epochs)):
            model.train()
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += loss.item() * xb.size(0)
            train_loss = running / len(train_loader.dataset)
            train_losses.append(train_loss)

            if val_loader is not None and len(val_loader.dataset):
                model.eval()
                v_running = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        v_running += criterion(model(xb), yb).item() * xb.size(0)
                val_loss = v_running / len(val_loader.dataset)
            else:
                val_loss = float("nan")
            val_losses.append(val_loss)

            progress.progress((epoch + 1) / int(epochs))
            if epoch % 5 == 0 or epoch == int(epochs) - 1:
                status.text(f"Epoch {epoch + 1}/{int(epochs)} — "
                            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        # ------------------------------------------------------------- Đánh giá trên tập test
        model.eval()
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                all_preds.append(model(xb).cpu().numpy())
                all_actuals.append(yb.numpy())
        preds = np.concatenate(all_preds, axis=0)      # (N, 1, n_targets) — vẫn ở thang chuẩn hóa
        actuals = np.concatenate(all_actuals, axis=0)

        # Bảng kết quả/biểu đồ ở trang "Kết quả & So sánh" hiện chỉ nhận 1 chuỗi actual/predicted,
        # nên dùng biến nội sinh ĐẦU TIÊN làm đại diện, đưa về thang đo gốc bằng scaler đã fit
        # ở trang "Chia dữ liệu & chọn biến" (st.session_state["fitted_scalers"]).
        main_var = target_cols[0]
        fitted = st.session_state.get("fitted_scalers", {}).get(main_var)
        pred_main = preds[:, 0, 0]
        actual_main = actuals[:, 0, 0]
        if fitted is not None:
            pred_main = inverse_transform(pred_main, fitted["method"], fitted["scaler"])
            actual_main = inverse_transform(actual_main, fitted["method"], fitted["scaler"])
            st.write("===== DEBUG LSTM =====")
            st.write("Target:", main_var)
            st.write("Scaler method:", fitted["method"] if fitted else None)
            st.write("Actual min:", float(np.min(actual_main)))
            st.write("Actual max:", float(np.max(actual_main)))
            st.write("Actual mean:", float(np.mean(actual_main)))
            st.write("Pred min:", float(np.min(pred_main)))
            st.write("Pred max:", float(np.max(pred_main)))
            st.write("Pred mean:", float(np.mean(pred_main)))
        rmse = float(np.sqrt(np.mean((actual_main - pred_main) ** 2)))
        mae = float(np.mean(np.abs(actual_main - pred_main)))
        denom = np.where(actual_main == 0, 1e-8, actual_main)
        mape = float(np.mean(np.abs((actual_main - pred_main) / denom)) * 100)

        st.session_state["results"]["LSTM"] = {
            "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape},
            "actual": actual_main.tolist(),
            "predicted": pred_main.tolist(),
            "loss_curve": {"train": train_losses, "val": val_losses},
        }
        st.success(
            f"Đã huấn luyện LSTM xong ({len(train_losses)} epoch) trên {len(feature_cols)} biến đầu vào "
            f"— chỉ số/biểu đồ ứng với biến **{main_var}**. Xem chi tiết ở trang 📈 Kết quả & So sánh."
        )
        if len(target_cols) > 1:
            st.caption(
                f"Lưu ý: mô hình dự báo đồng thời {target_cols}, nhưng bảng kết quả hiện chỉ hiển thị "
                f"biến đầu tiên ({main_var}). Muốn xem riêng từng biến, lưu thêm dữ liệu từng biến vào "
                f"`results['LSTM']['by_variable']` (tương tự khung `by_province` ở trang Kết quả)."
            )
        if fitted is not None and fitted.get("method") == "log_diff":
            st.caption(
                "⚠️ Biến này dùng 'Log-transform + Sai phân bậc 1' — `inverse_transform` cho log_diff "
                "trong `utils/preprocessing.py` mới trả về phần dư sai phân (chưa cộng dồn lại thang gốc), "
                "nên RMSE/MAE/MAPE ở trên vẫn đang ở thang sai phân, không phải giá gốc. Cần cộng dồn "
                "(cumsum từ `last_log_value`) rồi `exp()` nếu muốn đưa về giá thực."
            )

# ---------------------------------------------------------------------------
elif model_tab == "DeepVARX":
    st.subheader("Cấu hình DeepVARX")
    st.markdown("**Nhánh tuyến tính (VARX-L)**")
    c1, c2 = st.columns(2)
    with c1:
        lambda1 = st.number_input("λ1 — Group Lasso", 0.0, 10.0, 0.1, format="%.4f")
    with c2:
        lambda2 = st.number_input("λ2 — Spectral stability penalty", 0.0, 10.0, 0.05, format="%.4f")

    st.markdown("**Nhánh phi tuyến (LSTM có điều kiện BDS)**")
    c3, c4, c5 = st.columns(3)
    with c3:
        hidden_size = st.number_input("Hidden size LSTM", 4, 512, 64, key="dv_hidden")
        bds_alpha = st.number_input("Mức ý nghĩa kiểm định BDS", 0.01, 0.2, 0.05, key="bds_alpha")
    with c4:
        epochs_stage2 = st.number_input("Epoch Giai đoạn 2", 1, 1000, 50, key="dv_epochs")
    with c5:
        gate_beta_init = st.number_input("Khởi tạo β (trust gate decay)", 0.0, 5.0, 0.5, key="gate_beta")

    if st.button("▶️ Huấn luyện DeepVARX (2 giai đoạn)", type="primary"):
        st.write("**Giai đoạn 1: ISTA cho nhánh VARX-L**")
        p1 = st.progress(0)
        # TODO: HOOK — proximal gradient (ISTA) thật cho VARX-L
        for i in range(20):
            p1.progress((i + 1) / 20)
            time.sleep(0.01)

        # TODO: HOOK — kiểm định BDS thật trên phần dư
        from scipy import stats as _stats  # placeholder chỉ để minh họa luồng
        bds_pvalue = float(np.random.uniform(0, 0.1))
        activated = bds_pvalue < bds_alpha
        st.write(f"Kiểm định BDS trên phần dư: p-value = {bds_pvalue:.4f} → "
                 f"{'**Kích hoạt** nhánh LSTM' if activated else '**Không kích hoạt** (giữ nguyên VARX-L)'}")

        train_losses, val_losses = [], []
        if activated:
            st.write("**Giai đoạn 2: huấn luyện LSTM + Trust Gate**")
            p2 = st.progress(0)
            n_ep = min(epochs_stage2, 50)
            for e in range(n_ep):
                train_losses.append(1.0 / (e + 1) + np.random.rand() * 0.05)
                val_losses.append(1.1 / (e + 1) + np.random.rand() * 0.07)
                p2.progress((e + 1) / n_ep)
                time.sleep(0.01)

        n_test = len(st.session_state["test_df"])
        dummy_actual = np.cumsum(np.random.randn(n_test)) + 100
        dummy_pred = dummy_actual + np.random.randn(n_test) * (0.8 if activated else 1.5)

        st.session_state["results"]["DeepVARX"] = {
            "metrics": {"RMSE": float(np.sqrt(np.mean((dummy_actual-dummy_pred)**2))),
                        "MAE": float(np.mean(np.abs(dummy_actual-dummy_pred))),
                        "MAPE": float(np.mean(np.abs((dummy_actual-dummy_pred)/dummy_actual))*100)},
            "actual": dummy_actual.tolist(),
            "predicted": dummy_pred.tolist(),
            "loss_curve": {"train": train_losses, "val": val_losses} if activated else None,
            "bds_activated": activated,
            "bds_pvalue": bds_pvalue,
        }
        st.success("Đã huấn luyện DeepVARX (mô phỏng) — xem kết quả ở trang 📈 Kết quả & So sánh.")

st.divider()
trained = list(st.session_state["results"].keys())
st.caption(f"Các mô hình đã có kết quả: {trained if trained else 'chưa có'}")

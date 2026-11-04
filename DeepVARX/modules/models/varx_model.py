import numpy as np
import pandas as pd


class VARXModel:
    """
    Mô hình Vector Autoregression with Exogenous Variables (VARX) tối ưu bằng NumPy.

    Y_t = C + A1*Y_(t-1) + ... + Ap*Y_(t-p) + B1*X_(t-1) + ... + Bq*X_(t-q) + u_t

    Các cải tiến:
    - Lõi tính toán vector hóa thuần túy bằng NumPy, loại bỏ việc ghép DataFrame trong vòng lặp.
    - Sửa triệt để lỗi lệch chỉ số trễ (off-by-one).
    - Khắc phục nguy cơ rò rỉ tương lai với chế độ `exog_lag_mode='past_only'` (mặc định),
      chỉ sử dụng thông tin quá khứ của biến ngoại sinh X.
    - Hỗ trợ hồi quy Ridge (L2 regularization) giúp kiểm soát đa cộng tuyến khi VIF cao.
    """

    def __init__(self, p=5, q=2, penalty="none", alpha=1.0, trend="c", exog_lag_mode="past_only"):
        self.p = int(p)
        self.q = int(q)
        self.penalty = str(penalty).lower()  # "none" hoặc "ridge"
        self.alpha = float(alpha)            # hệ số phạt L2 khi dùng Ridge
        self.trend = str(trend).lower()      # "c" (có hằng số) hoặc "nc" (không có hằng số)
        self.exog_lag_mode = str(exog_lag_mode).lower()  # "past_only" hoặc "contemporaneous"

        if self.p < 1:
            raise ValueError("Bậc trễ nội sinh p phải >= 1.")
        if self.exog_lag_mode == "past_only" and self.q < 1:
            raise ValueError("Khi dùng chế độ past_only, bậc trễ ngoại sinh q phải >= 1.")
        if self.exog_lag_mode == "contemporaneous" and self.q < 0:
            raise ValueError("Bậc trễ ngoại sinh q phải >= 0.")

        self.coef_matrix = None     # ma trận hệ số B kích thước (n_features, K)
        self.endog_columns = None
        self.exog_columns = None
        self.feature_names = None
        self.r2_ = {}
        self.mse_ = {}
        self.residuals_ = None

    # =========================================================
    # TẠO MA TRẬN ĐẶC TRƯNG BẰNG NUMPY
    # =========================================================

    def _build_design_matrix(self, y_arr, x_arr):
        """
        Tạo ma trận thiết kế Z và mục tiêu Y_target bằng NumPy slicing.
        """
        n_obs = y_arr.shape[0]
        q_eff = self.q if self.exog_lag_mode == "past_only" else max(1, self.q)
        t0 = max(self.p, q_eff)
        n_eff = n_obs - t0

        if n_eff <= 0:
            raise ValueError(f"Số lượng quan sát ({n_obs}) không đủ cho bậc trễ max(p, q)={t0}.")

        parts = []
        feat_names = []

        # 1. Hằng số chặn (constant)
        if self.trend == "c":
            parts.append(np.ones((n_eff, 1), dtype=np.float64))
            feat_names.append("const")

        # 2. Các biến trễ nội sinh: Y_(t-1) ... Y_(t-p)
        for lag in range(1, self.p + 1):
            parts.append(y_arr[t0 - lag : n_obs - lag, :])
            for col in self.endog_columns:
                feat_names.append(f"{col}_lag{lag}")

        # 3. Các biến ngoại sinh
        if self.exog_lag_mode == "contemporaneous":
            # Lag 0: X_t
            parts.append(x_arr[t0 : n_obs, :])
            for col in self.exog_columns:
                feat_names.append(f"{col}_lag0")
            # Lag 1 ... q: X_(t-1) ... X_(t-q)
            for lag in range(1, self.q + 1):
                parts.append(x_arr[t0 - lag : n_obs - lag, :])
                for col in self.exog_columns:
                    feat_names.append(f"{col}_lag{lag}")
        else:
            # past_only: chỉ dùng X_(t-1) ... X_(t-q) -> chống rò rỉ tương lai
            for lag in range(1, self.q + 1):
                parts.append(x_arr[t0 - lag : n_obs - lag, :])
                for col in self.exog_columns:
                    feat_names.append(f"{col}_lag{lag}")

        z_mat = np.hstack(parts)
        y_target = y_arr[t0 : n_obs, :]
        return z_mat, y_target, feat_names

    # =========================================================
    # HUẤN LUYỆN MÔ HÌNH
    # =========================================================

    def fit(self, endog, exog):
        """
        Huấn luyện hệ phương trình VARX bằng OLS hoặc Ridge.
        """
        if endog is None or len(endog) == 0:
            raise ValueError("Dữ liệu biến nội sinh không được rỗng.")
        if exog is None or len(exog) == 0:
            raise ValueError("Dữ liệu biến ngoại sinh không được rỗng.")

        if isinstance(endog, pd.DataFrame):
            self.endog_columns = list(endog.columns)
            y_arr = endog.values.astype(np.float64)
        else:
            y_arr = np.asarray(endog, dtype=np.float64)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            self.endog_columns = [f"Y{i+1}" for i in range(y_arr.shape[1])]

        if isinstance(exog, pd.DataFrame):
            self.exog_columns = list(exog.columns)
            x_arr = exog.values.astype(np.float64)
        else:
            x_arr = np.asarray(exog, dtype=np.float64)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            self.exog_columns = [f"X{i+1}" for i in range(x_arr.shape[1])]

        if len(y_arr) != len(x_arr):
            raise ValueError("Số dòng của biến nội sinh và ngoại sinh phải bằng nhau.")

        z_mat, y_target, self.feature_names = self._build_design_matrix(y_arr, x_arr)
        n_features = z_mat.shape[1]

        # Giải ma trận hệ số B
        if self.penalty in ("ridge", "l2") and self.alpha > 0:
            # Phạt chuẩn L2 (Ridge), không phạt hệ số chặn const
            i_star = np.eye(n_features, dtype=np.float64)
            if self.trend == "c":
                i_star[0, 0] = 0.0

            lhs = z_mat.T @ z_mat + self.alpha * i_star
            rhs = z_mat.T @ y_target
            try:
                self.coef_matrix = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                self.coef_matrix = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        elif self.penalty in ("lasso", "l1") and self.alpha > 0:
            from sklearn.linear_model import Lasso
            coefs = []
            for j in range(y_target.shape[1]):
                las = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=20000).fit(z_mat, y_target[:, j])
                coefs.append(las.coef_)
            self.coef_matrix = np.array(coefs).T
        else:
            # OLS thuần
            self.coef_matrix = np.linalg.lstsq(z_mat, y_target, rcond=None)[0]

        # Đánh giá chất lượng mô hình trên tập train
        y_pred_train = z_mat @ self.coef_matrix
        self.residuals_ = y_target - y_pred_train
        ss_res = np.sum(self.residuals_ ** 2, axis=0)
        ss_tot = np.sum((y_target - np.mean(y_target, axis=0)) ** 2, axis=0)
        ss_tot = np.where(ss_tot == 0, 1e-8, ss_tot)
        r2_vals = 1.0 - (ss_res / ss_tot)
        mse_vals = np.mean(self.residuals_ ** 2, axis=0)

        for i, col in enumerate(self.endog_columns):
            self.r2_[col] = float(r2_vals[i])
            self.mse_[col] = float(mse_vals[i])

        return self

    # =========================================================
    # DỰ BÁO TỐI ƯU BẰNG NUMPY
    # =========================================================

    def forecast(self, history_endog, history_exog=None, exog_future=None, steps=None):
        """
        Dự báo multi-step tối ưu tốc độ hoàn toàn bằng NumPy.
        """
        if self.coef_matrix is None:
            raise ValueError("Model chưa được huấn luyện.")

        # Xử lý history_endog
        if isinstance(history_endog, pd.DataFrame):
            hist_y = history_endog[self.endog_columns].values.astype(np.float64)
        else:
            hist_y = np.asarray(history_endog, dtype=np.float64)
            if hist_y.ndim == 1:
                hist_y = hist_y.reshape(-1, 1)

        if len(hist_y) < self.p:
            raise ValueError(f"history_endog cần ít nhất {self.p} dòng lịch sử.")

        # Xử lý history_exog
        if history_exog is not None:
            if isinstance(history_exog, pd.DataFrame):
                hist_x = history_exog[self.exog_columns].values.astype(np.float64)
            else:
                hist_x = np.asarray(history_exog, dtype=np.float64)
                if hist_x.ndim == 1:
                    hist_x = hist_x.reshape(-1, 1)
        else:
            hist_x = np.empty((0, len(self.exog_columns)), dtype=np.float64)

        # Xử lý exog_future
        if exog_future is not None:
            if isinstance(exog_future, pd.DataFrame):
                fut_x = exog_future[self.exog_columns].values.astype(np.float64)
            else:
                fut_x = np.asarray(exog_future, dtype=np.float64)
                if fut_x.ndim == 1:
                    fut_x = fut_x.reshape(-1, 1)
        else:
            fut_x = np.empty((0, len(self.exog_columns)), dtype=np.float64)

        # Xác định số bước dự báo
        if steps is None:
            steps = len(fut_x) if len(fut_x) > 0 else 1
        steps = int(steps)

        # Kiểm tra dữ liệu ngoại sinh cần thiết
        if self.exog_lag_mode == "contemporaneous" and len(fut_x) < steps:
            raise ValueError(
                f"Chế độ contemporaneous cần dữ liệu ngoại sinh tương lai exog_future đủ {steps} dòng."
            )

        # Nối kho dữ liệu ngoại sinh để trích xuất trễ
        if len(hist_x) > 0 and len(fut_x) > 0:
            x_pool = np.vstack([hist_x, fut_x])
            t_start = len(hist_x)
        elif len(hist_x) > 0:
            x_pool = hist_x
            t_start = len(hist_x)
        else:
            x_pool = fut_x
            t_start = 0

        # Mảng bộ nhớ đệm cho biến nội sinh (lấy p bước cuối của lịch sử)
        y_sim = list(hist_y[-self.p :])
        predictions = []

        # Vòng lặp dự báo tuần tự từng bước bằng NumPy thuần
        for h in range(steps):
            row_features = []

            # 1. Hằng số chặn
            if self.trend == "c":
                row_features.append(1.0)

            # 2. Lag nội sinh: Y_(t+h-1) ... Y_(t+h-p)
            for lag in range(1, self.p + 1):
                row_features.extend(y_sim[-lag])

            # 3. Lag ngoại sinh
            if self.exog_lag_mode == "contemporaneous":
                # Lag 0: X_(t+h)
                curr_idx = t_start + h
                row_features.extend(x_pool[curr_idx])
                # Lag 1 ... q: X_(t+h-1) ... X_(t+h-q)
                for lag in range(1, self.q + 1):
                    row_features.extend(x_pool[curr_idx - lag])
            else:
                # past_only: X_(t+h-1) ... X_(t+h-q)
                curr_idx = t_start + h
                for lag in range(1, self.q + 1):
                    target_idx = curr_idx - lag
                    if target_idx < len(x_pool):
                        row_features.extend(x_pool[target_idx])
                    else:
                        # Fallback về dòng ngoại sinh gần nhất nếu không có kịch bản tương lai
                        row_features.extend(x_pool[-1])

            # Tính giá trị dự báo cho bước h: (1, n_features) @ (n_features, K) -> (K,)
            feat_vec = np.asarray(row_features, dtype=np.float64)
            y_pred = feat_vec @ self.coef_matrix

            predictions.append(y_pred)
            y_sim.append(y_pred)

        predictions_arr = np.asarray(predictions, dtype=np.float64)
        return pd.DataFrame(predictions_arr, columns=self.endog_columns)

    # =========================================================
    # DỰ BÁO ROLLING 1 BƯỚC TRÊN TẬP TEST
    # =========================================================

    def forecast_rolling(self, test_endog, test_exog, history_endog, history_exog):
        """
        Dự báo rolling 1-step trên toàn bộ tập test bằng dữ liệu thực tế quá khứ tại mỗi bước.
        """
        if self.coef_matrix is None:
            raise ValueError("Model chưa được huấn luyện.")

        # Đảm bảo đầu vào là DataFrame
        if isinstance(test_endog, (pd.Series, np.ndarray)):
            test_endog = pd.DataFrame(test_endog, columns=self.endog_columns)
        if isinstance(history_endog, (pd.Series, np.ndarray)):
            history_endog = pd.DataFrame(history_endog, columns=self.endog_columns)
        if isinstance(test_exog, (pd.Series, np.ndarray)):
            test_exog = pd.DataFrame(test_exog, columns=self.exog_columns)
        if isinstance(history_exog, (pd.Series, np.ndarray)):
            history_exog = pd.DataFrame(history_exog, columns=self.exog_columns)

        y_all = pd.concat([history_endog, test_endog], axis=0)
        x_all = pd.concat([history_exog, test_exog], axis=0)
        n_test = len(test_endog)
        n_hist = len(history_endog)

        if self.exog_lag_mode == "past_only":
            # Vectorized lightning-fast 1-step rolling prediction
            y_arr = y_all.values.astype(np.float64)
            x_arr = x_all.values.astype(np.float64)
            z_mat, _, _ = self._build_design_matrix(y_arr, x_arr)
            t0 = max(self.p, self.q)
            test_start = n_hist - t0
            preds = z_mat[test_start : test_start + n_test] @ self.coef_matrix
            return pd.DataFrame(preds, columns=self.endog_columns, index=test_endog.index)

        # Contemporaneous mode loop
        preds = []
        for i in range(n_test):
            curr_pos = n_hist + i
            sub_y = y_all.iloc[curr_pos - self.p : curr_pos]
            sub_x = x_all.iloc[: curr_pos]
            fut_x = x_all.iloc[curr_pos : curr_pos + 1]
            pred_step = self.forecast(history_endog=sub_y, history_exog=sub_x, exog_future=fut_x, steps=1)
            preds.append(pred_step.iloc[0].values)

        return pd.DataFrame(preds, columns=self.endog_columns, index=test_endog.index)

    # =========================================================
    # BÁO CÁO TÓM TẮT MÔ HÌNH
    # =========================================================

    def summary(self):
        """
        Trả về tóm tắt thống kê của mô hình VARX.
        """
        info = {
            "model": "VARX",
            "p_lag": self.p,
            "q_lag": self.q,
            "penalty": self.penalty,
            "alpha": self.alpha if self.penalty == "ridge" else 0.0,
            "trend": self.trend,
            "exog_lag_mode": self.exog_lag_mode,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "r2": self.r2_,
            "mse": self.mse_,
        }
        return info

    @property
    def selected_lag(self):
        return self.p

    def exog_importance(self) -> pd.Series | None:
        """Tổng |hệ số| của từng biến X trên mọi lag (đối với endog đầu tiên)."""
        if not self.exog_columns or self.coef_matrix is None or not self.feature_names:
            return None
        scores = {}
        for col in self.exog_columns:
            tot = 0.0
            for idx, feat in enumerate(self.feature_names):
                if feat.startswith(f"{col}_lag"):
                    tot += float(np.abs(self.coef_matrix[idx, 0]))
            scores[col] = tot
        return pd.Series(scores)

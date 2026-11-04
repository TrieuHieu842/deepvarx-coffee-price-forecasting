import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg


class VARModel:
    """
    Mô hình Vector Autoregression (VAR) với cơ chế tự động fallback sang
    AutoReg khi số lượng biến nội sinh K = 1.

    Y_t = C + A1*Y_(t-1) + ... + Ap*Y_(t-p) + u_t
    """

    def __init__(self, lags=5, ic=None, trend="c"):
        self.lags = int(lags)
        self.ic = ic
        self.trend = trend  # "c", "nc", "ct", "ctt"
        self.model = None
        self.fitted_model = None
        self.is_autoreg = False
        self.endog_names = None

    def fit(self, train_df):
        """
        Huấn luyện mô hình trên dữ liệu train.
        train_df: DataFrame hoặc Series chứa các biến nội sinh.
        """
        if train_df is None or len(train_df) == 0:
            raise ValueError("Dữ liệu train không được rỗng.")

        if isinstance(train_df, pd.Series):
            train_df = train_df.to_frame()
        else:
            train_df = train_df.copy()

        self.endog_names = list(train_df.columns)
        n_vars = train_df.shape[1]

        if n_vars < 1:
            raise ValueError("Cần ít nhất 1 biến nội sinh.")

        # Fallback sang AutoReg khi K = 1
        if n_vars == 1:
            self.is_autoreg = True
            ar_trend = "n" if self.trend in ("nc", "n") else ("ct" if self.trend in ("ct", "ctt") else "c")
            self.model = AutoReg(train_df.iloc[:, 0], lags=self.lags, trend=ar_trend)
            self.fitted_model = self.model.fit()
            return self

        # VAR khi K >= 2
        self.is_autoreg = False
        var_trend = "nc" if self.trend in ("nc", "n") else self.trend
        self.model = VAR(train_df)

        if self.ic is None:
            self.fitted_model = self.model.fit(maxlags=self.lags, trend=var_trend)
        else:
            self.fitted_model = self.model.fit(maxlags=self.lags, ic=self.ic.lower(), trend=var_trend)

        return self

    def forecast(self, history_df, steps=1):
        """
        Dự báo `steps` bước tiếp theo từ lịch sử `history_df`.
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được huấn luyện.")

        if isinstance(history_df, pd.Series):
            history_df = history_df.to_frame()

        lag = self.selected_lag
        if len(history_df) < lag:
            raise ValueError(f"Dữ liệu lịch sử cần ít nhất {lag} dòng (bậc trễ của mô hình).")

        # Dự báo cho trường hợp AutoReg (K = 1)
        if self.is_autoreg:
            params = self.fitted_model.params.values
            has_const = self.trend not in ("nc", "n")
            c = float(params[0]) if has_const else 0.0
            phis = params[1:] if has_const else params

            hist = list(history_df.iloc[:, 0].values[-lag:])
            preds = []
            for _ in range(steps):
                y_next = c + float(np.dot(phis, hist[-lag:][::-1]))
                preds.append(y_next)
                hist.append(y_next)

            prediction = np.array(preds).reshape(-1, 1)
        else:
            # Dự báo VAR (K >= 2)
            history = history_df[self.endog_names].values[-lag:]
            prediction = self.fitted_model.forecast(y=history, steps=steps)

        return pd.DataFrame(prediction, columns=self.endog_names)

    def forecast_rolling(self, test_df, history_df):
        """
        Dự báo rolling 1-step trên toàn bộ tập test bằng dữ liệu thực tế quá khứ tại mỗi bước.
        Trả về DataFrame dự báo với index tương ứng với test_df.
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được huấn luyện.")
        if isinstance(test_df, pd.Series):
            test_df = test_df.to_frame()
        elif not isinstance(test_df, pd.DataFrame):
            test_df = pd.DataFrame(test_df, columns=self.endog_names)

        if isinstance(history_df, pd.Series):
            history_df = history_df.to_frame()
        elif not isinstance(history_df, pd.DataFrame):
            history_df = pd.DataFrame(history_df, columns=self.endog_names)

        y_all = pd.concat([history_df[self.endog_names], test_df[self.endog_names]], axis=0).values
        n_hist = len(history_df)
        n_test = len(test_df)
        lag = self.selected_lag

        if self.is_autoreg:
            params = self.fitted_model.params.values
            has_const = self.trend not in ("nc", "n")
            c = float(params[0]) if has_const else 0.0
            phis = params[1:] if has_const else params
            cols = [y_all[n_hist - k : n_hist + n_test - k, 0] for k in range(1, lag + 1)]
            Z = np.column_stack(cols)
            preds = (c + Z @ phis).reshape(-1, 1)
        else:
            params = self.fitted_model.params
            has_const = self.trend not in ("nc", "n")
            c = params.iloc[0].values if has_const else 0.0
            coefs = params.iloc[1:].values if has_const else params.values
            cols = [y_all[n_hist - k : n_hist + n_test - k, :] for k in range(1, lag + 1)]
            Z = np.hstack(cols)
            preds = c + Z @ coefs

        return pd.DataFrame(preds, columns=self.endog_names, index=test_df.index)

    @property
    def selected_lag(self):
        if self.fitted_model is None:
            return None
        if self.is_autoreg:
            return self.lags
        return self.fitted_model.k_ar

    @property
    def is_stable(self):
        """
        Kiểm tra tính ổn định của mô hình (tất cả nghiệm đa thức đặc trưng nằm ngoài đường tròn đơn vị).
        """
        if self.fitted_model is None:
            return None

        if self.is_autoreg:
            roots = getattr(self.fitted_model, "roots", None)
            if roots is None or len(roots) == 0:
                return True
            return bool(np.all(np.abs(roots) > 1.0))

        return bool(self.fitted_model.is_stable())

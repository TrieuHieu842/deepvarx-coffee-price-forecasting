import pandas as pd
from statsmodels.tsa.api import VAR


class VARModel:
    """
    Mô hình Vector Autoregression (VAR).

    Y_t = C + A1*Y_(t-1) + ... + Ap*Y_(t-p) + u_t
    """

    def __init__(self, lags=5, ic=None):
        self.lags = lags
        self.ic = ic
        self.model = None
        self.fitted_model = None

    def fit(self, train_df):
        """
        Huấn luyện VAR trên dữ liệu train.
        train_df chỉ chứa các biến nội sinh.
        """

        if train_df is None or train_df.empty:
            raise ValueError("Dữ liệu train không được rỗng.")

        if train_df.shape[1] < 2:
            raise ValueError(
                "VAR cần ít nhất 2 biến nội sinh."
            )

        self.model = VAR(train_df)

        if self.ic is None:
            self.fitted_model = self.model.fit(
                maxlags=self.lags
            )
        else:
            self.fitted_model = self.model.fit(
                maxlags=self.lags,
                ic=self.ic.lower()
            )

        return self

    def forecast(self, history_df, steps):
        """
        Dự báo `steps` bước tiếp theo.
        """

        if self.fitted_model is None:
            raise ValueError(
                "Model chưa được huấn luyện."
            )

        lag = self.fitted_model.k_ar

        history = history_df.values[-lag:]

        prediction = self.fitted_model.forecast(
            y=history,
            steps=steps
        )

        return pd.DataFrame(
            prediction,
            columns=history_df.columns
        )

    @property
    def selected_lag(self):
        if self.fitted_model is None:
            return None

        return self.fitted_model.k_ar
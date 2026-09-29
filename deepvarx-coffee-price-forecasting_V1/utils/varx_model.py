import numpy as np
import pandas as pd
import statsmodels.api as sm


class VARXModel:
    """
    Mô hình Vector Autoregression with Exogenous Variables (VARX).

    Y_t = C
          + A1*Y_(t-1) + ... + Ap*Y_(t-p)
          + B0*X_t + B1*X_(t-1) + ... + Bq*X_(t-q)
          + U_t

    Trong đó:
        Y: vector biến nội sinh
        X: vector biến ngoại sinh
        p: độ trễ tối đa của Y
        q: độ trễ tối đa của X

    Mỗi phương trình trong hệ VARX được ước lượng bằng OLS.
    """

    def __init__(self, p=5, q=2):
        self.p = int(p)
        self.q = int(q)

        self.models = {}
        self.fitted_models = {}

        self.endog_columns = None
        self.exog_columns = None
        self.feature_columns = None

    # =========================================================
    # TẠO MA TRẬN ĐẶC TRƯNG
    # =========================================================

    def _create_lagged_data(self, endog, exog):
        """
        Tạo dữ liệu:

        Y_t
        Y_(t-1), ..., Y_(t-p)
        X_t, X_(t-1), ..., X_(t-q)
        """

        endog = endog.copy()
        exog = exog.copy()

        self.endog_columns = list(endog.columns)
        self.exog_columns = list(exog.columns)

        data = pd.DataFrame(index=endog.index)

        # -----------------------------------------------------
        # Biến nội sinh hiện tại Y_t
        # -----------------------------------------------------

        for col in self.endog_columns:
            data[col] = endog[col]

        # -----------------------------------------------------
        # Lag của Y
        # Y(t-1) ... Y(t-p)
        # -----------------------------------------------------

        for lag in range(1, self.p + 1):

            for col in self.endog_columns:

                data[f"{col}_lag{lag}"] = (
                    endog[col].shift(lag)
                )

        # -----------------------------------------------------
        # X hiện tại và lag của X
        # X(t), X(t-1), ..., X(t-q)
        # -----------------------------------------------------

        for lag in range(0, self.q + 1):

            for col in self.exog_columns:

                if lag == 0:
                    name = f"{col}_lag0"
                else:
                    name = f"{col}_lag{lag}"

                data[name] = exog[col].shift(lag)

        # Loại bỏ các dòng không đủ lag
        data = data.dropna()

        return data

    # =========================================================
    # HUẤN LUYỆN VARX
    # =========================================================

    def fit(self, endog, exog):

        if endog is None or endog.empty:
            raise ValueError(
                "Dữ liệu biến nội sinh không được rỗng."
            )

        if exog is None or exog.empty:
            raise ValueError(
                "Dữ liệu biến ngoại sinh không được rỗng."
            )

        if endog.shape[1] < 1:
            raise ValueError(
                "Cần ít nhất 1 biến nội sinh."
            )

        if exog.shape[1] < 1:
            raise ValueError(
                "Cần ít nhất 1 biến ngoại sinh."
            )

        if len(endog) != len(exog):
            raise ValueError(
                "Số dòng của biến nội sinh và ngoại sinh "
                "phải bằng nhau."
            )

        if len(endog) <= max(self.p, self.q):
            raise ValueError(
                "Số lượng dữ liệu không đủ để tạo các độ trễ."
            )

        # -----------------------------------------------------
        # Tạo dữ liệu lag
        # -----------------------------------------------------

        data = self._create_lagged_data(
            endog,
            exog
        )

        # -----------------------------------------------------
        # X = các biến giải thích
        # -----------------------------------------------------

        feature_columns = []

        # Lag của Y
        for lag in range(1, self.p + 1):

            for col in self.endog_columns:
                feature_columns.append(
                    f"{col}_lag{lag}"
                )

        # X hiện tại + lag X
        for lag in range(0, self.q + 1):

            for col in self.exog_columns:

                if lag == 0:
                    feature_columns.append(
                        f"{col}_lag0"
                    )
                else:
                    feature_columns.append(
                        f"{col}_lag{lag}"
                    )

        self.feature_columns = feature_columns

        X = data[feature_columns]

        # Thêm hệ số hằng C
        X = sm.add_constant(X)

        # -----------------------------------------------------
        # Ước lượng từng phương trình Y
        # -----------------------------------------------------

        for target in self.endog_columns:

            y = data[target]

            model = sm.OLS(
                y,
                X
            )

            fitted = model.fit()

            self.models[target] = model
            self.fitted_models[target] = fitted

        return self

    # =========================================================
    # DỰ BÁO
    # =========================================================

    def forecast(self, history_endog, history_exog, exog_future):

        if not self.fitted_models:
            raise ValueError(
                "Model chưa được huấn luyện."
            )

        history_y = history_endog.copy()
        history_x = history_exog.copy()

        future_x = exog_future.copy()

        predictions = []

        for step in range(len(future_x)):

            row = {}

            # ================================================
            # Lag Y
            # ================================================

            for lag in range(1, self.p + 1):

                for col in self.endog_columns:

                    value = history_y.iloc[-lag][col]

                    row[f"{col}_lag{lag}"] = value

            # ================================================
            # X hiện tại + lag X
            # ================================================

            combined_x = pd.concat(
                [
                    history_x,
                    future_x.iloc[:step]
                ],
                ignore_index=True
            )

            for lag in range(0, self.q + 1):

                for col in self.exog_columns:

                    value = combined_x.iloc[-1 - lag][col]

                    if lag == 0:
                        name = f"{col}_lag0"
                    else:
                        name = f"{col}_lag{lag}"

                    row[name] = value

            # ================================================
            # DataFrame đầu vào
            # ================================================

            X_row = pd.DataFrame(
                [row],
                columns=self.feature_columns
            )

            X_row = sm.add_constant(
                X_row,
                has_constant="add"
            )

            # ================================================
            # Dự báo Y
            # ================================================

            prediction = {}

            for target in self.endog_columns:

                model = self.fitted_models[target]

                value = model.predict(X_row).iloc[0]

                prediction[target] = float(value)

            predictions.append(prediction)

            # ================================================
            # Cập nhật history
            # ================================================

            history_y = pd.concat(
                [
                    history_y,
                    pd.DataFrame([prediction])
                ],
                ignore_index=True
            )

        return pd.DataFrame(
            predictions,
            columns=self.endog_columns
        )
        
    # =========================================================
    # LAG
    # =========================================================

    @property
    def selected_lag(self):
        return self.p

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self):
        """
        Trả về summary của các phương trình VARX.
        """

        result = {}

        for target, model in self.fitted_models.items():
            result[target] = model.summary()

        return result
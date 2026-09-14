"""
Kiến trúc mạng LSTM cho bài toán dự báo đa biến, đa bước (multi-horizon).
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Input:  (batch, seq_len, n_features)
    Output: (batch, horizon, n_targets)

    Cấu trúc: LSTM nhiều lớp -> lấy hidden state cuối cùng -> fully-connected
    chiếu ra (horizon * n_targets), sau đó reshape lại thành (horizon, n_targets).
    """

    def __init__(self, n_features: int, n_targets: int, horizon: int,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_targets = n_targets
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, horizon * n_targets)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]              # (batch, hidden_size) — bước thời gian cuối
        last_hidden = self.dropout(last_hidden)
        pred = self.fc(last_hidden)               # (batch, horizon * n_targets)
        pred = pred.view(-1, self.horizon, self.n_targets)
        return pred
    def make_windows(data: np.ndarray, target_idx: list, seq_len: int, horizon: int):
        """
        data: mảng (T, n_features) đã chuẩn hóa.
        target_idx: chỉ số cột (trong feature_cols) cần dự báo.
        Trả về X: (N, seq_len, n_features), y: (N, horizon, n_targets)
        """
        X, y = [], []
        T = data.shape[0]
        for t in range(seq_len, T - horizon + 1):
            X.append(data[t - seq_len:t, :])
            y.append(data[t:t + horizon, target_idx])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

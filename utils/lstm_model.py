"""
Kiến trúc LSTM và các hàm hỗ trợ tạo sliding window cho bài toán dự báo
chuỗi thời gian đa biến — chuyển thể từ `lstm_training/model.py` và
`lstm_training/dataset.py` (project huấn luyện độc lập) để dùng trực tiếp
trong app Streamlit này.

Khác với project `lstm_training` gốc (tự đọc CSV + tự chuẩn hóa), ở đây dữ liệu
đã được nạp, tiền xử lý và chuẩn hóa (fit-on-train-only) sẵn ở các trang trước
của app (`train_df` / `val_df` / `test_df` trong `st.session_state`), nên module
này chỉ giữ lại phần kiến trúc mô hình (`LSTMForecaster`) và phần tạo sliding
window (`make_windows`, `TimeSeriesDataset`).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LSTMForecaster(nn.Module):
    """
    Input:  (batch, seq_len, n_features)
    Output: (batch, horizon, n_targets)
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
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        pred = self.fc(last_hidden)
        pred = pred.view(-1, self.horizon, self.n_targets)
        return pred


def make_windows(data: np.ndarray, target_idx: list, seq_len: int, horizon: int = 1):
    """
    data: mảng (T, n_features) — ĐÃ chuẩn hóa từ trước.
    target_idx: chỉ số cột (trong feature_cols) cần dự báo.
    Trả về X: (N, seq_len, n_features), y: (N, horizon, n_targets)
    """
    X, y = [], []
    T = data.shape[0]
    for t in range(seq_len, T - horizon + 1):
        X.append(data[t - seq_len:t, :])
        y.append(data[t:t + horizon, target_idx])
    if not X:
        return np.empty((0, seq_len, data.shape[1]), dtype=np.float32), \
               np.empty((0, horizon, len(target_idx)), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

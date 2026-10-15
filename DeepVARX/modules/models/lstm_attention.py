"""LSTM tiêu chuẩn, khối Temporal Attention và Embedding tỉnh (cho mô hình Global)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProvinceEmbedding(nn.Module):
    """Mã hoá tỉnh thành vector học được, nối vào đầu vào ở mọi bước thời gian."""

    def __init__(self, n_prov: int = 0, dim: int = 0):
        super().__init__()
        self.emb = nn.Embedding(n_prov, dim) if n_prov > 0 and dim > 0 else None
        self.dim = dim if self.emb is not None else 0

    def forward(self, x: torch.Tensor, pid):
        if self.emb is None or pid is None:
            return x
        e = self.emb(pid).unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.cat([x, e], dim=-1)


class TemporalAttention(nn.Module):
    """Attention cộng (additive) trên các bước thời gian của chuỗi trạng thái ẩn LSTM."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h: torch.Tensor):
        e = self.v(torch.tanh(self.w(h))).squeeze(-1)      # (B, L)
        beta = torch.softmax(e, dim=1)
        ctx = (h * beta.unsqueeze(-1)).sum(dim=1)          # (B, H)
        return ctx, beta


class LSTMForecaster(nn.Module):
    """LSTM chuẩn: chuỗi (B, L, V) [+ embedding tỉnh] → giá trị kế tiếp của Y. Không có Attention."""

    has_attention = False

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 n_prov: int = 0, emb_dim: int = 0):
        super().__init__()
        self.embed = ProvinceEmbedding(n_prov, emb_dim)
        self.lstm = nn.LSTM(n_in + self.embed.dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, pid=None, return_attention: bool = False):
        h, _ = self.lstm(self.embed(x, pid))
        return self.head(h[:, -1])

"""Tiện ích Attention không phụ thuộc PyTorch (dùng được ở Trang 6 khi chưa cài torch)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def attention_verdict(attn_epoch: pd.DataFrame, threshold: float = 0.1, tail_frac: float = 0.3) -> pd.DataFrame:
    """Ablation: biến bị đề xuất loại nếu gate tương đối (đã nhân V) < ngưỡng LIÊN TỤC trong phần cuối huấn luyện."""
    k = max(1, int(len(attn_epoch) * tail_frac))
    tail = attn_epoch.iloc[-k:]
    out = pd.DataFrame({
        "Biến": attn_epoch.columns,
        "Gate TB (epoch đầu)": attn_epoch.iloc[0].to_numpy(),
        "Gate TB (cuối)": tail.mean().to_numpy(),
        "Gate tối đa (cuối)": tail.max().to_numpy(),
    })
    out["Đề xuất"] = np.where(out["Gate tối đa (cuối)"] < threshold, "❌ Loại (gate → 0)", "✅ Giữ")
    return out.sort_values("Gate TB (cuối)", ascending=False).reset_index(drop=True)

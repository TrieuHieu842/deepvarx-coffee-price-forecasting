"""DeepVARX: LSTM + Variable Attention (gate theo biến, theo từng bước thời gian) + Temporal Attention.

- DeepVAR  = cùng kiến trúc nhưng chỉ nhận chuỗi nội sinh (n_exog = 0).
- DeepVARX = nhận thêm các biến ngoại sinh X (Tập mở rộng).
Trọng số Variable Attention được ghi lại qua từng epoch để làm Ablation Study."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from modules.data_utils import difference
from modules.attention_utils import attention_verdict  # noqa: F401  (giữ để tương thích)
from modules.models.lstm_attention import ProvinceEmbedding, TemporalAttention

try:
    from utils.deepvarx_model import CoffeeDeepVARX
except Exception:
    CoffeeDeepVARX = None



class DeepVARX(nn.Module):
    has_attention = True

    def __init__(self, n_in: int, hidden: int = 48, layers: int = 1, dropout: float = 0.0,
                 n_prov: int = 0, emb_dim: int = 0, residual: bool = True):
        super().__init__()
        self.n_in = n_in
        self.residual = residual
        # 1. Variable Gating với khởi tạo 0: lúc đầu alpha = 1/V đồng nhất, không làm méo tín hiệu gốc
        self.var_gate = nn.Linear(n_in, n_in)
        nn.init.zeros_(self.var_gate.weight)
        nn.init.zeros_(self.var_gate.bias)

        self.embed = ProvinceEmbedding(n_prov, emb_dim)
        self.lstm = nn.LSTM(n_in + self.embed.dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.temporal = TemporalAttention(hidden)

        # 2. Residual Head: Linear trực tiếp không qua dropout để đảm bảo gradient truyền thẳng về LSTM
        self.head = nn.Linear(2 * hidden, 1)
        if self.residual:
            # Khởi tạo trọng số tầng output bằng 0: tại Epoch 0 mô hình trùng khớp Naive Benchmark
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, pid=None, return_attention: bool = False):
        alpha = torch.softmax(self.var_gate(x), dim=-1)          # (B, L, V), tổng theo V = 1
        h, _ = self.lstm(self.embed(x * alpha * self.n_in, pid)) # nhân V để trung bình gate ≈ 1
        ctx, beta = self.temporal(h)
        delta = self.head(torch.cat([ctx, h[:, -1]], dim=-1))
        if self.residual:
            base_prev = x[:, -1, 0:1]
            out = base_prev + delta
        else:
            out = delta
        if return_attention:
            return out, alpha, beta
        return out


# ----------------------------------------------------------------------------- dữ liệu cửa sổ
def prepare_windows(df: pd.DataFrame, target: str, feat_cols: list[str], mode: str, d: int, train_n: int,
                    seq_len: int, val_frac: float = 0.15) -> dict:
    """Tạo cửa sổ trượt theo thời gian. Scale (z-score) chỉ khớp trên TRAIN.
    mode: 'level' (gốc, d=0) hoặc 'diff' (sai phân bậc d). Cột 0 luôn là biến mục tiêu."""
    cols = [target] + [c for c in feat_cols if c != target]
    data = df[cols].astype(float)
    offset = 0
    if mode == "diff" and d > 0:
        data, offset = difference(data, d), d
    train_rows = train_n - offset
    mu = data.iloc[:train_rows].mean()
    sd = data.iloc[:train_rows].std().replace(0, 1.0)
    arr = ((data - mu) / sd).to_numpy(np.float32)
    idx = np.arange(seq_len, len(arr))
    X = np.stack([arr[t - seq_len:t] for t in idx])
    y = arr[idx, 0:1]
    tr_idx = idx[idx < train_rows]
    n_val = max(1, int(len(tr_idx) * val_frac))
    n_tr = len(tr_idx) - n_val
    m_tr, m_te = idx < train_rows, idx >= train_rows
    Xtr_all, ytr_all = X[m_tr], y[m_tr]
    return {
        "cols": cols, "offset": offset, "seq_len": seq_len,
        "Xtr": Xtr_all[:n_tr], "ytr": ytr_all[:n_tr], "Xva": Xtr_all[n_tr:], "yva": ytr_all[n_tr:],
        "Xte": X[m_te], "yte": y[m_te], "t_test": idx[m_te] + offset,
        "mu_t": float(mu.iloc[0]), "sd_t": float(sd.iloc[0]),
    }


class DirectionAwareLoss(nn.Module):
    """Loss kết hợp Huber Loss (beta=1.0) hoặc MSE với Directional Penalty:
    - beta=1.0: Đảm bảo miền gradient bậc 2 (L2/MSE) đủ rộng cho dữ liệu chuẩn hoá, triệt tiêu hiện tượng Lazy Shrinkage (co cụm về 0)
      và giảm mạnh RMSE so với Naive Benchmark.
    - Directional Penalty: Phạt khi mô hình đoán ngược hướng thay đổi giá (-Δŷ·Δy > 0), đẩy tỷ lệ đúng hướng > 55%."""

    def __init__(self, loss_type: str = "huber", beta: float = 1.0, dir_weight: float = 0.05):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "mse":
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.SmoothL1Loss(beta=beta)
        self.dir_weight = float(dir_weight)
        self.relu = nn.ReLU()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        loss = self.base_loss(pred, target)
        if self.dir_weight > 0 and prev is not None:
            delta_pred = pred - prev
            delta_true = target - prev
            # Phạt khi đoán ngược chiều: delta_pred * delta_true < 0 -> -delta_pred * delta_true > 0
            dir_penalty = self.relu(-delta_pred * delta_true).mean()
            loss = loss + self.dir_weight * dir_penalty
        return loss


# ----------------------------------------------------------------------------- huấn luyện
def train_model(model: nn.Module, data: dict, epochs: int = 60, lr: float = 3e-4, batch_size: int = 64,
                patience: int = 10, seed: int = 42, track_attention: bool = True, progress_cb=None,
                restore_best: bool = True, dir_weight: float = 0.05, loss_type: str = "huber",
                huber_beta: float = 1.0, **kwargs) -> dict:
    dir_weight = float(kwargs.get("dir_weight", dir_weight))
    loss_type = str(kwargs.get("loss_type", loss_type))
    huber_beta = float(kwargs.get("huber_beta", huber_beta))
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = torch.tensor(data["Xtr"]), torch.tensor(data["ytr"])
    Xva, yva = torch.tensor(data["Xva"]), torch.tensor(data["yva"])
    Xte = torch.tensor(data["Xte"])
    P = {k: (torch.tensor(data[k], dtype=torch.long) if k in data else None) for k in ("Ptr", "Pva", "Pte")}

    def pick(t, idx):
        return None if t is None else t[idx]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = DirectionAwareLoss(loss_type=loss_type, beta=huber_beta, dir_weight=dir_weight)
    can_track = track_attention and getattr(model, "has_attention", False)
    hist, attn_rows = [], []
    best_val, best_state, best_epoch, bad = np.inf, None, 0, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            prev_b = Xtr[b][:, -1, 0:1] if getattr(model, "residual", False) else None
            loss = loss_fn(model(Xtr[b], pick(P['Ptr'], b)), ytr[b], prev_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        scheduler.step()
        tr_loss = tot / len(Xtr)
        model.eval()
        with torch.no_grad():
            if can_track:
                out, alpha, _ = model(Xva, P['Pva'], return_attention=True)
                if alpha is not None:
                    attn_rows.append((alpha.mean(dim=(0, 1)) * model.n_in).numpy())
            else:
                out = model(Xva, P['Pva'])
            prev_va = Xva[:, -1, 0:1] if getattr(model, "residual", False) else None
            va_loss = loss_fn(out, yva, prev_va).item()
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best_val - 1e-6:
            best_val, best_state, best_epoch, bad = va_loss, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if progress_cb:
            progress_cb(ep, epochs, tr_loss, va_loss)
        if bad >= patience:
            break

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)
    model.eval()
    result = {"history": pd.DataFrame(hist), "best_epoch": best_epoch}
    with torch.no_grad():
        if getattr(model, "has_attention", False):
            out, alpha, beta = model(Xte, P['Pte'], return_attention=True)
            if alpha is not None:
                result["attn_lag"] = (alpha.mean(dim=0) * model.n_in).numpy()     # (L, V)
                result["attn_test"] = (alpha.mean(dim=1) * model.n_in).numpy()     # (N_test, V): gate theo từng ngày TEST
            if beta is not None:
                result["attn_time"] = beta.mean(dim=0).numpy()                     # (L,)
        else:
            out = model(Xte, P['Pte'])
    result["pred_scaled"] = out.squeeze(-1).numpy()
    if can_track and attn_rows:
        result["attn_epoch"] = pd.DataFrame(attn_rows, columns=data["cols"], index=np.arange(1, len(attn_rows) + 1))
    return result


# ----------------------------------------------------------------------------- cửa sổ cho mô hình GLOBAL (toàn panel)
def prepare_panel_windows(C: pd.DataFrame, date_col: str, entity_col: str, target: str, feats: list[str],
                          seq_len: int, train_end, scalers: dict, val_frac: float = 0.15) -> dict:
    """Cửa sổ trượt tạo RIÊNG từng tỉnh (không băng qua ranh giới tỉnh) rồi nối lại. TRAIN/VAL/TEST chia theo
    NGÀY chung: test = ngày > train_end; val = phần cuối theo ngày của TRAIN. Dữ liệu C đã chuẩn hoá theo từng tỉnh."""
    from numpy.lib.stride_tricks import sliding_window_view

    C = C.copy()
    C[date_col] = pd.to_datetime(C[date_col])
    cols = [target] + list(feats)
    X, y, dates, pid, prov, lvl, prev = [], [], [], [], [], [], []
    for e, g in C.groupby(entity_col, observed=True, sort=False):
        g = g.sort_values(date_col)
        arr = g[cols].to_numpy(np.float32)
        T = len(arr)
        if T <= seq_len:
            continue
        w = sliding_window_view(arr, (seq_len, arr.shape[1]))[:, 0][: T - seq_len]
        X.append(np.ascontiguousarray(w))
        idx = np.arange(seq_len, T)
        y.append(arr[idx, 0:1])
        dates.append(g[date_col].to_numpy()[idx])
        pid.append(np.full(len(idx), int(g["province_id"].iloc[0])))
        prov.append(np.full(len(idx), str(e), dtype=object))
        lv = g["y_level"].to_numpy(float)
        lvl.append(lv[idx]); prev.append(lv[idx - 1])
    X, y, dates, pid, prov = (np.concatenate(a) for a in (X, y, dates, pid, prov))
    lvl, prev = np.concatenate(lvl), np.concatenate(prev)
    is_tr = dates <= np.datetime64(train_end)
    tr_dates = np.sort(np.unique(dates[is_tr]))
    cut = tr_dates[int(len(tr_dates) * (1 - val_frac))]
    m_va, m_te = is_tr & (dates > cut), ~is_tr
    m_tr = is_tr & ~m_va
    mu = np.array([scalers[p][0] for p in prov[m_te]]); sd = np.array([scalers[p][1] for p in prov[m_te]])
    return {"cols": cols, "Xtr": X[m_tr], "ytr": y[m_tr], "Ptr": pid[m_tr], "Xva": X[m_va], "yva": y[m_va],
            "Pva": pid[m_va], "Xte": X[m_te], "yte": y[m_te], "Pte": pid[m_te], "dates_te": dates[m_te],
            "prov_te": prov[m_te], "true_te": lvl[m_te], "prev_te": prev[m_te], "mu_te": mu, "sd_te": sd}

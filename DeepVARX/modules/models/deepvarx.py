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


class DeepVARX(nn.Module):
    has_attention = True

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 n_prov: int = 0, emb_dim: int = 0):
        super().__init__()
        self.n_in = n_in
        self.var_gate = nn.Linear(n_in, n_in)   # bias riêng từng biến + phụ thuộc giá trị đầu vào
        self.embed = ProvinceEmbedding(n_prov, emb_dim)
        self.lstm = nn.LSTM(n_in + self.embed.dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.temporal = TemporalAttention(hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x: torch.Tensor, pid=None, return_attention: bool = False):
        alpha = torch.softmax(self.var_gate(x), dim=-1)          # (B, L, V), tổng theo V = 1
        h, _ = self.lstm(self.embed(x * alpha * self.n_in, pid)) # nhân V để trung bình gate ≈ 1
        ctx, beta = self.temporal(h)
        out = self.head(torch.cat([ctx, h[:, -1]], dim=-1))
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


# ----------------------------------------------------------------------------- huấn luyện
def train_model(model: nn.Module, data: dict, epochs: int = 60, lr: float = 1e-3, batch_size: int = 64,
                patience: int = 10, seed: int = 42, track_attention: bool = True, progress_cb=None,
                restore_best: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = torch.tensor(data["Xtr"]), torch.tensor(data["ytr"])
    Xva, yva = torch.tensor(data["Xva"]), torch.tensor(data["yva"])
    Xte = torch.tensor(data["Xte"])
    P = {k: (torch.tensor(data[k], dtype=torch.long) if k in data else None) for k in ("Ptr", "Pva", "Pte")}

    def pick(t, idx):
        return None if t is None else t[idx]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
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
            loss = loss_fn(model(Xtr[b], pick(P['Ptr'], b)), ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        tr_loss = tot / len(Xtr)
        model.eval()
        with torch.no_grad():
            if can_track:
                out, alpha, _ = model(Xva, P['Pva'], return_attention=True)
                attn_rows.append((alpha.mean(dim=(0, 1)) * model.n_in).numpy())
            else:
                out = model(Xva, P['Pva'])
            va_loss = loss_fn(out, yva).item()
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
            result["attn_lag"] = (alpha.mean(dim=0) * model.n_in).numpy()     # (L, V)
            result["attn_time"] = beta.mean(dim=0).numpy()                     # (L,)
            result["attn_test"] = (alpha.mean(dim=1) * model.n_in).numpy()     # (N_test, V): gate theo từng ngày TEST
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

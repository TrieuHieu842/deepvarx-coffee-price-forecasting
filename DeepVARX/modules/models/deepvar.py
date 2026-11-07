"""
DeepVAR / DeepVARX  (v2) - module độc lập cho luận văn dự báo giá cà phê.

Kiến trúc (equation-by-equation, K MLP độc lập, KHÔNG chia sẻ trọng số):
    DeepVAR  : y_{k,t} = f_k( y_{t-1}, ..., y_{t-p} )                       + e_{k,t}
    DeepVARX : y_{k,t} = f_k( y_{t-1}, ..., y_{t-p}, x_{t-1}, ..., x_{t-s} ) + e_{k,t}

Đây là phiên bản MLP theo kiến trúc equation-by-equation lấy cảm hứng từ Deep VAR
(Altmeyer, Agustí, Vidal-Quadras Costa, 2021). Module KHÔNG tuyên bố tái lập toàn bộ
cấu hình thực nghiệm của bài báo gốc.

Quy ước dữ liệu (khớp pipeline chung):
    Y: np.ndarray (T, K)  biến nội sinh, thứ tự cột = thứ tự tỉnh, giữ nguyên.
    X: np.ndarray (T, M)  biến ngoại sinh, hoặc None (DeepVAR).
    n_train, n_val: số THỜI ĐIỂM của Train / Validation. Phần còn lại là Test.
    Module không chia lại dữ liệu, không nội suy / forward-fill missing.

Dự báo: rolling one-step-ahead (h=1) với dữ liệu THỰC đến t-1.
    h>1 được hỗ trợ ở dạng dự báo TRỰC TIẾP: Z dựng từ thông tin đến t-1,
    target là Y[t+h-1]; idx trả về là chỉ số thời điểm của target.

API tương thích ngược với v1: DeepVARConfig, DeepVAR(cfg, device), fit(Y, X, n_train, n_val),
predict(Y, X, start) -> (y_hat, idx), make_lagged, StandardScaler, rmse, mae, smape,
thuộc tính .history (list val-MSE), .best_val_mse.
"""
from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "DeepVARConfig", "DeepVAR", "StandardScaler", "make_lagged",
    "rmse", "mae", "mape", "smape", "evaluate_by_province", "format_table",
]

_MIN_STD = 1e-8


# ----------------------------------------------------------------------------
# 0. Kiểm tra đầu vào
# ----------------------------------------------------------------------------
def _check_int(name: str, v, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise ValueError(f"{name} phải là số nguyên, nhận {v!r} ({type(v).__name__}).")
    if v < lo:
        raise ValueError(f"{name} phải >= {lo}, nhận {v}.")
    return int(v)


def _as_2d_float(name: str, a, allow_none: bool = False) -> Optional[np.ndarray]:
    """Kiểm tra mảng (T, D), trả về bản sao float64. Không sửa dữ liệu gốc."""
    if a is None:
        if allow_none:
            return None
        raise ValueError(f"{name} không được là None.")
    if not isinstance(a, np.ndarray):
        raise ValueError(f"{name} phải là numpy.ndarray, nhận {type(a).__name__}.")
    if a.ndim != 2:
        raise ValueError(f"{name} phải có 2 chiều (T, D), nhận shape {a.shape}.")
    if a.size == 0:
        raise ValueError(f"{name} rỗng (shape {a.shape}).")
    try:
        out = a.astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} phải là dữ liệu số: {e}") from None
    bad_nan, bad_inf = np.isnan(out), np.isinf(out)
    if bad_nan.any():
        r, c = np.argwhere(bad_nan)[0]
        raise ValueError(
            f"{name} có {int(bad_nan.sum())} giá trị NaN (đầu tiên tại dòng {r}, cột {c}). "
            "Hãy xử lý missing trong pipeline tiền xử lý chung, không phải trong module mô hình.")
    if bad_inf.any():
        r, c = np.argwhere(bad_inf)[0]
        raise ValueError(
            f"{name} có {int(bad_inf.sum())} giá trị Inf (đầu tiên tại dòng {r}, cột {c}).")
    return out


# ----------------------------------------------------------------------------
# 1. Chuẩn hóa (chỉ fit trên Train; predict KHÔNG fit lại)
# ----------------------------------------------------------------------------
class StandardScaler:
    """z = (a - mu) / sd. Cột có sd ~ 0 (hằng số) dùng sd=1 để tránh chia cho 0."""

    def __init__(self, min_std: float = _MIN_STD):
        self.min_std = min_std
        self.mu = None
        self.sd = None

    def fit(self, a: np.ndarray) -> "StandardScaler":
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] < 1:
            raise ValueError(f"StandardScaler.fit cần mảng (n>=1, D), nhận {a.shape}.")
        self.mu = a.mean(0, keepdims=True)
        sd = a.std(0, keepdims=True)
        # ngưỡng tương đối theo độ lớn của mean để bắt cả trường hợp "gần hằng số"
        self.constant_cols = sd <= self.min_std * np.maximum(1.0, np.abs(self.mu))
        self.sd = np.where(self.constant_cols, 1.0, sd)
        return self

    def _check(self):
        if self.mu is None:
            raise RuntimeError("StandardScaler chưa được fit.")

    def transform(self, a: np.ndarray) -> np.ndarray:
        self._check()
        return (a - self.mu) / self.sd

    def inverse(self, a: np.ndarray) -> np.ndarray:
        self._check()
        return a * self.sd + self.mu


# ----------------------------------------------------------------------------
# 2. Cửa sổ trễ
# ----------------------------------------------------------------------------
def make_lagged(Y: np.ndarray, X: Optional[np.ndarray], p: int, s: int, h: int = 1
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (Z, Target, idx).

    Với mỗi t:  Z[n]      = [Y[t-1],...,Y[t-p], X[t-1],...,X[t-s]]  (flatten, thứ tự cột giữ nguyên)
                Target[n] = Y[t+h-1]          (h=1 -> Y[t])
                idx[n]    = t+h-1             (chỉ số thời điểm của TARGET)
    Z chỉ dùng thông tin đến t-1 -> không rò rỉ tương lai. X=None -> DeepVAR.
    """
    Y = _as_2d_float("Y", Y)
    X = _as_2d_float("X", X, allow_none=True)
    p = _check_int("p", p, 1)
    h = _check_int("h", h, 1)
    if X is not None:
        s = _check_int("s", s, 1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X và Y khác số dòng: X={X.shape[0]}, Y={Y.shape[0]}.")
    T = Y.shape[0]
    start = max(p, s) if X is not None else p
    ts = np.arange(start, T - h + 1)
    if len(ts) < 1:
        raise ValueError(
            f"Không đủ quan sát để tạo lag: T={T}, p={p}, s={s if X is not None else '-'}, h={h} "
            f"(cần T >= {start + h}).")
    parts = [Y[ts - i] for i in range(1, p + 1)]
    if X is not None:
        parts += [X[ts - j] for j in range(1, s + 1)]
    Z = np.concatenate(parts, axis=1).astype(np.float32)
    target = Y[ts + h - 1].astype(np.float32)
    return Z, target, ts + h - 1


# ----------------------------------------------------------------------------
# 3. K MLP độc lập, tính song song (không vòng for qua tỉnh trong forward)
# ----------------------------------------------------------------------------
class _ParallelMLP(nn.Module):
    """K MLP độc lập (mỗi phương trình một mạng), cùng nhận vector Z. Output (B, K). Có hỗ trợ Residual Learning."""

    def __init__(self, K: int, d_in: int, hidden: Sequence[int], dropout: float, residual: bool = True):
        super().__init__()
        self.K = K
        self.residual = residual
        dims = [d_in] + list(hidden) + [1]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            bound = 1.0 / math.sqrt(a)           # tương đương init mặc định của nn.Linear
            self.W.append(nn.Parameter((torch.rand(K, a, b) * 2 - 1) * bound))
            self.b.append(nn.Parameter((torch.rand(K, 1, b) * 2 - 1) * bound))
        if self.residual:
            self.W[-1].data.zero_()
            self.b[-1].data.zero_()
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:      # z: (B, d_in)
        h = z.unsqueeze(0).expand(self.K, -1, -1)            # (K, B, d_in)
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = torch.bmm(h, w) + b                          # (K, B, out)
            if i < last:
                h = self.drop(torch.relu(h))
        delta = h.squeeze(-1).t()
        return (z[:, :self.K] + delta) if self.residual else delta


# ----------------------------------------------------------------------------
# 4. Cấu hình, seed, thiết bị
# ----------------------------------------------------------------------------
@dataclass
class DeepVARConfig:
    p: int = 5
    s: int = 5                       # chỉ dùng khi có X (DeepVARX)
    h: int = 1
    hidden: Tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        self.hidden = tuple(self.hidden)
        _check_int("p", self.p, 1)
        _check_int("s", self.s, 1)
        _check_int("h", self.h, 1)
        if len(self.hidden) < 1:
            raise ValueError("hidden cần ít nhất 1 tầng ẩn, ví dụ (32,) hoặc (64, 32).")
        for u in self.hidden:
            _check_int("hidden[i]", u, 1)
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout phải thuộc [0, 1), nhận {self.dropout}.")
        if not self.lr > 0:
            raise ValueError(f"lr phải > 0, nhận {self.lr}.")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay phải >= 0, nhận {self.weight_decay}.")
        _check_int("batch_size", self.batch_size, 1)
        _check_int("max_epochs", self.max_epochs, 1)
        _check_int("patience", self.patience, 1)
        _check_int("seed", self.seed, 0)


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_device(device) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(device)
    except (RuntimeError, TypeError) as e:
        raise ValueError(f"device không hợp lệ: {device!r} ({e})") from None
    if dev.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("Yêu cầu CUDA nhưng không có GPU, chuyển sang CPU.", RuntimeWarning)
        return torch.device("cpu")
    return dev


# ----------------------------------------------------------------------------
# 5. DeepVAR / DeepVARX
# ----------------------------------------------------------------------------
class DeepVAR:
    """DeepVAR (X=None) hoặc DeepVARX (có X). Loại mô hình được quyết định ở lần fit."""

    def __init__(self, cfg: Optional[DeepVARConfig] = None, device="cpu"):
        self.cfg = cfg if cfg is not None else DeepVARConfig()
        self.device = _resolve_device(device)
        self.net = None

    # ---------------- huấn luyện ----------------
    def fit(self, Y: np.ndarray, X: Optional[np.ndarray], n_train: int, n_val: int,
            verbose: int = 1, progress_cb=None) -> "DeepVAR":
        """verbose: 0 im lặng; 1 in số tham số/mẫu và kết quả; 2 thêm log mỗi 10 epoch.
        progress_cb(ep, max_epochs, train_mse, val_mse): callback cập nhật tiến độ (dùng cho Streamlit)."""
        c = self.cfg
        Y = _as_2d_float("Y", Y)
        X = _as_2d_float("X", X, allow_none=True)
        T, K = Y.shape
        if X is not None and X.shape[0] != T:
            raise ValueError(f"X và Y khác số dòng: X={X.shape[0]}, Y={T}.")
        n_train = _check_int("n_train", n_train, 1)
        n_val = _check_int("n_val", n_val, 1)
        if n_train + n_val > T:
            raise ValueError(f"n_train + n_val = {n_train + n_val} vượt quá T = {T}.")

        _set_seed(c.seed, c.deterministic)
        self.use_x = X is not None
        self.K, self.M = K, (X.shape[1] if X is not None else 0)
        self.n_train, self.n_val = n_train, n_val

        # scaler chỉ fit trên Train
        self.sy = StandardScaler().fit(Y[:n_train])
        Ys = self.sy.transform(Y)
        if self.use_x:
            self.sx = StandardScaler().fit(X[:n_train])
            Xs = self.sx.transform(X)
        else:
            self.sx, Xs = None, None

        Z, Tgt, idx = make_lagged(Ys, Xs, c.p, c.s, c.h)
        tr = idx < n_train
        va = (idx >= n_train) & (idx < n_train + n_val)
        n_tr, n_va = int(tr.sum()), int(va.sum())
        if n_tr < 2:
            raise ValueError(f"Chỉ có {n_tr} mẫu Train sau khi tạo lag (cần >= 2). "
                             f"Giảm p/s hoặc tăng n_train.")
        if n_va < 1:
            raise ValueError("Validation rỗng sau khi tạo lag. Tăng n_val hoặc giảm p/s/h.")

        d_in = Z.shape[1]
        self.net = _ParallelMLP(K, d_in, c.hidden, c.dropout).to(self.device)
        self.n_params = int(sum(p.numel() for p in self.net.parameters()))
        self.params_per_equation = self.n_params // K
        self.n_train_samples, self.n_val_samples = n_tr, n_va
        if self.params_per_equation > n_tr:
            warnings.warn(
                f"Mỗi phương trình có {self.params_per_equation} tham số > {n_tr} mẫu Train: "
                "nguy cơ overfit cao. Cân nhắc hidden nhỏ hơn / dropout / weight_decay lớn hơn.",
                RuntimeWarning)
        if verbose >= 1:
            try:
                print(f"[{self.model_name}] K={K} d_in={d_in} hidden={c.hidden} | "
                      f"tham số={self.n_params} ({self.params_per_equation}/phương trình) | "
                      f"mẫu Train={n_tr}, Val={n_va} | device={self.device}")
            except UnicodeEncodeError:
                print(f"[{self.model_name}] K={K} d_in={d_in} hidden={c.hidden} | "
                      f"params={self.n_params} ({self.params_per_equation}/eq) | "
                      f"train={n_tr}, val={n_va} | device={self.device}")

        to = lambda a: torch.from_numpy(a).to(self.device)
        Ztr, Ttr, Zva, Tva = to(Z[tr]), to(Tgt[tr]), to(Z[va]), to(Tgt[va])
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        lossf = nn.SmoothL1Loss(beta=0.01)  # Huber Loss giúp chống sốc giá ngoại lai
        gen = torch.Generator()
        gen.manual_seed(c.seed)

        best, best_state, best_epoch, bad = math.inf, None, 0, 0
        self.history_detail: Dict[str, List[float]] = {
            "train_batch_loss": [],   # loss khi huấn luyện (dropout bật)
            "train_mse": [],          # MSE trên Train ở chế độ eval (so sánh trực tiếp với val)
            "val_mse": [],
        }
        for ep in range(1, c.max_epochs + 1):
            self.net.train()
            perm = torch.randperm(n_tr, generator=gen).to(self.device)
            running = 0.0
            for i in range(0, n_tr, c.batch_size):
                b = perm[i:i + c.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = lossf(self.net(Ztr[b]), Ttr[b])
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Loss không hữu hạn ({loss.item()}) tại epoch {ep}. "
                        "Thử giảm lr, kiểm tra dữ liệu hoặc scale đầu vào.")
                loss.backward()
                if c.grad_clip:
                    nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
                opt.step()
                running += loss.item() * len(b)
            self.net.eval()
            with torch.no_grad():
                tr_mse = lossf(self.net(Ztr), Ttr).item()
                v = lossf(self.net(Zva), Tva).item()
            if not (math.isfinite(tr_mse) and math.isfinite(v)):
                raise RuntimeError(f"MSE không hữu hạn tại epoch {ep} (train={tr_mse}, val={v}).")
            self.history_detail["train_batch_loss"].append(running / n_tr)
            self.history_detail["train_mse"].append(tr_mse)
            self.history_detail["val_mse"].append(v)
            if progress_cb is not None:
                progress_cb(ep, c.max_epochs, tr_mse, v)
            if verbose >= 2 and ep % 10 == 0:
                print(f"  epoch {ep:4d} | train_mse={tr_mse:.5f} | val_mse={v:.5f}")
            if v < best - c.min_delta:
                best, best_epoch, bad = v, ep, 0
                best_state = {k: x.detach().clone() for k, x in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break
        if best_state is None:
            raise RuntimeError("Không có trạng thái tốt nhất để nạp (val-MSE không cải thiện hợp lệ).")
        self.net.load_state_dict(best_state)
        self.net.eval()
        self.best_val_mse, self.best_epoch = float(best), int(best_epoch)
        self.epochs_run = len(self.history_detail["val_mse"])
        self.history = list(self.history_detail["val_mse"])        # tương thích v1
        if verbose >= 1:
            h_ = self.history_detail
            try:
                print(f"[{self.model_name}] best_epoch={self.best_epoch}/{self.epochs_run} | "
                      f"best_val_mse={self.best_val_mse:.5f} | "
                      f"train_mse@best={h_['train_mse'][self.best_epoch - 1]:.5f}")
            except UnicodeEncodeError:
                print(f"[{self.model_name}] best_epoch={self.best_epoch}/{self.epochs_run} | "
                      f"best_val_mse={self.best_val_mse:.5f}")
        return self

    # ---------------- dự báo rolling one-step-ahead ----------------
    @torch.no_grad()
    def predict(self, Y: np.ndarray, X: Optional[np.ndarray], start: int
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (y_hat (N,K) thang gốc của Y, idx (N,)). `start` là chỉ số thời điểm
        của target đầu tiên cần dự báo (vd n_train + n_val cho Test). Dùng scaler đã fit,
        dữ liệu thực đến t-1; không fit lại gì cả."""
        if self.net is None:
            raise RuntimeError("Mô hình chưa được fit.")
        c = self.cfg
        Y = _as_2d_float("Y", Y)
        if Y.shape[1] != self.K:
            raise ValueError(f"Y có {Y.shape[1]} cột, lúc fit là {self.K}.")
        if self.use_x:
            if X is None:
                raise ValueError("Mô hình là DeepVARX (fit có X): predict bắt buộc truyền X.")
            X = _as_2d_float("X", X)
            if X.shape[1] != self.M:
                raise ValueError(f"X có {X.shape[1]} cột, lúc fit là {self.M}.")
            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"X và Y khác số dòng: X={X.shape[0]}, Y={Y.shape[0]}.")
            Xs = self.sx.transform(X)
        else:
            if X is not None:
                raise ValueError("Mô hình là DeepVAR (fit không có X): predict không được truyền X.")
            Xs = None
        Ys = self.sy.transform(Y)
        Z, _, idx = make_lagged(Ys, Xs, c.p, c.s, c.h)
        start = _check_int("start", start, 0)
        if not (idx[0] <= start <= idx[-1]):
            raise ValueError(f"start={start} không hợp lệ; phạm vi dự báo được là "
                             f"[{int(idx[0])}, {int(idx[-1])}].")
        m = idx >= start
        self.net.eval()
        out = self.net(torch.from_numpy(Z[m]).to(self.device)).cpu().numpy().astype(np.float64)
        yhat = self.sy.inverse(out)
        assert yhat.shape == (int(m.sum()), self.K)
        return yhat, idx[m]

    # ---------------- tiện ích ----------------
    @property
    def model_name(self) -> str:
        return "DeepVARX" if getattr(self, "use_x", False) else "DeepVAR"

    def summary(self) -> Dict[str, object]:
        if self.net is None:
            raise RuntimeError("Mô hình chưa được fit.")
        return {"model": self.model_name, "K": self.K, "M": self.M, "hidden": self.cfg.hidden,
                "p": self.cfg.p, "s": self.cfg.s, "h": self.cfg.h, "dropout": self.cfg.dropout,
                "weight_decay": self.cfg.weight_decay, "seed": self.cfg.seed,
                "n_params": self.n_params, "params_per_equation": self.params_per_equation,
                "n_train_samples": self.n_train_samples, "n_val_samples": self.n_val_samples,
                "best_epoch": self.best_epoch, "epochs_run": self.epochs_run,
                "best_val_mse": self.best_val_mse}


# ----------------------------------------------------------------------------
# 6. Chỉ số đánh giá
# ----------------------------------------------------------------------------
def _paired(y, yh) -> Tuple[np.ndarray, np.ndarray]:
    y, yh = np.asarray(y, dtype=np.float64), np.asarray(yh, dtype=np.float64)
    if y.shape != yh.shape:
        raise ValueError(f"y và y_hat khác shape: {y.shape} vs {yh.shape}.")
    if y.size == 0:
        raise ValueError("Không có quan sát để đánh giá.")
    if not (np.isfinite(y).all() and np.isfinite(yh).all()):
        raise ValueError("y hoặc y_hat chứa NaN/Inf.")
    return y, yh


def rmse(y, yh) -> float:
    y, yh = _paired(y, yh)
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mae(y, yh) -> float:
    y, yh = _paired(y, yh)
    return float(np.mean(np.abs(y - yh)))


def mape(y, yh, eps: float = 1e-8) -> float:
    """MAPE (%) loại các điểm có |y| <= eps; NaN nếu không còn điểm nào.
    Nên tính trên GIÁ (VND/kg); trên log-return MAPE không có ý nghĩa vì y ~ 0."""
    y, yh = _paired(y, yh)
    m = np.abs(y) > eps
    if not m.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(y[m] - yh[m]) / np.abs(y[m])))


def smape(y, yh, eps: float = 1e-8) -> float:
    """sMAPE (%) = mean(2|y-yh| / (|y|+|yh|)); điểm có mẫu số <= eps tính là 0 sai số."""
    y, yh = _paired(y, yh)
    den = np.abs(y) + np.abs(yh)
    term = np.where(den > eps, 2.0 * np.abs(y - yh) / np.where(den > eps, den, 1.0), 0.0)
    return float(100.0 * np.mean(term))


_METRICS = (("RMSE", rmse), ("MAE", mae), ("MAPE", mape), ("sMAPE", smape))


def evaluate_by_province(Y_true: np.ndarray, Y_pred: np.ndarray,
                         provinces: Optional[Sequence[str]] = None, eps: float = 1e-8) -> Dict:
    """Chỉ số theo từng tỉnh (giữ thứ tự cột) + trung bình CÁC CHỈ SỐ THEO TỈNH.
    Y_true, Y_pred: (N, K), cùng thang đo và cùng tập thời gian."""
    Y_true, Y_pred = _paired(Y_true, Y_pred)
    if Y_true.ndim != 2:
        raise ValueError("Y_true/Y_pred phải có shape (N, K).")
    K = Y_true.shape[1]
    names = list(provinces) if provinces is not None else [f"P{k}" for k in range(K)]
    if len(names) != K:
        raise ValueError(f"provinces có {len(names)} phần tử nhưng K={K}.")
    rows = []
    for k, nm in enumerate(names):
        r = {"Province": nm}
        for label, fn in _METRICS:
            r[label] = fn(Y_true[:, k], Y_pred[:, k], eps) if label in ("MAPE", "sMAPE") \
                else fn(Y_true[:, k], Y_pred[:, k])
        rows.append(r)
    mean = {"Province": "Mean"}
    for label, _ in _METRICS:
        vals = np.array([r[label] for r in rows], dtype=np.float64)
        mean[label] = float(np.mean(vals[np.isfinite(vals)])) if np.isfinite(vals).any() else float("nan")
    return {"per_province": rows, "mean": mean}


def format_table(result: Dict, digits: int = 4) -> str:
    head = "| Province | RMSE | MAE | MAPE | sMAPE |\n|---|---:|---:|---:|---:|"
    lines = [head]
    for r in result["per_province"] + [result["mean"]]:
        lines.append(f"| {r['Province']} | " + " | ".join(
            f"{r[m]:.{digits}f}" for m in ("RMSE", "MAE", "MAPE", "sMAPE")) + " |")
    return "\n".join(lines)

"""
Trang 14 — Huấn luyện mô hình (bản viết lại theo đặc tả Trang 14–15)

PIPELINE: Naive → VAR → VARX → VARX-L → LSTM → DeepVAR → DeepVARX

Nguyên tắc thiết kế (và lý do — đây là các chỗ bản cũ sai):
  1. Trang này CHỈ huấn luyện bằng TRAIN và chọn/dừng bằng VAL. KHÔNG tính số đo TEST ở đây.
     TEST chỉ được sinh ở bước "Khóa và đánh giá TEST" (một lần; có bộ đếm số lần xem TEST
     để dòng 8 của Leakage Audit đọc).
  2. Mọi số đo tính trên ĐƠN VỊ GỐC (VND/kg): dự báo được đảo log / sai phân / chuẩn hóa trước khi
     tính MAE, RMSE, sMAPE. (Bản cũ tính trên z-score nhưng lưu dự báo đã đảo chuẩn hóa.)
  3. X chưa biết trước không bao giờ vào đầu vào. VARX dùng chiến lược DIRECT: mỗi horizon h có một
     hồi quy riêng chỉ dùng thông tin tại thời điểm gốc t. Chỉ các biến khai báo
     known_future_vars (Tab D) mới được dùng giá trị tại t+h.
  4. Không có "số giả". Mô hình chưa nối hook hiển thị "Chưa nối", không sinh loss mô phỏng.
  5. Không `except: break` nuốt lỗi; lỗi được ghi vào trạng thái của từng lần chạy.

HỢP ĐỒNG VỚI TRANG 13 (st.session_state) — [XÁC NHẬN] khớp với code Trang 13 của bạn:
  experiment_config = {
      "p": int, "q": int, "H": int,                      # lookback Y, lookback X, horizon
      "transform": {"log": bool, "diff": 0 | 1},         # chỉ áp cho biến nội sinh
      "scaler": "StandardScaler" | "MinMaxScaler" | "None",
      "split": {"train_ratio": .., "val_ratio": .., "test_ratio": ..},
      "folds": [{"n_train": .., "n_val": .., "n_test": ..}, ...],   # tùy chọn (walk-forward)
      "known_future_vars": [...],                        # tùy chọn (từ Tab D)
  }
  leakage_audit = {"passed": bool, ...}                  # Trang 13 ghi sau khi kiểm toán
Khi thiếu experiment_config, trang đọc lại các khóa cũ (split_config, scaler_type, lookback_p,
horizon_q).

HOOK cho mô hình của bạn: tạo models_hooks.py với
    HOOKS_USER = {"VARX-L": fit_varxl, "DeepVAR": fit_deepvar, "DeepVARX": fit_deepvarx}
trong đó mỗi fit_xxx(ctx) -> dict gồm:
    forecaster : callable(prep, i) -> ndarray (H, K) ĐƠN VỊ GỐC, chỉ dùng thông tin tới chỉ số i
    n_params   : int
    loss_type  : str ("NLL", "MSE chuẩn hóa", ...)
    val_loss   : float (tùy chọn)
    best_epoch : int  (tùy chọn)
    curves     : {"train": [...], "val": [...]} (tùy chọn; cho Trang 15)
    ckpt       : đường dẫn tệp đã lưu, hoặc dict các mảng numpy để trang tự lưu .npz
ctx gồm: prep, p, q, H, seed, hyper, known_future (mảng bool theo biến X), log (callable).
"""
import hashlib
import json
import os
import platform
import random
import sys
import time
from importlib import metadata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
META_COLS = {"date", "province", "lat", "lon", "coffee_price_vnd_kg_raw"}
PIPELINE = ["Naive", "VAR", "VARX", "VARX-L", "LSTM", "DeepVAR", "DeepVARX"]
NEURAL = ["LSTM", "DeepVAR", "DeepVARX"]
CKPT_DIR = "checkpoints"


# ═══════════════════════════════════════════════════════════════════════════
# LÕI TÍNH TOÁN (numpy thuần — kiểm thử được không cần Streamlit/PyTorch)
# ═══════════════════════════════════════════════════════════════════════════
class Scaler:
    """Chuẩn hóa tự cài để có thể kiểm tra: tham số CHỈ fit trên TRAIN."""

    def __init__(self, kind="StandardScaler"):
        self.kind, self.a, self.b = kind, None, None

    def fit(self, X):
        X = np.asarray(X, float)
        if self.kind == "StandardScaler":
            self.a = X.mean(0)
            s = X.std(0)
            s[s == 0] = 1.0
            self.b = s
        elif self.kind == "MinMaxScaler":
            lo, hi = X.min(0), X.max(0)
            r = hi - lo
            r[r == 0] = 1.0
            self.a, self.b = lo, r
        else:
            self.a, self.b = np.zeros(X.shape[1]), np.ones(X.shape[1])
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.a) / self.b

    def inverse(self, Z):
        return np.asarray(Z, float) * self.b + self.a


class Prep:
    """raw (N×D) → [log nội sinh] → [sai phân nội sinh] → chuẩn hóa (fit trên TRAIN).

    Mọi dự báo của mô hình nằm ở miền đã chuẩn hóa (Zy); `invert` đưa về đơn vị gốc.
    Chỉ số i luôn là chỉ số hàng của mảng đã biến đổi; `raw_endo[i]` là giá gốc cùng hàng."""

    def __init__(self, raw, endo_idx, exog_idx, n_train, n_val, n_end=None,
                 scaler_kind="StandardScaler", use_log=False, diff=0):
        R = np.asarray(raw, float)
        self.endo_idx, self.exog_idx = list(endo_idx), list(exog_idx)
        self.use_log, self.diff = bool(use_log), int(diff)
        T = R.copy()
        self.raw_endo = R[:, self.endo_idx].copy()
        if self.use_log:
            if (T[:, self.endo_idx] <= 0).any():
                raise ValueError("Có giá trị ≤ 0 ở biến nội sinh: không thể lấy log.")
            T[:, self.endo_idx] = np.log(T[:, self.endo_idx])
        self.level = T[:, self.endo_idx].copy()          # mức (sau log) — dùng để cộng dồn sai phân
        if self.diff:
            T[1:, self.endo_idx] = T[1:, self.endo_idx] - T[:-1, self.endo_idx]
            T, self.level, self.raw_endo = T[1:], self.level[1:], self.raw_endo[1:]
        if n_end is not None:
            T, self.level, self.raw_endo = T[:n_end], self.level[:n_end], self.raw_endo[:n_end]
        self.N, self.n_tr, self.n_vl = len(T), int(n_train), int(n_val)
        if self.n_tr + self.n_vl >= self.N:
            raise ValueError("TRAIN + VAL phải nhỏ hơn tổng số quan sát (cần còn đoạn TEST).")
        self.scaler = Scaler(scaler_kind).fit(T[: self.n_tr])      # CHỈ TRAIN
        Z = self.scaler.transform(T)
        self.Zy = Z[:, self.endo_idx]
        self.Zx = Z[:, self.exog_idx] if self.exog_idx else np.zeros((self.N, 0))
        self.K, self.M = self.Zy.shape[1], self.Zx.shape[1]

    def origins(self, seg, H, lookback=1):
        """Điểm gốc i sao cho MỌI mục tiêu i+1..i+H nằm trọn trong đoạn seg."""
        ntr, nvl, N = self.n_tr, self.n_vl, self.N
        if seg == "train":
            lo, hi = lookback - 1, ntr - 1
        elif seg == "val":
            lo, hi = ntr - 1, ntr + nvl - 1
        else:
            lo, hi = ntr + nvl - 1, N - 1
        return np.arange(lo, hi - H + 1) if hi - H >= lo else np.array([], dtype=int)

    def invert(self, i, Fz):
        """Dự báo chuẩn hóa (H×K) tại gốc i → đơn vị gốc (H×K)."""
        V = np.asarray(Fz, float) * self.scaler.b[self.endo_idx] + self.scaler.a[self.endo_idx]
        if self.diff:
            V = self.level[i] + np.cumsum(V, axis=0)
        if self.use_log:
            V = np.exp(V)
        return V

    def naive(self, i, H):
        return np.repeat(self.raw_endo[i][None, :], H, axis=0)


def fit_var_core(Zy, n_train, p):
    """VAR(p) bằng OLS trên TRAIN. B có hình (1+K·p)×K, cột: [hằng số, lag1 (K), lag2 (K), ...]."""
    K = Zy.shape[1]
    Y = Zy[:n_train]
    rows = n_train - p
    c = 1 + K * p
    if rows <= c:
        raise ValueError(f"VAR: số hàng hiệu dụng {rows} ≤ số hệ số mỗi phương trình {c}.")
    X = np.hstack([np.ones((rows, 1))] + [Y[p - j:n_train - j] for j in range(1, p + 1)])
    Tt = Y[p:n_train]
    B = np.linalg.lstsq(X, Tt, rcond=None)[0]
    resid = Tt - X @ B
    sig = resid.T @ resid / rows
    ld = float(np.linalg.slogdet(sig)[1])
    m = K * (K * p + 1)                        # số tham số tự do (gồm hằng số), như statsmodels
    return B, dict(aic=ld + 2 * m / rows, bic=ld + np.log(rows) * m / rows, n_params=m, rows=rows)


def var_forecast(B, Zy, i, p, H):
    lags = [Zy[i - j] for j in range(p)]       # lags[0] = Y_i (gần nhất)
    out = []
    for _ in range(H):
        y = np.concatenate([[1.0]] + lags) @ B
        out.append(y)
        lags = [y] + lags[:-1]
    return np.array(out)


def _varx_row(Zy, Zx, t, h, p, q, kf):
    """Đặc trưng tại gốc t cho horizon h. Biến past-only: Zx[t-j]; known-future: Zx[t+h-j]."""
    parts = [1.0]
    for j in range(p):
        parts.extend(Zy[t - j])
    for j in range(q):
        parts.extend(np.where(kf, Zx[t + h - j], Zx[t - j]))
    return np.asarray(parts, float)


def fit_varx_direct(prep, p, q, H, kf):
    """VARX chiến lược DIRECT: H hồi quy riêng, mỗi hồi quy chỉ dùng thông tin tại gốc t."""
    if prep.M == 0:
        raise ValueError("VARX cần ít nhất 1 biến ngoại sinh.")
    K, M = prep.K, prep.M
    kf = np.asarray(kf, bool)
    c = 1 + K * p + M * q
    start = max(p, q) - 1
    Bs, ranks = [], []
    for h in range(1, H + 1):
        ts = np.arange(start, prep.n_tr - h)               # mục tiêu t+h ≤ n_tr-1
        if len(ts) <= c:
            raise ValueError(f"VARX: N_eff={len(ts)} ≤ c={c} (h={h}) — không ước lượng được; "
                             "giảm p,q hoặc dùng VARX-L.")
        X = np.vstack([_varx_row(prep.Zy, prep.Zx, t, h, p, q, kf) for t in ts])
        Yt = prep.Zy[ts + h]
        B, _, rk, _ = np.linalg.lstsq(X, Yt, rcond=None)
        Bs.append(B)
        ranks.append(int(rk))
    n_eff_H = prep.n_tr - H - start
    return Bs, dict(c=c, n_params=H * K * c, ratio=c / n_eff_H, n_eff=n_eff_H,
                    rank_deficient=any(r < c for r in ranks))


def varx_forecast(Bs, prep, i, p, q, H, kf):
    kf = np.asarray(kf, bool)
    return np.vstack([_varx_row(prep.Zy, prep.Zx, i, h, p, q, kf) @ Bs[h - 1]
                      for h in range(1, H + 1)])


def predict(forecaster, prep, seg, H):
    """Dự báo mọi điểm gốc của đoạn seg → (P, A) đơn vị GỐC, hình (n, H, K)."""
    idx = prep.origins(seg, H)
    if len(idx) == 0:
        raise ValueError(f"Đoạn {seg} quá ngắn cho horizon H={H}.")
    P = np.stack([forecaster(prep, int(i)) for i in idx])
    A = np.stack([prep.raw_endo[i + 1:i + 1 + H] for i in idx])
    if not np.isfinite(P).all():
        raise ValueError("Dự báo chứa NaN/Inf.")
    return P, A


def metrics(P, A):
    e = P - A
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                sMAPE=float(np.mean(2 * np.abs(e) / (np.abs(P) + np.abs(A) + 1e-12)) * 100))


def train_with_early_stopping(train_epoch, val_epoch, snapshot, restore, max_epochs, patience, log=None):
    """Vòng lặp epoch dùng chung. Dừng sớm theo VAL loss; khôi phục trạng thái tốt nhất."""
    best, best_ep, wait, state = np.inf, 0, 0, None
    tr, vl = [], []
    for ep in range(1, max_epochs + 1):
        t_loss, v_loss = float(train_epoch()), float(val_epoch())
        if not (np.isfinite(t_loss) and np.isfinite(v_loss)):
            raise FloatingPointError(f"Loss không hữu hạn ở epoch {ep} (train={t_loss}, val={v_loss}).")
        tr.append(t_loss)
        vl.append(v_loss)
        if v_loss < best - 1e-12:
            best, best_ep, wait, state = v_loss, ep, 0, snapshot()
        else:
            wait += 1
        if log:
            log(f"Epoch {ep:3d}/{max_epochs} | Train {t_loss:.6f} | Val {v_loss:.6f}"
                + ("  ← best" if best_ep == ep else ""))
        if wait >= patience:
            if log:
                log(f"Dừng sớm tại epoch {ep}: Val không cải thiện {patience} epoch liên tiếp.")
            break
    if state is not None:
        restore(state)
    return dict(train=tr, val=vl, best_epoch=best_ep, best_val=float(best), stopped_epoch=len(tr))


def overfit_start(tr, vl, patience):
    """Cờ quá khớp (Trang 15): Val tăng liên tiếp ≥ patience epoch trong khi Train giảm.
    Trả về epoch (đếm từ 1) mà chuỗi tăng bắt đầu, hoặc None."""
    run = 0
    for k in range(1, len(vl)):
        run = run + 1 if (vl[k] > vl[k - 1] and tr[k] < tr[k - 1]) else 0
        if run >= patience:
            return k - patience + 2
    return None


def agg_curves(curves):
    """Căn các đường học có độ dài khác nhau (do dừng sớm) → mean, std, số seed còn lại theo epoch."""
    L = max(len(c) for c in curves)
    M = np.full((len(curves), L), np.nan)
    for r, c in enumerate(curves):
        M[r, :len(c)] = c
    n = np.sum(~np.isnan(M), axis=0)
    mean = np.nanmean(M, axis=0)
    std = np.where(n > 1, np.nanstd(M, axis=0, ddof=1), 0.0)
    return mean, std, n


def seed_sd_rel_error(S):
    """Sai số tương đối xấp xỉ của độ lệch chuẩn ước lượng từ S seed: 1/√(2(S−1))."""
    return 1.0 / np.sqrt(2 * (S - 1)) if S >= 2 else float("nan")


def n_runs_estimate(n_classical, n_neural, F, S):
    return n_classical * F + n_neural * F * S


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def data_hash(arr, columns):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(arr, float)).tobytes())
    h.update("|".join(map(str, columns)).encode())
    return h.hexdigest()


def build_manifest(cfg, d_hash, models, seeds, hyper, folds):
    libs = {}
    for lib in ("numpy", "pandas", "scipy", "torch", "streamlit", "statsmodels"):
        try:
            libs[lib] = metadata.version(lib)
        except Exception:
            libs[lib] = None
    hw = {"platform": platform.platform(), "processor": platform.processor(),
          "python": sys.version.split()[0]}
    try:
        import torch
        hw["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        hw["cuda"] = None
    try:
        code_hash = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    except Exception:
        code_hash = None
    return dict(data_sha256=d_hash, code_sha256=code_hash, libraries=libs, hardware=hw,
                config=cfg, models=models, seeds=seeds, hyperparameters=hyper, folds=folds)


# ── Mô hình: mỗi hàm fit nhận ctx và trả dict (xem docstring đầu tệp) ──────
def fit_naive(ctx):
    H = ctx["H"]
    return dict(forecaster=lambda prep, i: prep.naive(i, H), n_params=0, loss_type="RMSE val (gốc)")


def fit_var(ctx):
    p, H, prep = ctx["p"], ctx["H"], ctx["prep"]
    B, info = fit_var_core(prep.Zy, prep.n_tr, p)
    return dict(forecaster=lambda pr, i: pr.invert(i, var_forecast(B, pr.Zy, i, p, H)),
                n_params=info["n_params"], loss_type="RMSE val (gốc)", aic=info["aic"], bic=info["bic"],
                ckpt={"B": B}, note=f"OLS, dự báo đệ quy; N_eff={info['rows']}")


def fit_varx(ctx):
    p, q, H, prep, kf = ctx["p"], ctx["q"], ctx["H"], ctx["prep"], ctx["known_future"]
    Bs, info = fit_varx_direct(prep, p, q, H, kf)
    note = (f"DIRECT, c={info['c']}, N_eff={info['n_eff']}, c/N_eff={info['ratio']:.3f}"
            + (" — HẠNG THIẾU (đa cộng tuyến)" if info["rank_deficient"] else ""))
    return dict(forecaster=lambda pr, i: pr.invert(i, varx_forecast(Bs, pr, i, p, q, H, kf)),
                n_params=info["n_params"], loss_type="RMSE val (gốc)",
                ckpt={f"B_h{h + 1}": b for h, b in enumerate(Bs)}, note=note)


def fit_lstm(ctx):
    """LSTM THAM CHIẾU (dự báo điểm, MSE, đầu ra trực tiếp H·K). Cửa sổ chung L=p cho cả Y và X.
    Chưa chạy thử trong môi trường soạn thảo (không có PyTorch): hãy chạy thử trên máy bạn."""
    import torch
    import torch.nn as nn
    prep, p, H, hy, log = ctx["prep"], ctx["p"], ctx["H"], ctx["hyper"], ctx["log"]
    seed_everything(ctx["seed"])
    K, M, L = prep.K, prep.M, p
    if ctx["q"] != p:
        log(f"Lưu ý: LSTM tham chiếu dùng cửa sổ chung p={p} cho cả X (q={ctx['q']} bị bỏ qua).")

    def window(i):
        return np.hstack([prep.Zy[i - L + 1:i + 1], prep.Zx[i - L + 1:i + 1]])

    def make(seg):
        idx = prep.origins(seg, H, lookback=L)
        x = np.stack([window(i) for i in idx])
        y = np.stack([prep.Zy[i + 1:i + 1 + H].reshape(-1) for i in idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    Xtr, Ytr = make("train")
    Xvl, Yvl = make("val")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(K + M, hy["hidden"], hy["layers"], batch_first=True,
                                dropout=hy["dropout"] if hy["layers"] > 1 else 0.0)
            self.head = nn.Linear(hy["hidden"], H * K)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=hy["lr"])
    gen = torch.Generator().manual_seed(ctx["seed"])
    bs = hy["batch"]

    def train_epoch():
        net.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for k in range(0, len(Xtr), bs):
            b = perm[k:k + bs]
            opt.zero_grad()
            loss = torch.mean((net(Xtr[b]) - Ytr[b]) ** 2)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        return tot / len(Xtr)

    def val_epoch():
        net.eval()
        with torch.no_grad():
            return torch.mean((net(Xvl) - Yvl) ** 2).item()

    hist = train_with_early_stopping(
        train_epoch, val_epoch,
        snapshot=lambda: {k: v.detach().clone() for k, v in net.state_dict().items()},
        restore=net.load_state_dict, max_epochs=hy["epochs"], patience=hy["patience"], log=log)
    net.eval()

    def forecaster(pr, i):
        with torch.no_grad():
            out = net(torch.tensor(window(i)[None], dtype=torch.float32)).numpy().reshape(H, K)
        return pr.invert(i, out)

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"LSTM_f{ctx['fold']}_s{ctx['seed']}.pt")
    torch.save({"state": net.state_dict(), "hyper": hy, "seed": ctx["seed"]}, path)
    return dict(forecaster=forecaster, n_params=sum(p_.numel() for p_ in net.parameters()),
                loss_type="MSE chuẩn hóa", val_loss=hist["best_val"], best_epoch=hist["best_epoch"],
                curves={"train": hist["train"], "val": hist["val"]}, ckpt=path)


# Mô hình chưa có code thật: để None → trạng thái "Chưa nối" (TODO: HOOK)
HOOKS = {"VARX-L": None, "DeepVAR": None, "DeepVARX": None}
try:
    from models_hooks import HOOKS_USER      # noqa: E402  (tệp của bạn, nếu có)
    HOOKS.update(HOOKS_USER)
except ImportError:
    pass
FITTERS = {"Naive": fit_naive, "VAR": fit_var, "VARX": fit_varx, "LSTM": fit_lstm,
           **{k: v for k, v in HOOKS.items() if v is not None}}


def run_one(name, ctx):
    """Huấn luyện 1 lần (1 mô hình × 1 fold × 1 seed) và đo VAL. Không đụng tới TEST."""
    prep, H = ctx["prep"], ctx["H"]
    lines = []

    def log(msg):
        lines.append(msg)
        if ctx.get("ui_log"):
            ctx["ui_log"](msg)

    ctx = dict(ctx, log=log)
    rec = dict(model=name, fold=ctx["fold"], seed=ctx["seed"], status="Done", error=None)
    if name not in FITTERS:
        rec.update(status="Chưa nối", error="Chưa có hook cho mô hình này.")
        return rec, None
    try:
        t0 = time.time()
        res = FITTERS[name](ctx)
        rec["train_time"] = time.time() - t0                  # CHỈ thời gian fit (không gồm dự báo)
        Pv, Av = predict(res["forecaster"], prep, "val", H)
        vm = metrics(Pv, Av)
        ck = res.get("ckpt")
        if isinstance(ck, dict):
            os.makedirs(CKPT_DIR, exist_ok=True)
            path = os.path.join(CKPT_DIR, f"{name}_f{ctx['fold']}_s{ctx['seed']}.npz")
            np.savez(path, **ck)
            ck = path
        rec.update(n_params=res.get("n_params"), loss_type=res.get("loss_type", "RMSE val (gốc)"),
                   val_loss=res.get("val_loss", vm["RMSE"]), best_epoch=res.get("best_epoch"),
                   curves=res.get("curves"), ckpt=ck, aic=res.get("aic"), bic=res.get("bic"),
                   note=res.get("note"), val_metrics=vm)
        rec["log"] = lines[-300:]
        return rec, res
    except Exception as e:                                    # lỗi được ghi, không nuốt
        rec.update(status="Lỗi", error=f"{type(e).__name__}: {e}", log=lines[-300:])
        return rec, None


def summarize_runs(runs):
    """Bảng trạng thái theo đặc tả: Status, Training time, Parameters, Val loss, Best epoch, Checkpoint."""
    rows = []
    for name in PIPELINE:
        rs = [r for r in runs if r["model"] == name]
        if not rs:
            continue
        ok = [r for r in rs if r["status"] == "Done"]
        if len(ok) == len(rs):
            status = "Done"
        elif any(r["status"] == "Chưa nối" for r in rs):
            status = "Chưa nối"
        else:
            status = f"Lỗi ({len(rs) - len(ok)}/{len(rs)} lần chạy)"

        def ms(key, fmt):
            v = [r[key] for r in ok if r.get(key) is not None]
            if not v:
                return "—"
            return (fmt.format(np.mean(v)) + (f" ± {fmt.format(np.std(v, ddof=1))}" if len(v) > 1 else ""))
        be = [r["best_epoch"] for r in ok if r.get("best_epoch")]
        ck = [r["ckpt"] for r in ok if r.get("ckpt")]
        rows.append({"Mô hình": name, "Trạng thái": status, "Số lần chạy": len(rs),
                     "Thời gian huấn luyện (s)": ms("train_time", "{:.2f}"),
                     "Tham số": (ok[0].get("n_params") if ok else None),
                     "Val loss": ms("val_loss", "{:.4g}"),
                     "Loại loss": (ok[0].get("loss_type") if ok else "—"),
                     "Best epoch (trung vị)": (int(np.median(be)) if be else "—"),
                     "AIC / BIC (train)": (f"{ok[0]['aic']:.3f} / {ok[0]['bic']:.3f}"
                                           if ok and ok[0].get("aic") is not None else "—"),
                     "Checkpoint": (f"{ck[0]}" + (f" (+{len(ck) - 1})" if len(ck) > 1 else "")) if ck else "—",
                     "Ghi chú": (ok[0].get("note") if ok else (rs[0].get("error") or ""))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# GIAO DIỆN
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Huấn luyện mô hình", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()
    _e, _x = st.session_state.get("endogenous_vars", []), st.session_state.get("exogenous_vars", [])
    if _e:
        st.info("Endog: " + ", ".join(_e[:3]))
    if _x:
        st.info(f"Exog: {len(_x)} biến")

st.caption(APP_HEADER)
st.title("Trang 14 — Huấn luyện mô hình")
st.markdown("**Câu hỏi của trang:** mỗi mô hình đã được huấn luyện đúng quy trình và đủ ổn định chưa? "
            "Trang chỉ dùng **TRAIN** để học và **VAL** để dừng/chọn. **TEST không được tính ở đây.**")

# ── Điều kiện đầu vào ─────────────────────────────────────────────────────
proc, raw = st.session_state.get("processed_data"), st.session_state.get("raw_data")
df_all = proc if proc is not None else raw
endo = st.session_state.get("endogenous_vars", [])
exog = st.session_state.get("exogenous_vars", [])
if df_all is None or not endo:
    st.warning("Chưa đủ điều kiện. Quay lại Trang 1 & 13.")
    st.stop()

cfg_in = st.session_state.get("experiment_config")
if cfg_in is None:                                            # tương thích khóa cũ
    sc_old = st.session_state.get("split_config", {})
    p0 = int(st.session_state.get("lookback_p", 30))
    cfg_in = {"p": p0, "q": p0, "H": int(st.session_state.get("horizon_q", 7)),
              "transform": {"log": False, "diff": 0},
              "scaler": st.session_state.get("scaler_type", "StandardScaler"),
              "split": {"train_ratio": sc_old.get("train_ratio", 0.72),
                        "val_ratio": sc_old.get("val_ratio", 0.08),
                        "test_ratio": sc_old.get("test_ratio", 0.20)}}
    st.warning("Chưa có `experiment_config` từ Trang 13 — đang đọc khóa cũ/mặc định. "
               "Hãy hoàn tất Trang 13 để cấu hình được kiểm toán.")

audit = st.session_state.get("leakage_audit")
valid_experiment = bool(audit and audit.get("passed"))
if not valid_experiment:
    st.error("Leakage Audit (Trang 13) chưa PASS hoặc chưa chạy → huấn luyện bị khóa.")
    if not st.checkbox("Chạy thử không có audit (kết quả đánh dấu KHÔNG HỢP LỆ, không được đưa vào Trang 16)"):
        st.stop()

# ── Dữ liệu chuỗi thời gian ───────────────────────────────────────────────
date_col = st.session_state.get("date_col", "date")
group_col = st.session_state.get("group_col", "province")
all_vars = [c for c in endo + [e for e in exog if e not in endo] if c in df_all.columns]
if date_col in df_all.columns and group_col in df_all.columns:
    ts = df_all.assign(**{date_col: pd.to_datetime(df_all[date_col])}) \
               .groupby(date_col)[all_vars].mean().sort_index()
elif date_col in df_all.columns:
    ts = df_all.assign(**{date_col: pd.to_datetime(df_all[date_col])}).set_index(date_col)[all_vars].sort_index()
else:
    ts = df_all[all_vars].copy()
n_before = len(ts)
ts = ts.dropna()
if len(ts) < n_before:
    st.warning(f"Đã bỏ {n_before - len(ts)} hàng có giá trị khuyết (không điền). Cần khớp cách xử lý khuyết "
               "của Trang 13; hàng bị bỏ có thể tạo khoảng trống ngày làm lệch lag.")
endo_u = [c for c in endo if c in ts.columns]
exog_u = [c for c in exog if c in ts.columns and c not in endo_u]
cols = list(ts.columns)
endo_idx = [cols.index(c) for c in endo_u]
exog_idx = [cols.index(c) for c in exog_u]
raw_np = ts.values.astype(float)
D_HASH = data_hash(raw_np, cols)

p_, q_, H_ = int(cfg_in["p"]), int(cfg_in.get("q", cfg_in["p"])), int(cfg_in["H"])
tf = cfg_in.get("transform", {"log": False, "diff": 0})
kf_names = [c for c in cfg_in.get("known_future_vars", []) if c in exog_u]
kf_mask = np.array([c in kf_names for c in exog_u], bool)

# folds
Nt = len(ts) - (1 if tf.get("diff") else 0)
if cfg_in.get("folds"):
    folds = [dict(f) for f in cfg_in["folds"]]
else:
    sp = cfg_in["split"]
    ntr, nvl = int(Nt * sp["train_ratio"]), int(Nt * sp["val_ratio"])
    folds = [{"n_train": ntr, "n_val": nvl, "n_test": Nt - ntr - nvl}]
F_ = len(folds)

# ── Tóm tắt cấu hình ──────────────────────────────────────────────────────
st.subheader("Cấu hình thực nghiệm (đọc từ Trang 13, chỉ xem)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nội sinh K", len(endo_u))
c2.metric("Ngoại sinh M", len(exog_u))
c3.metric("p / q / H", f"{p_} / {q_} / {H_}")
c4.metric("Số fold F", F_)
c5.metric("Scaler", cfg_in.get("scaler", "StandardScaler"))
f0 = folds[0]
st.caption(f"Fold 1: TRAIN {f0['n_train']:,} · VAL {f0['n_val']:,} · TEST {f0['n_test']:,} quan sát. "
           f"Biến đổi: log={bool(tf.get('log'))}, sai phân={int(tf.get('diff', 0))}. "
           f"X known-future: {kf_names or 'không có (mọi X là past-only)'}. "
           f"Hash dữ liệu: `{D_HASH[:12]}…`")
if not valid_experiment:
    st.error("⚠ THỰC NGHIỆM CHƯA ĐƯỢC KIỂM TOÁN — kết quả không hợp lệ cho báo cáo.")

CFG_EFF = dict(p=p_, q=q_, H=H_, transform=tf, scaler=cfg_in.get("scaler"), folds=folds,
               known_future=kf_names, endo=endo_u, exog=exog_u)
CFG_SIG = hashlib.md5((json.dumps(CFG_EFF, default=str, sort_keys=True) + D_HASH).encode()).hexdigest()[:12]
if st.session_state.get("t14_cfg_sig") not in (None, CFG_SIG):
    st.info("Cấu hình hoặc dữ liệu đã đổi so với lần huấn luyện trước — kết quả cũ đã bị xóa.")
    for k in ("t14_runs", "t14_fitted", "model_results", "t14_manifest"):
        st.session_state.pop(k, None)
    st.session_state["test_eval_count"] = 0
st.session_state["t14_cfg_sig"] = CFG_SIG

# ── Chọn pipeline & siêu tham số ──────────────────────────────────────────
st.divider()
st.subheader("Pipeline huấn luyện")
available = [m for m in PIPELINE if m in FITTERS]
missing = [m for m in PIPELINE if m not in FITTERS]
sel = st.multiselect("Mô hình chạy (theo thứ tự pipeline)", PIPELINE,
                     default=[m for m in PIPELINE if m in available])
if missing:
    st.caption("Chưa nối hook (hiển thị 'Chưa nối', không sinh số giả): " + ", ".join(missing)
               + ". Nối qua `models_hooks.HOOKS_USER` — xem docstring đầu tệp.")

hyper = dict(hidden=128, layers=2, dropout=0.2, epochs=100, lr=1e-3, patience=10, batch=64)
S_ = 1
if any(m in NEURAL for m in sel):
    st.markdown("**Mô hình mạng**")
    h1, h2, h3, h4 = st.columns(4)
    hyper["hidden"] = int(h1.number_input("Hidden size", 16, 512, 128, step=16))
    hyper["layers"] = int(h2.number_input("Số lớp LSTM", 1, 4, 2))
    hyper["dropout"] = float(h3.slider("Dropout", 0.0, 0.5, 0.2, 0.05))
    hyper["epochs"] = int(h4.number_input("Epoch tối đa", 10, 300, 100, step=10))
    h5, h6, h7 = st.columns(3)
    hyper["lr"] = float(h5.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3))
    hyper["patience"] = int(h6.slider("Patience (early stopping theo Val loss)", 3, 30, 10))
    hyper["batch"] = int(h7.select_slider("Batch size", [16, 32, 64, 128, 256], value=64))
    S_ = int(st.slider("Số seed S (mô hình mạng)", 1, 10, 5))
    base_seed = int(st.number_input("Seed gốc", 0, 10 ** 6, 42))
    rel = seed_sd_rel_error(S_)
    st.caption("Độ lệch chuẩn ước lượng từ S seed có sai số tương đối xấp xỉ 1/√(2(S−1)): "
               + (f"S={S_} → khoảng {rel:.0%}." if S_ >= 2 else "S=1 → không ước lượng được nhiễu seed."))
else:
    base_seed = 42
seeds = [base_seed + k for k in range(S_)]

n_cls = sum(m in sel for m in ("Naive", "VAR", "VARX", "VARX-L"))
n_net = sum(m in NEURAL for m in sel)
n_total = n_runs_estimate(n_cls, n_net, F_, S_)
st.info(f"**Quy mô tính toán:** {n_cls} mô hình cổ điển × {F_} fold + {n_net} mô hình mạng × {F_} fold × "
        f"{S_} seed = **{n_total} lần chạy** (riêng mạng: {n_net * F_ * S_}).")

if st.button("Bắt đầu huấn luyện pipeline", type="primary", use_container_width=True, disabled=not sel):
    runs, fitted = [], {}
    st.session_state["test_eval_count"] = 0 if st.session_state.get("test_eval_count") is None \
        else st.session_state["test_eval_count"]
    log_area, bar = st.empty(), st.progress(0.0)
    buf, done = [], 0

    def ui_log(msg):
        buf.append(msg)
        log_area.code("\n".join(buf[-25:]), language="text")

    for name in [m for m in PIPELINE if m in sel]:
        is_net = name in NEURAL
        for fi, fd in enumerate(folds):
            try:
                prep = Prep(raw_np, endo_idx, exog_idx, fd["n_train"], fd["n_val"],
                            n_end=fd["n_train"] + fd["n_val"] + fd["n_test"],
                            scaler_kind=cfg_in.get("scaler", "StandardScaler"),
                            use_log=tf.get("log", False), diff=tf.get("diff", 0))
            except Exception as e:
                runs.append(dict(model=name, fold=fi, seed=None, status="Lỗi",
                                 error=f"Prep: {e}", log=[]))
                continue
            for sd in (seeds if is_net else [seeds[0]]):
                ui_log(f"── {name} | fold {fi + 1}/{F_} | seed {sd} ──")
                ctx = dict(prep=prep, p=p_, q=q_, H=H_, seed=sd, fold=fi, hyper=hyper,
                           known_future=kf_mask, ui_log=ui_log)
                rec, res = run_one(name, ctx)
                runs.append(rec)
                if res is not None:
                    fitted[(name, fi, sd)] = dict(res=res, prep=prep)
                if rec["status"] != "Done":
                    ui_log(f"[{rec['status']}] {rec.get('error')}")
                else:
                    ui_log(f"Val RMSE (gốc) = {rec['val_metrics']['RMSE']:.4f} · "
                           f"thời gian fit = {rec['train_time']:.2f}s")
                done += 1
                bar.progress(min(done / max(n_total, 1), 1.0))
    st.session_state["t14_runs"] = runs
    st.session_state["t14_fitted"] = fitted
    st.session_state["t14_manifest"] = build_manifest(CFG_EFF, D_HASH, sel, seeds, hyper, folds)
    st.session_state["model_results"] = {}               # kết quả TEST cũ không còn hợp lệ

runs = st.session_state.get("t14_runs", [])
if not runs:
    st.stop()

# ── Kết quả huấn luyện ────────────────────────────────────────────────────
tab_a, tab_b, tab_c, tab_d = st.tabs(["Bảng trạng thái", "Nhật ký epoch", "Trang 15 — Learning curve",
                                      "Manifest tái lập"])

with tab_a:
    st.dataframe(summarize_runs(runs), use_container_width=True, hide_index=True)
    st.caption("Val loss **không so sánh giữa các mô hình** vì hàm mất mát khác nhau (NLL của DeepVAR/X, "
               "MSE của LSTM, RMSE gốc của mô hình cổ điển). So sánh mô hình dùng các số đo ở Trang 16.")
    vm_rows = [{"Mô hình": r["model"], "Fold": r["fold"] + 1, "Seed": r["seed"],
                "Val MAE": r["val_metrics"]["MAE"], "Val RMSE": r["val_metrics"]["RMSE"],
                "Val sMAPE (%)": r["val_metrics"]["sMAPE"]}
               for r in runs if r.get("status") == "Done"]
    if vm_rows:
        vdf = pd.DataFrame(vm_rows)
        with st.expander("Số đo VAL theo từng lần chạy (đơn vị gốc, VND/kg)"):
            st.dataframe(vdf, use_container_width=True, hide_index=True)
        nets = [m for m in NEURAL if m in vdf["Mô hình"].values and vdf[vdf["Mô hình"] == m]["Seed"].nunique() > 1]
        if nets:
            st.markdown("**Nhiễu giữa các seed (Val RMSE, đơn vị gốc)** — chênh lệch giữa hai mô hình nhỏ hơn "
                        "nhiễu này thì **không được kết luận** ở Trang 16.")
            sd_rows = []
            for m in nets:
                v = vdf[vdf["Mô hình"] == m].groupby("Fold")["Val RMSE"].agg(["mean", "std", "count"])
                for fo, row in v.iterrows():
                    sd_rows.append({"Mô hình": m, "Fold": fo, "Val RMSE (mean)": row["mean"],
                                    "SD giữa seed": row["std"], "S": int(row["count"]),
                                    "Sai số tương đối của SD": seed_sd_rel_error(int(row["count"]))})
            st.dataframe(pd.DataFrame(sd_rows).style.format(
                {"Val RMSE (mean)": "{:.4f}", "SD giữa seed": "{:.4f}", "Sai số tương đối của SD": "{:.0%}"}),
                use_container_width=True, hide_index=True)
    errs = [r for r in runs if r["status"] == "Lỗi"]
    for r in errs[:10]:
        st.error(f"{r['model']} (fold {r['fold'] + 1}, seed {r.get('seed')}): {r.get('error')}")

with tab_b:
    ms = sorted({r["model"] for r in runs})
    c1, c2, c3 = st.columns(3)
    m_sel = c1.selectbox("Mô hình", ms, key="t14_logm")
    sub = [r for r in runs if r["model"] == m_sel]
    f_sel = c2.selectbox("Fold", sorted({r["fold"] for r in sub}), format_func=lambda v: v + 1, key="t14_logf")
    sub = [r for r in sub if r["fold"] == f_sel]
    s_sel = c3.selectbox("Seed", [r.get("seed") for r in sub], key="t14_logs")
    rr = [r for r in sub if r.get("seed") == s_sel][0]
    st.code("\n".join(rr.get("log", [])) or "(không có nhật ký — mô hình cổ điển không có epoch)", language="text")
    if rr.get("best_epoch"):
        st.success(f"Best epoch = {rr['best_epoch']} · Val loss tại best epoch = {rr['val_loss']:.6f} · "
                   f"thời gian fit = {rr['train_time']:.2f}s")

with tab_c:
    st.markdown("**Mục đích:** chẩn đoán quá khớp / chưa hội tụ. **Không dùng để kết luận hiệu năng** "
                "(việc đó thuộc Trang 16).")
    cm = [m for m in NEURAL if any(r["model"] == m and r.get("curves") for r in runs)]
    if not cm:
        st.info("Chưa có mô hình mạng nào trả về đường học (cần LSTM hoặc hook DeepVAR/DeepVARX).")
    else:
        c1, c2 = st.columns(2)
        cmod = c1.selectbox("Mô hình", cm, key="t15_m")
        cs = [r for r in runs if r["model"] == cmod and r.get("curves")]
        cfold = c2.selectbox("Fold", sorted({r["fold"] for r in cs}), format_func=lambda v: v + 1, key="t15_f")
        cs = [r for r in cs if r["fold"] == cfold]
        trm, trs, nn_ = agg_curves([r["curves"]["train"] for r in cs])
        vlm, vls, _ = agg_curves([r["curves"]["val"] for r in cs])
        ep = np.arange(1, len(trm) + 1)
        fig = go.Figure()
        for nm, m_, s_, col in (("Train", trm, trs, "#3b82f6"), ("Val", vlm, vls, "#ef4444")):
            fig.add_trace(go.Scatter(x=ep, y=m_ + s_, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=ep, y=m_ - s_, fill="tonexty", line=dict(width=0),
                                     fillcolor=col.replace("#", "rgba(") if False else "rgba(150,150,150,0.2)",
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=ep, y=m_, name=f"{nm} (mean{' ± SD' if len(cs) > 1 else ''})",
                                     line=dict(color=col)))
        for r in cs:
            if r.get("best_epoch"):
                fig.add_shape(type="line", x0=r["best_epoch"], x1=r["best_epoch"], y0=0, y1=1, yref="paper",
                              line=dict(dash="dot", width=1, color="#22c55e"))
        fig.update_layout(height=420, xaxis_title="epoch", yaxis_title="loss", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Đường nét chấm xanh: best epoch của từng seed. Train loss tính khi dropout đang bật nên "
                   "thường cao hơn so với khi suy luận — G = L_val − L_train bên dưới chỉ để chẩn đoán.")
        gr = []
        for r in cs:
            tr_, vl_, be = r["curves"]["train"], r["curves"]["val"], r["best_epoch"]
            st_ep = overfit_start(tr_, vl_, hyper["patience"])
            if st_ep is None:
                flag = "Không"
            elif st_ep < be:
                flag = f"⚠ quá khớp bắt đầu từ epoch {st_ep} (trước best epoch {be})"
            else:
                flag = f"⚠ quá khớp bắt đầu từ epoch {st_ep}"
            gr.append({"Seed": r["seed"], "Best epoch": be, "L_train": tr_[be - 1], "L_val": vl_[be - 1],
                       "G = L_val − L_train": vl_[be - 1] - tr_[be - 1], "Epoch đã chạy": len(vl_),
                       f"Cờ quá khớp (patience={hyper['patience']})": flag})
        st.dataframe(pd.DataFrame(gr).style.format(
            {"L_train": "{:.5f}", "L_val": "{:.5f}", "G = L_val − L_train": "{:.5f}"}),
            use_container_width=True, hide_index=True)
        if len(cs) > 1 and nn_.min() < len(cs):
            st.caption(f"Số seed còn chạy ở epoch cuối: {int(nn_[-1])}/{len(cs)} (các seed dừng sớm khác nhau).")

with tab_d:
    man = st.session_state.get("t14_manifest")
    if man:
        st.json(man, expanded=False)
        st.download_button("Tải manifest (JSON)", json.dumps(man, ensure_ascii=False, indent=2, default=str),
                           "manifest_trang14.json", "application/json")
        st.caption("Gồm seed, phiên bản thư viện, phần cứng, hash dữ liệu và mã nguồn, cấu hình — dùng cho "
                   "Sổ minh chứng (Trang 17).")

# ── Ghi thẻ minh chứng ────────────────────────────────────────────────────
evid = st.session_state.setdefault("evidence_log", {})
for _, row in summarize_runs(runs).iterrows():
    evid[f"T14-{row['Mô hình']}"] = {
        "ID": f"T14-{row['Mô hình']}", "Trang": "14", "Kết luận": f"Trạng thái huấn luyện: {row['Trạng thái']}",
        "Con số": f"Tham số={row['Tham số']}; Val loss={row['Val loss']} ({row['Loại loss']}); "
                  f"best epoch={row['Best epoch (trung vị)']}; thời gian fit={row['Thời gian huấn luyện (s)']}s",
        "Phương pháp": "Huấn luyện trên TRAIN, dừng/chọn theo VAL; không dùng TEST",
        "Đoạn dữ liệu": f"{F_} fold; hash dữ liệu {D_HASH[:12]}", "Nguồn": "Manifest Trang 14",
        "Giới hạn": "Val loss không so sánh được giữa các mô hình khác hàm mất mát.", "sig": CFG_SIG}

# ── Khóa & đánh giá TEST (một lần) ────────────────────────────────────────
st.divider()
st.subheader("Khóa cấu hình và đánh giá TEST (một lần)")
n_seen = int(st.session_state.get("test_eval_count", 0) or 0)
st.caption("Chỉ bấm khi **đã chốt** p, q, H, siêu tham số và danh sách mô hình. Mỗi lần bấm là một lần xem TEST; "
           "số lần được ghi vào `st.session_state['test_eval_count']` để dòng 8 của Leakage Audit "
           "(“số lần đánh giá test = 1”) đọc.")
if n_seen >= 1:
    st.warning(f"TEST đã được xem **{n_seen}** lần. Xem lại sau khi chỉnh cấu hình là chọn theo TEST; "
               "kết quả sẽ bị đánh dấu không đạt dòng 8 của audit.")
fitted = st.session_state.get("t14_fitted", {})
if st.button("Khóa và sinh dự báo TEST", disabled=not fitted or not valid_experiment,
             help="Bị khóa nếu chưa có mô hình nào huấn luyện xong hoặc audit chưa PASS."):
    n_seen += 1
    st.session_state["test_eval_count"] = n_seen
    results = {}
    naive_mae = {}
    per = {}
    for (name, fi, sd), obj in fitted.items():
        t0 = time.time()
        P, A = predict(obj["res"]["forecaster"], obj["prep"], "test", H_)
        per.setdefault(name, []).append(dict(fold=fi, seed=sd, P=P, A=A, m=metrics(P, A),
                                             t=time.time() - t0))
    for name, lst in per.items():
        if name == "Naive":
            for it in lst:
                naive_mae[it["fold"]] = it["m"]["MAE"]
    for name, lst in per.items():
        first = [it for it in lst if it["fold"] == lst[0]["fold"]]
        Pm = np.mean([it["P"] for it in first], axis=0)           # trung bình qua seed (cùng fold)
        tm = {k: float(np.mean([it["m"][k] for it in lst])) for k in ("MAE", "RMSE", "sMAPE")}
        if naive_mae:
            tm["rMAE"] = float(np.mean([it["m"]["MAE"] / naive_mae[it["fold"]]
                                        for it in lst if it["fold"] in naive_mae]))
        tr_ = [r for r in runs if r["model"] == name and r.get("status") == "Done"]
        results[name] = {
            "model": name, "p": p_, "q": q_, "H": H_,
            "val_metrics": {k: float(np.mean([r["val_metrics"][k] for r in tr_])) for k in ("MAE", "RMSE", "sMAPE")},
            "test_metrics": tm,
            "test_metrics_by_run": [dict(fold=it["fold"], seed=it["seed"], **it["m"]) for it in lst],
            "train_time": float(np.mean([r["train_time"] for r in tr_])),
            "test_time": float(np.mean([it["t"] for it in lst])),
            "pred_test": Pm.tolist(),        # (n_gốc, H, K) — đơn vị gốc, trung bình qua seed của fold đầu
            "actual_test": first[0]["A"].tolist(),
            "endo_names": endo_u, "valid": valid_experiment,
            "test_views": n_seen}
    st.session_state["model_results"] = results
    st.success(f"Đã sinh dự báo TEST cho {len(results)} mô hình (lần xem TEST thứ {n_seen}).")

res_show = st.session_state.get("model_results", {})
if res_show:
    rows = [{"Mô hình": m, "TEST MAE": r["test_metrics"]["MAE"], "TEST RMSE": r["test_metrics"]["RMSE"],
             "TEST sMAPE (%)": r["test_metrics"]["sMAPE"], "rMAE (so với Naive)": r["test_metrics"].get("rMAE")}
            for m, r in res_show.items()]
    st.dataframe(pd.DataFrame(rows).style.format(
        {"TEST MAE": "{:.4f}", "TEST RMSE": "{:.4f}", "TEST sMAPE (%)": "{:.3f}", "rMAE (so với Naive)": "{:.3f}"},
        na_rep="—"), use_container_width=True, hide_index=True)
    st.caption("Đơn vị gốc (VND/kg). Bảng này chỉ để kiểm tra đã sinh dự báo; so sánh có kiểm định "
               "(Diebold–Mariano, khoảng tin cậy) thực hiện ở Trang 16.")

st.divider()
if st.button("Tiếp theo: Kết quả & So sánh →", use_container_width=True):
    st.switch_page("pages/7_Ket_qua.py")
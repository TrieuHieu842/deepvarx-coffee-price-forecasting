"""Grid Search tìm siêu tham số cho 3 mô hình học sâu: DeepVAR, LSTM, DeepVARX.
Bao phủ 100% các siêu tham số trong Bảng điều khiển siêu tham số của đề tài:

1. Số Epochs & Early Stopping (Patience)
2. Batch Size: [16, 32, 64, 128, 256]
3. Learning Rate: [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
4. Hidden Size: [16, 32, 48, 64, 128]
5. Độ dài chuỗi (seq_len): [5, 10, 15, 20]
6. Số lớp LSTM (layers): [1, 2, 3]
7. Hàm mất mát (Loss Function):
   - Huber (beta=1.0)
   - MSE Loss
   - Smooth L1 (beta=0.01)
8. Hệ số phạt sai hướng (Direction Penalty lambda): [0.0, 0.02, 0.05, 0.1]
9. Bật Residual Learning: [True, False]
10. Bật Temporal Attention (LSTM + Attention): [True, False] (cho mô hình LSTM)
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Thêm thư mục DeepVARX vào sys.path để nạp đúng các module
_CURR_DIR = Path(__file__).resolve().parent
if str(_CURR_DIR) not in sys.path:
    sys.path.insert(0, str(_CURR_DIR))

from modules.data_utils import metrics
from modules.models.deepvar import DeepVAR, DeepVARConfig
from modules.models.deepvarx import DeepVARX, DirectionAwareLoss
from modules.models.lstm_attention import LSTMForecaster, LSTMAttentionForecaster
from modules.panel_data import detect_columns, normalize_wide, train_end_from_ratio, wide_gap_report, wide_interpolate

# Cấu hình danh sách biến chuẩn
DEFAULT_ENDOG_COLS = [
    "price_dak_lak",
    "price_lam_dồng",
    "price_gia_lai",
    "price_dak_nong",
]

DEFAULT_EXOG_COLS = [
    "ret_london",
    "ret_arabica",
    "ret_brent",
    "ret_dxy",
    "ret_usd_vnd",
    "temp_mean_ch",
    "precip_nasa_ch",
    "humidity_air_ch",
    "london_price_usd_ton",
    "arabica_price_usd_lb",
]

MODEL_CANONICAL = {
    "deepvar": "DeepVAR",
    "lstm": "LSTM",
    "deepvarx": "DeepVARX",
}

PROVINCE_LABELS = {
    "price_dak_lak": "Đắk Lắk",
    "price_lam_dồng": "Lâm Đồng",
    "price_gia_lai": "Gia Lai",
    "price_dak_nong": "Đắk Nông",
}


def load_dataset(
    csv_path: str | Path,
    endog_cols: List[str] | None = None,
    exog_cols: List[str] | None = None,
    split_mode: str = "pages",
) -> Tuple[pd.DataFrame, int, int, int, List[str], List[str]]:
    """Nạp dữ liệu theo đúng pipeline Trang 1–5 (Wide + nội suy ffill + chia 70% Train+Val [85/15] / 30% Test)."""
    csv_file = Path(csv_path)
    if not csv_file.is_absolute() and not csv_file.exists():
        candidate = _CURR_DIR / csv_file
        if candidate.exists():
            csv_file = candidate
    if not csv_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {csv_file}")

    raw = pd.read_csv(csv_file)
    det = detect_columns(raw)
    date_col = det["date"] or "date"
    master, _ = normalize_wide(raw, date_col, dayfirst=False, dup_policy="mean")

    active_endog = endog_cols if endog_cols else [c for c in DEFAULT_ENDOG_COLS if c in master.columns]
    if exog_cols:
        active_exog = exog_cols
    else:
        active_exog = [c for c in DEFAULT_EXOG_COLS if c in master.columns]

    missing_endog = [c for c in active_endog if c not in master.columns]
    if missing_endog:
        raise ValueError(f"Thiếu các cột biến nội sinh trong CSV: {missing_endog}")

    missing_exog = [c for c in active_exog if c not in master.columns]
    if missing_exog:
        raise ValueError(f"Thiếu các cột biến ngoại sinh trong CSV: {missing_exog}")

    train_end = train_end_from_ratio(master, date_col, 0.70)
    # Áp dụng lọc gap & nội suy ffill tương tự Trang 1 (bảo toàn 3.608 ngày từ 2013-01-02)
    nums_all = [c for c in master.columns if c != date_col and pd.api.types.is_numeric_dtype(master[c])]
    x_all = [c for c in nums_all if c not in active_endog and c not in det.get("derived_endog", [])]
    check_cols = active_endog + [c for c in x_all if c not in active_endog]
    rep = wide_gap_report(master, date_col, check_cols, train_end)
    drop_gap = set(
        rep.loc[
            (~rep["Biến"].isin(set(active_endog)))
            & ((rep["Max gap (dòng)"] > 30) | (rep["% khuyết (toàn bộ)"] > 40)),
            "Biến",
        ].tolist()
    )
    after_cols = [c for c in check_cols if c not in drop_gap]
    interp = wide_interpolate(master, date_col, after_cols, train_end, "ffill")
    keep_cols = [date_col] + active_endog + [c for c in active_exog if c not in active_endog]
    df = interp[keep_cols].rename(columns={date_col: "date"}).sort_values("date").reset_index(drop=True)

    T = len(df)
    if split_mode == "70_15_15":
        n_train = int(T * 0.70)
        n_val = int(T * 0.15)
        n_test = T - n_train - n_val
    else:
        n_train_total = int((pd.to_datetime(df["date"]) <= pd.Timestamp(train_end)).sum())
        n_val = max(1, int(n_train_total * 0.15))
        n_train = n_train_total - n_val
        n_test = T - n_train_total

    return df, n_train, n_val, n_test, active_endog, active_exog


def compute_naive_benchmark(df: pd.DataFrame, n_train: int, n_val: int, endog_cols: List[str]) -> Dict[str, float]:
    """Tính toán sai số Naive Benchmark (y_hat_t = y_{t-1}) trên Val & Test để làm thước đo so sánh."""
    Y = df[endog_cols].values.astype(np.float64)

    # Validation
    y_true_val = Y[n_train: n_train + n_val]
    y_prev_val = Y[n_train - 1: n_train + n_val - 1]

    # Test
    y_true_test = Y[n_train + n_val:]
    y_prev_test = Y[n_train + n_val - 1: -1]

    val_res = [metrics(y_true_val[:, k], y_prev_val[:, k]) for k in range(len(endog_cols))]
    test_res = [metrics(y_true_test[:, k], y_prev_test[:, k]) for k in range(len(endog_cols))]

    return {
        "naive_val_rmse": float(np.mean([r["RMSE"] for r in val_res])),
        "naive_val_mae": float(np.mean([r["MAE"] for r in val_res])),
        "naive_val_mape": float(np.mean([r["MAPE (%)"] for r in val_res])),
        "naive_test_rmse": float(np.mean([r["RMSE"] for r in test_res])),
        "naive_test_mae": float(np.mean([r["MAE"] for r in test_res])),
    }


def create_sliding_windows(
    df: pd.DataFrame,
    endog_cols: List[str],
    exog_cols: List[str],
    seq_len: int,
    n_train: int,
    n_val: int,
) -> Dict[str, Any]:
    """Tạo cửa sổ trượt (sliding windows) chuẩn hoá z-score chỉ trên tập Train (Zero Leakage)."""
    feature_cols = endog_cols + exog_cols
    tr_df = df.iloc[:n_train]
    mu = tr_df[feature_cols].mean().values.astype(np.float32)
    sd = tr_df[feature_cols].std().replace(0, 1.0).fillna(1.0).values.astype(np.float32)

    scaled_matrix = ((df[feature_cols].values - mu) / sd).astype(np.float32)
    T = len(df)
    n_out = len(endog_cols)

    idx = np.arange(seq_len, T)
    X = np.stack([scaled_matrix[t - seq_len:t] for t in idx])
    y = scaled_matrix[idx, :n_out]

    m_tr = idx < n_train
    m_va = (idx >= n_train) & (idx < n_train + n_val)
    m_te = idx >= n_train + n_val

    true_prices = df[endog_cols].values.astype(np.float64)
    y_true_va = true_prices[idx[m_va]]
    y_prev_va = true_prices[idx[m_va] - 1]
    y_true_te = true_prices[idx[m_te]]
    y_prev_te = true_prices[idx[m_te] - 1]

    return {
        "X_tr": torch.tensor(X[m_tr], dtype=torch.float32),
        "y_tr": torch.tensor(y[m_tr], dtype=torch.float32),
        "X_va": torch.tensor(X[m_va], dtype=torch.float32),
        "y_va": torch.tensor(y[m_va], dtype=torch.float32),
        "X_te": torch.tensor(X[m_te], dtype=torch.float32),
        "y_te": torch.tensor(y[m_te], dtype=torch.float32),
        "mu_y": mu[:n_out],
        "sd_y": sd[:n_out],
        "y_true_va": y_true_va,
        "y_prev_va": y_prev_va,
        "y_true_te": y_true_te,
        "y_prev_te": y_prev_te,
        "dates_va": df["date"].iloc[idx[m_va]].values,
        "dates_te": df["date"].iloc[idx[m_te]].values,
    }


def train_eval_neural_net(
    model: nn.Module,
    windows: Dict[str, Any],
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    endog_cols: List[str],
    dir_weight: float = 0.05,
    loss_type: str = "huber",
    huber_beta: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Huấn luyện mô hình PyTorch (LSTM / DeepVARX) với các hàm mất mát & Direction Penalty từ UI."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model = model.to(dev)

    X_tr = windows["X_tr"].to(dev)
    y_tr = windows["y_tr"].to(dev)
    X_va = windows["X_va"].to(dev)
    y_va = windows["y_va"].to(dev)
    X_te = windows["X_te"].to(dev)

    n_out = len(endog_cols)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = DirectionAwareLoss(loss_type=loss_type, beta=huber_beta, dir_weight=dir_weight)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_count = 0
    n_tr = len(X_tr)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        tot_loss = 0.0
        for i in range(0, n_tr, batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            prev_b = X_tr[b][:, -1, :n_out] if getattr(model, "residual", False) else None
            out = model(X_tr[b])
            loss = loss_fn(out, y_tr[b], prev_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item() * len(b)
        scheduler.step()

        # Đánh giá trên Validation
        model.eval()
        with torch.no_grad():
            out_va = model(X_va)
            prev_va = X_va[:, -1, :n_out] if getattr(model, "residual", False) else None
            va_loss = loss_fn(out_va, y_va, prev_va).item()

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Dự báo điểm tối ưu trên Val và Test
    model.eval()
    with torch.no_grad():
        pred_va_s = model(X_va).cpu().numpy()
        pred_te_s = model(X_te).cpu().numpy()

    if pred_va_s.ndim == 1:
        pred_va_s = pred_va_s[:, None]
    if pred_te_s.ndim == 1:
        pred_te_s = pred_te_s[:, None]

    # Chuyển về thang giá thực tế VNĐ/kg
    mu_y, sd_y = windows["mu_y"], windows["sd_y"]
    pred_va = pred_va_s * sd_y + mu_y
    pred_te = pred_te_s * sd_y + mu_y

    # Tính toán Metric
    val_m_list = [
        metrics(windows["y_true_va"][:, k], pred_va[:, k], windows["y_prev_va"][:, k])
        for k in range(n_out)
    ]
    test_m_list = [
        metrics(windows["y_true_te"][:, k], pred_te[:, k], windows["y_prev_te"][:, k])
        for k in range(n_out)
    ]

    metric_keys = ["RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "R²", "Đúng hướng (%)"]
    val_avg = {k: float(np.mean([m[k] for m in val_m_list])) for k in metric_keys}
    test_avg = {k: float(np.mean([m[k] for m in test_m_list])) for k in metric_keys}

    prov_val_rmse = {f"val_rmse_{col}": float(val_m_list[k]["RMSE"]) for k, col in enumerate(endog_cols)}

    return {
        "best_epoch": best_epoch,
        "epochs_run": ep,
        "train_time_sec": round(train_time, 2),
        "val_loss": round(best_val_loss, 5),
        "val_rmse": round(val_avg["RMSE"], 2),
        "val_mae": round(val_avg["MAE"], 2),
        "val_mape": round(val_avg["MAPE (%)"], 2),
        "val_smape": round(val_avg["sMAPE (%)"], 2),
        "val_r2": round(val_avg["R²"], 4),
        "val_da": round(val_avg["Đúng hướng (%)"], 1),
        "test_rmse": round(test_avg["RMSE"], 2),
        "test_mae": round(test_avg["MAE"], 2),
        "test_mape": round(test_avg["MAPE (%)"], 2),
        "test_smape": round(test_avg["sMAPE (%)"], 2),
        "test_r2": round(test_avg["R²"], 4),
        "test_da": round(test_avg["Đúng hướng (%)"], 1),
        **prov_val_rmse,
    }


def evaluate_deepvar(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    endog_cols: List[str],
    hp: Dict[str, Any],
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Huấn luyện và đánh giá mô hình DeepVAR (Parallel MLP) theo Altmeyer et al. (2021)."""
    Y = df[endog_cols].values.astype(np.float64)
    p = hp["seq_len"]
    hidden_val = hp["hidden"]
    if isinstance(hidden_val, tuple):
        hidden = hidden_val
    else:
        hidden = (int(hidden_val), max(16, int(hidden_val) // 2))

    cfg = DeepVARConfig(
        p=p,
        hidden=hidden,
        dropout=hp.get("dropout", 0.1),
        lr=hp["lr"],
        weight_decay=hp.get("weight_decay", 1e-5),
        batch_size=hp["batch_size"],
        max_epochs=hp["epochs"],
        patience=hp["patience"],
        seed=seed,
    )

    t0 = time.perf_counter()
    model = DeepVAR(cfg, device=device)
    model.fit(Y, None, n_train, n_val, verbose=0)
    train_time = time.perf_counter() - t0

    # Dự báo rolling từ n_train
    yhat_all, idx_all = model.predict(Y, None, start=n_train)
    if yhat_all.ndim == 1:
        yhat_all = yhat_all[:, None]

    m_va = (idx_all >= n_train) & (idx_all < n_train + n_val)
    m_te = idx_all >= n_train + n_val

    y_true_va = Y[idx_all[m_va]]
    y_prev_va = Y[idx_all[m_va] - 1]
    yhat_va = yhat_all[m_va]

    y_true_te = Y[idx_all[m_te]]
    y_prev_te = Y[idx_all[m_te] - 1]
    yhat_te = yhat_all[m_te]

    n_out = len(endog_cols)
    val_m_list = [metrics(y_true_va[:, k], yhat_va[:, k], y_prev_va[:, k]) for k in range(n_out)]
    test_m_list = [metrics(y_true_te[:, k], yhat_te[:, k], y_prev_te[:, k]) for k in range(n_out)]

    metric_keys = ["RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "R²", "Đúng hướng (%)"]
    val_avg = {k: float(np.mean([m[k] for m in val_m_list])) for k in metric_keys}
    test_avg = {k: float(np.mean([m[k] for m in test_m_list])) for k in metric_keys}

    prov_val_rmse = {f"val_rmse_{col}": float(val_m_list[k]["RMSE"]) for k, col in enumerate(endog_cols)}

    return {
        "best_epoch": model.best_epoch,
        "epochs_run": model.epochs_run,
        "train_time_sec": round(train_time, 2),
        "val_loss": round(model.best_val_mse, 5),
        "val_rmse": round(val_avg["RMSE"], 2),
        "val_mae": round(val_avg["MAE"], 2),
        "val_mape": round(val_avg["MAPE (%)"], 2),
        "val_smape": round(val_avg["sMAPE (%)"], 2),
        "val_r2": round(val_avg["R²"], 4),
        "val_da": round(val_avg["Đúng hướng (%)"], 1),
        "test_rmse": round(test_avg["RMSE"], 2),
        "test_mae": round(test_avg["MAE"], 2),
        "test_mape": round(test_avg["MAPE (%)"], 2),
        "test_smape": round(test_avg["sMAPE (%)"], 2),
        "test_r2": round(test_avg["R²"], 4),
        "test_da": round(test_avg["Đúng hướng (%)"], 1),
        **prov_val_rmse,
    }


def get_hyperparameter_grid(mode: str = "quick") -> Dict[str, Dict[str, List[Any]]]:
    """Tập siêu tham số được lấy chính xác theo giao diện Bảng điều khiển siêu tham số:
    
    1. Batch Size: [16, 32, 64, 128, 256]
    2. Learning Rate: [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
    3. Hidden Size: [16, 32, 48, 64, 128]
    4. seq_len: [5, 10, 15, 20]
    5. Số lớp LSTM: [1, 2]
    6. Hàm mất mát:
       - 'huber_1.0': Huber (beta=1.0)
       - 'mse': MSE Loss
       - 'smooth_l1_0.01': Smooth L1 (beta=0.01)
    7. Direction Penalty lambda: [0.0, 0.02, 0.05, 0.1]
    8. Residual Learning: [True, False]
    9. Temporal Attention (LSTM): [True, False]
    """
    if mode == "quick":
        # Chế độ kiểm tra nhanh: tập con tiêu biểu
        return {
            "DeepVAR": {
                "seq_len": [5, 10],
                "hidden": [32, 64],
                "lr": [1e-3],
                "batch_size": [32],
                "dropout": [0.1],
                "weight_decay": [1e-5],
            },
            "LSTM": {
                "seq_len": [5, 10],
                "hidden": [32, 64],
                "layers": [1],
                "lr": [1e-3],
                "batch_size": [32],
                "loss_spec": [("huber", 1.0)],
                "dir_weight": [0.0, 0.05],
                "use_residual": [True],
                "use_attention": [False, True],
            },
            "DeepVARX": {
                "seq_len": [5, 10],
                "hidden": [32, 64],
                "layers": [1],
                "lr": [1e-3],
                "batch_size": [32],
                "dir_weight": [0.05],
                "loss_spec": [("huber", 1.0)],
                "use_residual": [True],
            },
        }

    if mode == "smart":
        # Tìm kiếm tập trung vào các siêu tham số ảnh hưởng mạnh nhất (cân bằng giữa độ phủ và thời gian chạy)
        return {
            "DeepVAR": {
                "seq_len": [5, 10, 15],
                "hidden": [32, 64, 128],
                "lr": [3e-4, 1e-3],
                "batch_size": [16, 32, 64],
                "dropout": [0.1],
                "weight_decay": [1e-5],
            },
            "LSTM": {
                "seq_len": [5, 10, 15],
                "hidden": [32, 64, 128],
                "layers": [1],
                "lr": [3e-4, 1e-3],
                "batch_size": [16, 32, 64],
                "loss_spec": [("huber", 1.0)],
                "dir_weight": [0.0, 0.05],
                "use_residual": [True],
                "use_attention": [False, True],
            },
            "DeepVARX": {
                "seq_len": [5, 10, 15],
                "hidden": [32, 64, 128],
                "layers": [1],
                "lr": [3e-4, 1e-3],
                "batch_size": [16, 32, 64],
                "loss_spec": [("huber", 1.0), ("mse", 1.0)],
                "dir_weight": [0.0, 0.05],
                "use_residual": [True],
            },
        }

    # Chế độ FULL: Toàn bộ phạm vi siêu tham số của giao diện
    return {
        "DeepVAR": {
            "seq_len": [5, 10, 15],
            "hidden": [32, 48, 64, 128],
            "lr": [3e-4, 5e-4, 1e-3],
            "batch_size": [16, 32, 64],
            "dropout": [0.0, 0.1],
            "weight_decay": [1e-5],
        },
        "LSTM": {
            "seq_len": [5, 10, 15],
            "hidden": [32, 48, 64, 128],
            "layers": [1, 2],
            "lr": [3e-4, 5e-4, 1e-3],
            "batch_size": [16, 32, 64],
            "loss_spec": [("huber", 1.0), ("mse", 1.0), ("huber", 0.01)],
            "dir_weight": [0.0, 0.05],
            "use_residual": [True, False],
            "use_attention": [False, True],
        },
        "DeepVARX": {
            "seq_len": [5, 10, 15],
            "hidden": [32, 48, 64, 128],
            "layers": [1, 2],
            "lr": [3e-4, 5e-4, 1e-3],
            "batch_size": [16, 32, 64],
            "loss_spec": [("huber", 1.0), ("mse", 1.0), ("huber", 0.01)],
            "dir_weight": [0.0, 0.02, 0.05, 0.1],
            "use_residual": [True, False],
        },
    }


def run_grid_search(
    data_path: str | Path = _CURR_DIR / "data" / "data_daily_clean_v2.csv",
    output_csv: str | Path = _CURR_DIR / "data" / "results" / "hyperparameter_grid_search_results.csv",
    target_models: List[str] | None = None,
    endog_cols: List[str] | None = None,
    exog_cols: List[str] | None = None,
    mode: str = "smart",
    max_epochs: int = 30,
    patience: int = 10,
    device: str = "cpu",
    seed: int = 42,
    split_mode: str = "pages",
) -> pd.DataFrame:
    """Chạy toàn bộ quy trình Grid Search cho 3 mô hình học sâu và lưu kết quả vào CSV."""
    df, n_train, n_val, n_test, active_endog, active_exog = load_dataset(data_path, endog_cols, exog_cols, split_mode=split_mode)
    print("=" * 85)
    print("QUY TRÌNH GRID SEARCH TÌM SIÊU THAM SỐ 3 MÔ HÌNH HỌC SÂU (CHUẨN GIAO DIỆN)")
    print("=" * 85)
    print(f"File dữ liệu   : {data_path}")
    print(f"Tổng quan sát  : {len(df)} dòng")
    print(f"Phân chia mẫu  : Train={n_train}, Val={n_val}, Test={n_test} (mode={split_mode})")
    print(f"Biến nội sinh Y ({len(active_endog)}): {', '.join(active_endog)}")
    print(f"Biến ngoại sinh X ({len(active_exog)}): {', '.join(active_exog)}")
    print(f"Chế độ tìm kiếm: {mode.upper()} | Thiết bị: {device.upper()} | Max Epochs: {max_epochs} | Patience: {patience}")

    # Benchmark Naive
    naive = compute_naive_benchmark(df, n_train, n_val, active_endog)
    print(f"Naive Benchmark: Val RMSE = {naive['naive_val_rmse']:,.2f} | Test RMSE = {naive['naive_test_rmse']:,.2f} VNĐ/kg")
    print("-" * 85)

    out_file = Path(output_csv)
    if not out_file.is_absolute():
        out_file = _CURR_DIR / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    grid_defs = get_hyperparameter_grid(mode)
    if target_models is None:
        target_models = ["DeepVAR", "LSTM", "DeepVARX"]
    else:
        target_models = [MODEL_CANONICAL.get(m.lower(), m) for m in target_models]

    windows_cache: Dict[int, Dict[str, Any]] = {}
    feature_cols = active_endog + active_exog

    results_rows: List[Dict[str, Any]] = []

    for model_name in target_models:
        if model_name not in grid_defs:
            print(f"Bỏ qua mô hình không nhận diện: {model_name}")
            continue

        raw_grid = grid_defs[model_name]
        param_names = list(raw_grid.keys())
        combinations = list(itertools.product(*raw_grid.values()))
        total_runs = len(combinations)
        print(f"\n[BẮT ĐẦU] Mô hình: {model_name} - Tổng {total_runs} cấu hình siêu tham số")

        for idx_run, comb in enumerate(combinations, 1):
            hp = dict(zip(param_names, comb))
            hp["epochs"] = max_epochs
            hp["patience"] = patience

            seq_len = int(hp["seq_len"])
            hidden_val = hp["hidden"]
            lr = float(hp["lr"])
            bs = int(hp["batch_size"])

            # Xử lý thông số Loss & Beta
            if model_name == "DeepVAR":
                loss_type, huber_beta = "mse", np.nan
                loss_display = "MSE"
                dw_info = ""
            else:
                loss_spec = hp.get("loss_spec", ("huber", 1.0))
                loss_type, huber_beta = loss_spec
                loss_display = f"{loss_type}(β={huber_beta})" if loss_type != "mse" else "MSE"
                dw_val = hp.get("dir_weight", 0.05 if model_name == "DeepVARX" else 0.0)
                dw_info = f", dw={dw_val}"
            att_info = f", Attn={'Bật' if hp.get('use_attention') else 'Tắt'}" if model_name == "LSTM" else ""
            res_info = f", Res={'Bật' if hp.get('use_residual', True) else 'Tắt'}" if model_name != "DeepVAR" else ""

            print(f"  ({idx_run}/{total_runs}) {model_name} | seq={seq_len}, h={hidden_val}, lr={lr:g}, bs={bs}, {loss_display}{dw_info}{res_info}{att_info}...", end="", flush=True)

            try:
                if model_name == "DeepVAR":
                    run_res = evaluate_deepvar(df, n_train, n_val, active_endog, hp, device=device, seed=seed)
                    dir_w = 0.0
                    use_res = True
                    use_att = False
                else:
                    if seq_len not in windows_cache:
                        windows_cache[seq_len] = create_sliding_windows(df, active_endog, active_exog, seq_len, n_train, n_val)
                    w = windows_cache[seq_len]

                    n_in = len(feature_cols)
                    n_out = len(active_endog)
                    layers = int(hp.get("layers", 1))
                    use_res = bool(hp.get("use_residual", True))
                    use_att = bool(hp.get("use_attention", False))

                    if model_name == "LSTM":
                        Cls = LSTMAttentionForecaster if use_att else LSTMForecaster
                        net = Cls(
                            n_in=n_in,
                            hidden=int(hidden_val),
                            layers=layers,
                            dropout=0.1 if layers > 1 else 0.0,
                            n_prov=0,
                            emb_dim=0,
                            residual=use_res,
                            n_out=n_out,
                        )
                        dir_w = float(hp.get("dir_weight", 0.0))
                    else:  # DeepVARX
                        net = DeepVARX(
                            n_in=n_in,
                            hidden=int(hidden_val),
                            layers=layers,
                            dropout=0.1 if layers > 1 else 0.0,
                            n_prov=0,
                            emb_dim=0,
                            residual=use_res,
                            n_out=n_out,
                        )
                        dir_w = float(hp.get("dir_weight", 0.05))

                    run_res = train_eval_neural_net(
                        net,
                        w,
                        epochs=max_epochs,
                        batch_size=bs,
                        lr=lr,
                        patience=patience,
                        endog_cols=active_endog,
                        dir_weight=dir_w,
                        loss_type=loss_type,
                        huber_beta=huber_beta,
                        device=device,
                        seed=seed,
                    )

                val_rmse = run_res["val_rmse"]
                beat_naive = val_rmse < naive["naive_val_rmse"]
                diff_naive = val_rmse - naive["naive_val_rmse"]

                row = {
                    "model": model_name,
                    "n_endog": len(active_endog),
                    "endog_names": "+".join(active_endog),
                    "seq_len": seq_len,
                    "hidden": hidden_val,
                    "layers": hp.get("layers", 1),
                    "lr": lr,
                    "batch_size": bs,
                    "dropout": hp.get("dropout", 0.1 if hp.get("layers", 1) > 1 or model_name == "DeepVAR" else 0.0),
                    "weight_decay": hp.get("weight_decay", 1e-5 if model_name == "DeepVAR" else 1e-4),
                    "loss_type": loss_type,
                    "huber_beta": huber_beta,
                    "dir_weight": dir_w,
                    "use_residual": use_res,
                    "use_attention": use_att,
                    "best_epoch": run_res["best_epoch"],
                    "epochs_run": run_res["epochs_run"],
                    "train_time_sec": run_res["train_time_sec"],
                    "val_loss": run_res["val_loss"],
                    "val_rmse": run_res["val_rmse"],
                    "val_mae": run_res["val_mae"],
                    "val_mape": run_res["val_mape"],
                    "val_smape": run_res["val_smape"],
                    "val_r2": run_res["val_r2"],
                    "val_da": run_res["val_da"],
                    "test_rmse": run_res["test_rmse"],
                    "test_mae": run_res["test_mae"],
                    "test_mape": run_res["test_mape"],
                    "test_smape": run_res["test_smape"],
                    "test_r2": run_res["test_r2"],
                    "test_da": run_res["test_da"],
                    "naive_val_rmse": naive["naive_val_rmse"],
                    "naive_test_rmse": naive["naive_test_rmse"],
                    "val_rmse_diff_naive": round(diff_naive, 2),
                    "beat_naive_val": beat_naive,
                    "beat_naive_test": run_res["test_rmse"] < naive["naive_test_rmse"],
                }
                for ecol in active_endog:
                    row[f"val_rmse_{ecol}"] = run_res.get(f"val_rmse_{ecol}")

                results_rows.append(row)

                status_naive = "THẮNG Naive" if beat_naive else "Thua Naive"
                print(f" Xong! Val RMSE: {val_rmse:,.1f} | Test RMSE: {run_res['test_rmse']:,.1f} | DA: {run_res['test_da']:.1f}% | {status_naive} | ({run_res['train_time_sec']}s)")

            except Exception as e:
                print(f" LỖI: {e}")

        if results_rows:
            pd.DataFrame(results_rows).to_csv(out_file, index=False, encoding="utf-8-sig")

    result_df = pd.DataFrame(results_rows)
    if not result_df.empty:
        result_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 85)
    print("HOÀN THÀNH TOÀN BỘ GRID SEARCH!")
    print(f"Kết quả tổng hợp đã xuất ra file: {out_file.resolve()}")
    print("=" * 85)

    print("\nBẢNG TOP 1 CẤU HÌNH TỐI ƯU CỦA TỪNG MÔ HÌNH (CHỌN THEO VALIDATION RMSE — KHÔNG RÒ RỈ TEST):")
    summary_list = []
    best_rows = []
    for m in target_models:
        sub = result_df[result_df["model"] == m]
        if not sub.empty:
            best = sub.sort_values(["val_rmse", "val_mae"]).iloc[0]
            best_rows.append(best)
            summary_list.append({
                "Mô hình": m,
                "seq_len": best["seq_len"],
                "hidden": best["hidden"],
                "lr": best["lr"],
                "batch_size": best["batch_size"],
                "Loss": "MSE" if best["loss_type"] == "mse" else f"{best['loss_type']}(β={best['huber_beta']})",
                "dir_w": best["dir_weight"],
                "Attn": "Bật" if best["use_attention"] else "Tắt",
                "best_ep": best["best_epoch"],
                "Val RMSE": f"{best['val_rmse']:,.2f}",
                "Test RMSE": f"{best['test_rmse']:,.2f}",
                "Test MAE": f"{best['test_mae']:,.2f}",
                "Test DA": f"{best['test_da']:.1f}%",
                "Thắng Naive Test": "Có" if best["beat_naive_test"] else "Không",
            })
    if summary_list:
        print(pd.DataFrame(summary_list).to_string(index=False))
        best_file = out_file.with_name(out_file.stem + "_best.csv")
        pd.DataFrame(best_rows).to_csv(best_file, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu Top 1 từng mô hình ra: {best_file.resolve()}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid Search siêu tham số cho 3 mô hình học sâu.")
    parser.add_argument("--data", type=str, default=str(_CURR_DIR / "data" / "data_daily_clean_v2.csv"), help="Đường dẫn file dữ liệu CSV")
    parser.add_argument("--output", type=str, default=str(_CURR_DIR / "data" / "results" / "hyperparameter_grid_search_results.csv"), help="File CSV xuất kết quả")
    parser.add_argument("--mode", type=str, choices=["quick", "smart", "full"], default="smart", help="Chế độ tìm kiếm (quick: test nhanh; smart: tìm tập trung tối ưu; full: vét cạn đầy đủ)")
    parser.add_argument("--models", type=str, default="all", help="Danh sách mô hình cần chạy (vd: 'all', hoặc 'deepvar,lstm,deepvarx')")
    parser.add_argument("--endog", type=str, default="all", help="Danh sách biến nội sinh ('all' cho 4 tỉnh, hoặc vd: 'price_dak_lak')")
    parser.add_argument("--exog", type=str, default="all", help="Danh sách biến ngoại sinh ('all' cho 10 biến chuẩn, hoặc truyền danh sách cách nhau bởi dấu phẩy)")
    parser.add_argument("--split-mode", type=str, choices=["pages", "70_15_15"], default="pages", help="Cách chia dữ liệu ('pages': 70% train+val [85/15] & 30% test giống Trang 1–5; '70_15_15': 70/15/15)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch tối đa cho mỗi cấu hình")
    parser.add_argument("--patience", type=int, default=10, help="Số epoch dừng sớm (Early stopping patience)")
    parser.add_argument("--device", type=str, default="auto", help="Thiết bị ('cpu', 'cuda', hoặc 'auto')")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")

    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu"
    chosen_models = ["DeepVAR", "LSTM", "DeepVARX"] if args.models == "all" else [m.strip() for m in args.models.split(",")]
    chosen_endog = None if args.endog == "all" else [c.strip() for c in args.endog.split(",")]
    chosen_exog = None if args.exog == "all" else [c.strip() for c in args.exog.split(",")]

    run_grid_search(
        data_path=args.data,
        output_csv=args.output,
        target_models=chosen_models,
        endog_cols=chosen_endog,
        exog_cols=chosen_exog,
        mode=args.mode,
        max_epochs=args.epochs,
        patience=args.patience,
        device=dev,
        seed=args.seed,
        split_mode=args.split_mode,
    )

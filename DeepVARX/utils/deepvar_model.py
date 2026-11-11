# -*- coding: utf-8 -*-
"""
========================================================================================
MÔ-ĐUN MÔ HÌNH DEEPVAR XÁC SUẤT ĐA BIẾN (LOW-RANK GAUSSIAN PROCESS PARAMETRIZATION)
Ứng dụng: Dự báo giá cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
========================================================================================
Tổng hợp cơ sở lý thuyết từ các tài liệu nghiên cứu và thực nghiệm LSTM:
  1. Salinas et al. (NeurIPS 2019) - "High-Dimensional Multivariate Forecasting with
     Low-Rank Gaussian Copula Processes" (DeepVAR / Vec-LSTM-LowRank):
     Tham số hóa phân phối chuẩn đa biến đồng thời p(y_t | h_t) = N(mu_t, Sigma_t) với
     ma trận hiệp phương sai hạng thấp + đường chéo: Sigma_t = diag(d_t) + V_t V_t^T,
     nắm bắt biến động phương sai theo thời gian và tương quan không gian giữa các tỉnh.
  2. Salinas et al. (IJF 2020) - DeepAR: Mạng hồi quy tự hồi quy sinh mẫu Monte Carlo
     và kiểm định độ hiệu chỉnh khoảng tin cậy bằng NLL, CRPS, CRPS-Sum.
  3. Seitzer et al. (ICLR 2022) - Beta-NLL: Trọng số hóa log-likelihood theo phương sai
     (beta = 0.5) kết hợp tách nhánh Mean (LSTM_mu) và nhánh Covariance (LSTM_cov) nhằm
     triệt tiêu xung đột gradient và hiện tượng co sập phương sai sớm trên chuỗi tài chính.
  4. Beck, Dovern & Vogl (2025) - "Mind the naive forecast!" & Kadir et al. (2026):
     Chuỗi giá I(1) phải dừng hóa về log-return r_t = ln(P_t/P_{t-1}), neo giá khôi phục
     trên Naive P_{t-1}, kiến trúc 1 lớp LSTM (64 units Mean, 32 units Cov), Adam/AdamW
     lr=1e-3, batch=16, dùng CRPS làm tiêu chí Early Stopping và hiệu chỉnh quy mô
     phương sai tau trên Validation (không chạm Test).
  5. Mai et al. (2018) & O'Brien (2007) / Vatcheva et al. (2016): Truyền dẫn giá T+1
     từ sàn London và sàng lọc đa cộng tuyến VIF < 10.

Ba phiên bản thực nghiệm đồng bộ với hệ thống LSTM:
  - DeepVAR 1 (Baseline)       : Chỉ dùng giá cà phê quá khứ (3 tỉnh).
  - DeepVAR 2 (Multivariate)   : Giá quá khứ + Thời tiết cơ bản + Kinh tế cơ bản (11 biến).
  - DeepVAR 3 (Full Exogenous) : Giá quá khứ + Toàn bộ biến ngoại sinh phù hợp (20 biến).
========================================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from torch.distributions import LowRankMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

TARGET_PROVINCES: List[str] = ["dak_lak", "lam_dong", "gia_lai"]

CORE_WEATHER_COLS: List[str] = ["temp_mean", "precip_nasa", "humidity_air"]
EXT_WEATHER_COLS: List[str] = ["d_soil_moisture", "solar_radiation", "wind_speed_2m"]
CORE_ECON_LEVEL_COLS: List[str] = [
    "london_price_usd_ton",
    "arabica_price_usd_lb",
    "brent_price_usd_barrel",
    "dxy_index",
    "usd_vnd_rate",
]
EXT_ECON_LEVEL_COLS: List[str] = ["usd_brl_rate"]
CALENDAR_FRICTION_COLS: List[str] = [
    "sin_day_of_year",
    "cos_day_of_year",
    "is_harvest_season_vn",
    "is_harvest_season_brazil",
    "exog_stale_days",
]


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên trên Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CoffeeDataBundle:
    """Dữ liệu đồng bộ theo ngày, chuỗi dừng và 3 tập đặc trưng cho DeepVAR 1 / 2 / 3."""

    raw_df: pd.DataFrame
    price_df: pd.DataFrame
    ret_df: pd.DataFrame
    dates: pd.DatetimeIndex
    train_end: int
    val_end: int
    feature_sets: Dict[str, pd.DataFrame]
    exog_unscreened: pd.DataFrame


def load_and_prepare_coffee_data(
    csv_path: str = "data_daily_panel_clean_v2.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> CoffeeDataBundle:
    """
    Nạp dữ liệu panel và xây dựng chuỗi dừng I(0) + 3 tập biến đầu vào (lag >= 1).
    Bảo đảm đồng bộ hoàn toàn ngày chia Train / Val / Test với hệ thống LSTM và VAR/VARX.
    """
    raw_df = pd.read_csv(csv_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    price_raw = (
        raw_df[raw_df["province_slug"].isin(TARGET_PROVINCES)]
        .pivot(index="date", columns="province_slug", values="price_vnd_kg")[TARGET_PROVINCES]
        .dropna()
        .sort_index()
    )

    ret_df = np.log(price_raw).diff().dropna()
    dates = ret_df.index
    price_df = price_raw.loc[dates]

    all_econ_level_cols = CORE_ECON_LEVEL_COLS + EXT_ECON_LEVEL_COLS
    econ_level_df = raw_df.groupby("date")[all_econ_level_cols].mean().loc[dates]
    econ_ret_df = np.log(econ_level_df).diff().fillna(0.0)
    econ_ret_df.columns = [
        "ret_london",
        "ret_arabica",
        "ret_brent",
        "ret_dxy",
        "ret_usd_vnd",
        "ret_usd_brl",
    ]

    raw_weather_cols = [
        "temp_mean",
        "temp_max",
        "temp_min",
        "precip_nasa",
        "soil_moisture_top",
        "humidity_air",
        "solar_radiation",
        "wind_speed_2m",
    ]
    weather_mean_df = raw_df.groupby("date")[raw_weather_cols].mean().loc[dates]
    weather_mean_df["d_soil_moisture"] = weather_mean_df["soil_moisture_top"].diff().fillna(0.0)

    cal_df = raw_df.groupby("date")[CALENDAR_FRICTION_COLS].mean().loc[dates]

    x_v1 = ret_df.copy()
    x_v2 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS],
            econ_ret_df[["ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd"]],
        ],
        axis=1,
    )
    x_v3 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS + EXT_WEATHER_COLS],
            econ_ret_df,
            cal_df,
        ],
        axis=1,
    )
    exog_unscreened = pd.concat([weather_mean_df[raw_weather_cols], econ_ret_df, cal_df], axis=1)

    n_total = len(ret_df)
    train_end = int(n_total * train_ratio)
    val_end = int(n_total * (train_ratio + val_ratio))

    return CoffeeDataBundle(
        raw_df=raw_df,
        price_df=price_df,
        ret_df=ret_df,
        dates=dates,
        train_end=train_end,
        val_end=val_end,
        feature_sets={
            "DeepVAR 1": x_v1,
            "DeepVAR 2": x_v2,
            "DeepVAR 3": x_v3,
        },
        exog_unscreened=exog_unscreened,
    )


def compute_diagnostics(bundle: CoffeeDataBundle) -> Dict[str, pd.DataFrame]:
    """Chẩn đoán dịch chuyển miền giá, tính dừng ADF, VIF và ma trận tương quan liên tỉnh."""
    tr_end = bundle.train_end
    va_end = bundle.val_end

    split_rows = []
    for split_name, s, e in [
        ("Train", 0, tr_end),
        ("Validation", tr_end, va_end),
        ("Test", va_end, len(bundle.price_df)),
    ]:
        p_slice = bundle.price_df.iloc[s:e]
        r_slice = bundle.ret_df.iloc[s:e]
        for prov in TARGET_PROVINCES:
            split_rows.append(
                {
                    "Split": split_name,
                    "Province": prov,
                    "Days": len(p_slice),
                    "Price_Min": float(p_slice[prov].min()),
                    "Price_Max": float(p_slice[prov].max()),
                    "Price_Mean": float(p_slice[prov].mean()),
                    "Return_Std": float(r_slice[prov].std()),
                }
            )
    split_stats_df = pd.DataFrame(split_rows)

    # Tương quan liên tỉnh trên Train và Test (cơ sở sử dụng hiệp phương sai Low-Rank)
    emp_corr_train = bundle.ret_df.iloc[:tr_end].corr().round(4)
    emp_corr_test = bundle.ret_df.iloc[va_end:].corr().round(4)

    # Kiểm định ADF trên Train
    exog_screened = bundle.feature_sets["DeepVAR 3"].drop(columns=TARGET_PROVINCES).iloc[:tr_end]
    adf_rows = []
    for col in ["dak_lak", "lam_dong", "gia_lai"]:
        st, pv, *_ = adfuller(bundle.ret_df[col].iloc[:tr_end], autolag="AIC")
        adf_rows.append({"Variable": f"ret_{col}", "ADF_Stat": round(float(st), 3), "p_value": round(float(pv), 4), "Stationary_5pct": bool(pv < 0.05)})
    for col in exog_screened.columns:
        st, pv, *_ = adfuller(exog_screened[col], autolag="AIC")
        adf_rows.append({"Variable": col, "ADF_Stat": round(float(st), 3), "p_value": round(float(pv), 4), "Stationary_5pct": bool(pv < 0.05)})
    adf_df = pd.DataFrame(adf_rows)

    # Kiểm định VIF
    mat = np.column_stack([np.ones(len(exog_screened)), exog_screened.values])
    vif_df = pd.DataFrame(
        [
            {"Feature": c, "VIF_After_Screening": round(float(variance_inflation_factor(mat, i + 1)), 2)}
            for i, c in enumerate(exog_screened.columns)
        ]
    )

    return {
        "split_stats": split_stats_df,
        "emp_corr_train": emp_corr_train,
        "emp_corr_test": emp_corr_test,
        "adf": adf_df,
        "vif": vif_df,
    }


class CoffeeDeepVAR(nn.Module):
    """
    Mô hình DeepVAR xác suất đa biến (Low-Rank Gaussian Process Parametrization):
      - Nhánh kỳ vọng (Mean Encoder): [Dropout] -> LSTM_mu(1L, 64u) -> Linear -> mu_t in R^K
      - Nhánh hiệp phương sai (Cov Encoder): [Dropout] -> LSTM_cov(1L, 32u) ->
          * diag_head   -> d_t = softplus(z_d) + min_std^2 in R^K_{>0}
          * factor_head -> V_t in R^{K x rank} (nếu rank > 0)
      - Phân phối đồng thời: y_t | h_t ~ N(mu_t, diag(d_t) + V_t V_t^T).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        cov_hidden_dim: int = 32,
        num_layers: int = 1,
        out_dim: int = 3,
        rank: int = 2,
        dropout: float = 0.0,
        in_dropout: float = 0.0,
        min_std: float = 0.20,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.rank = rank
        self.min_std = min_std

        self.in_drop = nn.Dropout(in_dropout) if in_dropout > 0.0 else nn.Identity()

        self.lstm_mu = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop_mu = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.mu_head = nn.Linear(hidden_dim, out_dim)

        self.lstm_cov = nn.LSTM(
            input_size=in_dim,
            hidden_size=cov_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.drop_cov = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.diag_head = nn.Linear(cov_hidden_dim, out_dim)
        self.factor_head = nn.Linear(cov_hidden_dim, out_dim * rank) if rank > 0 else None

        nn.init.constant_(self.diag_head.bias, 0.0)
        if self.factor_head is not None:
            nn.init.xavier_uniform_(self.factor_head.weight, gain=0.2)
            nn.init.zeros_(self.factor_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        xd = self.in_drop(x)

        out_mu, _ = self.lstm_mu(xd)
        h_mu = self.drop_mu(out_mu[:, -1, :])
        mu = self.mu_head(h_mu)

        out_cov, _ = self.lstm_cov(xd)
        h_cov = self.drop_cov(out_cov[:, -1, :])
        d = F.softplus(self.diag_head(h_cov)) + (self.min_std ** 2)
        V = self.factor_head(h_cov).view(-1, self.out_dim, self.rank) if self.rank > 0 else None
        return mu, d, V

    def joint_covariance(self, d: torch.Tensor, V: Optional[torch.Tensor]) -> torch.Tensor:
        """Tái tạo ma trận hiệp phương sai Sigma_t = diag(d_t) + V_t V_t^T kích thước (B, K, K)."""
        cov = torch.diag_embed(d)
        if V is not None:
            cov = cov + torch.bmm(V, V.transpose(1, 2))
        return cov

    def log_prob(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        d: torch.Tensor,
        V: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Tính log-likelihood đa biến chính xác cho cả Low-Rank+Diag và Diagonal."""
        if self.rank > 0 and V is not None:
            dist = LowRankMultivariateNormal(loc=mu, cov_factor=V, cov_diag=d)
            return dist.log_prob(y)
        return -0.5 * (math.log(2.0 * math.pi) + torch.log(d) + ((y - mu) ** 2) / d).sum(dim=-1)

    def beta_nll_loss(self, x: torch.Tensor, y: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        """Hàm mất mát Beta-NLL (Seitzer et al., 2022) cân bằng học kỳ vọng và phương sai."""
        mu, d, V = self.forward(x)
        nll_vec = -self.log_prob(y, mu, d, V)
        if beta > 0.0:
            v_factor = torch.zeros_like(d) if V is None else (V ** 2).sum(dim=-1)
            marg_var = d + v_factor
            weight = (marg_var.mean(dim=-1).detach()) ** beta
            return (weight * nll_vec).mean()
        return nll_vec.mean()


def gaussian_crps_torch(y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """CRPS giải tích chính xác cho phân phối chuẩn biên N(mu, sigma^2)."""
    s_safe = torch.clamp(sigma, min=1e-8)
    z = (y_true - mu) / s_safe
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return s_safe * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def gaussian_crps_numpy(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Phiên bản NumPy của CRPS giải tích từng phần tử (dùng cho kiểm định DM xác suất)."""
    with torch.no_grad():
        res = gaussian_crps_torch(
            torch.tensor(np.array(y_true, dtype=np.float64, copy=True)),
            torch.tensor(np.array(mu, dtype=np.float64, copy=True)),
            torch.tensor(np.array(sigma, dtype=np.float64, copy=True)),
        )
    return res.numpy()


def build_causal_windows(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    seq_len: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Tạo cửa sổ đầu vào [t - seq_len, ..., t - 1] cho mục tiêu t."""
    eff_start = max(start_idx, seq_len)
    xs, ys, indices = [], [], []
    for t in range(eff_start, end_idx):
        xs.append(x_scaled[t - seq_len : t])
        ys.append(y_scaled[t])
        indices.append(t)
    return (
        torch.tensor(np.asarray(xs), dtype=torch.float32),
        torch.tensor(np.asarray(ys), dtype=torch.float32),
        np.asarray(indices, dtype=np.int64),
    )


def fit_single_deepvar_seed(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    hidden_dim: int = 64,
    cov_hidden_dim: int = 32,
    num_layers: int = 1,
    rank: int = 2,
    dropout: float = 0.1,
    in_dropout: float = 0.0,
    min_std: float = 0.20,
    beta_nll: float = 0.5,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    n_mc_samples: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Huấn luyện 1 seed DeepVAR xác suất đa biến:
      - Chuẩn hóa X, Y chỉ fit trên Train.
      - Huấn luyện bằng Beta-NLL trên LowRankMultivariateNormal.
      - Early Stopping theo Validation CRPS (strictly proper scoring rule).
      - Hiệu chỉnh hệ số quy mô phương sai tau trên Validation (không chạm Test):
            Sigma_cal = tau^2 * Sigma_raw.
      - Sinh 1000 mẫu Monte Carlo trên phân phối đa biến để tính khoảng tin cậy giá VND.
    """
    set_seed(seed)
    X_arr = X_df.values.astype(np.float32)
    Y_arr = Y_df.values.astype(np.float32)

    x_scaler = StandardScaler().fit(X_arr[:train_end])
    y_scaler = StandardScaler().fit(Y_arr[:train_end])
    X_s = x_scaler.transform(X_arr).astype(np.float32)
    Y_s = y_scaler.transform(Y_arr).astype(np.float32)

    X_tr, Y_tr, _ = build_causal_windows(X_s, Y_s, seq_len, seq_len, train_end)
    X_va, Y_va, va_idx = build_causal_windows(X_s, Y_s, seq_len, train_end, val_end)
    X_te, Y_te, te_idx = build_causal_windows(X_s, Y_s, seq_len, val_end, len(Y_arr))

    K = Y_df.shape[1]
    model = CoffeeDeepVAR(
        in_dim=X_df.shape[1],
        hidden_dim=hidden_dim,
        cov_hidden_dim=cov_hidden_dim,
        num_layers=num_layers,
        out_dim=K,
        rank=rank,
        dropout=dropout,
        in_dropout=in_dropout,
        min_std=min_std,
    )
    OptCls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
    opt = OptCls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    best_crps = float("inf")
    best_state = None
    best_epoch = 0
    pat_cnt = 0
    train_losses, val_crps_hist, val_nll_hist = [], [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for bx, by in loader:
            opt.zero_grad()
            loss = model.beta_nll_loss(bx, by, beta=beta_nll)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ep_loss += loss.item()
            nb += 1
        train_losses.append(ep_loss / max(nb, 1))

        model.eval()
        with torch.no_grad():
            mu_v, d_v, V_v = model(X_va)
            v_factor = torch.zeros_like(d_v) if V_v is None else (V_v ** 2).sum(dim=-1)
            marg_std = torch.sqrt(d_v + v_factor)
            v_crps = float(gaussian_crps_torch(Y_va, mu_v, marg_std).mean().item())
            v_nll = float((-model.log_prob(Y_va, mu_v, d_v, V_v)).mean().item())
        val_crps_hist.append(v_crps)
        val_nll_hist.append(v_nll)

        if v_crps < best_crps:
            best_crps = v_crps
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        # 1. Tính hệ số hiệu chỉnh phương sai tau trên Validation (closed-form MLE)
        mu_v, d_v, V_v = model(X_va)
        cov_v = model.joint_covariance(d_v, V_v)
        diff_v = (Y_va - mu_v).unsqueeze(-1)
        quad_v = torch.bmm(diff_v.transpose(1, 2), torch.linalg.solve(cov_v, diff_v)).squeeze()
        tau = float(torch.sqrt(torch.mean(quad_v) / K).item())
        tau = max(tau, 1.0)  # chỉ mở rộng hoặc giữ nguyên độ bất định, không thu hẹp quá mức

        # 2. Dự báo trên tập Test
        mu_t, d_t, V_t = model(X_te)
        cov_t_raw = model.joint_covariance(d_t, V_t)
        d_t_cal = d_t * (tau ** 2)
        V_t_cal = V_t * tau if V_t is not None else None
        cov_t_cal = cov_t_raw * (tau ** 2)

        test_nll_raw_s = float((-model.log_prob(Y_te, mu_t, d_t, V_t)).mean().item())
        test_nll_cal_s = float((-model.log_prob(Y_te, mu_t, d_t_cal, V_t_cal)).mean().item())

        # Sinh mẫu Monte Carlo trên phân phối đã hiệu chỉnh (Salinas et al., 2019)
        torch.manual_seed(seed + 999)
        if rank > 0 and V_t_cal is not None:
            dist_te_cal = LowRankMultivariateNormal(loc=mu_t, cov_factor=V_t_cal, cov_diag=d_t_cal)
            mc_s = dist_te_cal.sample((n_mc_samples,))  # (M, N_te, K)
        else:
            eps_mc = torch.randn(n_mc_samples, mu_t.size(0), K)
            mc_s = mu_t.unsqueeze(0) + eps_mc * torch.sqrt(d_t_cal).unsqueeze(0)

    # Biến đổi ngược về không gian log-return gốc (VND log-diff)
    scale = y_scaler.scale_  # (K,)
    mean_y = y_scaler.mean_
    mu_r = y_scaler.inverse_transform(mu_t.numpy())
    cov_scale = np.outer(scale, scale)[None, :, :]
    cov_r_raw = cov_t_raw.numpy() * cov_scale
    cov_r_cal = cov_t_cal.numpy() * cov_scale
    std_r_raw = np.sqrt(np.diagonal(cov_r_raw, axis1=1, axis2=2))
    std_r_cal = np.sqrt(np.diagonal(cov_r_cal, axis1=1, axis2=2))

    # NLL trên thang log-return gốc = NLL_scaled + sum(log(scale))
    log_jac = float(np.sum(np.log(scale)))
    test_nll_raw_ret = test_nll_raw_s + log_jac
    test_nll_cal_ret = test_nll_cal_s + log_jac

    # Mẫu Monte Carlo trên thang giá VND/kg
    mc_r = mc_s.numpy() * scale[None, None, :] + mean_y[None, None, :]  # (M, N_te, K)
    test_prev_p = price_df.shift(1).iloc[te_idx].values
    test_true_p = price_df.iloc[te_idx].values
    test_true_r = Y_df.iloc[te_idx].values

    mc_prices = test_prev_p[None, :, :] * np.exp(mc_r)  # (M, N_te, K)
    pred_p_median = test_prev_p * np.exp(mu_r)
    pred_p_q025 = np.quantile(mc_prices, 0.025, axis=0)
    pred_p_q975 = np.quantile(mc_prices, 0.975, axis=0)
    pred_p_q10 = np.quantile(mc_prices, 0.10, axis=0)
    pred_p_q90 = np.quantile(mc_prices, 0.90, axis=0)

    # Tính ma trận tương quan dự báo trung bình trên Test từ Sigma_r_cal
    denom = np.sqrt(np.einsum("tii->ti", cov_r_cal)[:, :, None] * np.einsum("tii->ti", cov_r_cal)[:, None, :])
    pred_corr_mats = cov_r_cal / np.maximum(denom, 1e-12)
    mean_pred_corr = np.mean(pred_corr_mats, axis=0)

    # Validation RMSE để kiểm chứng
    val_prev_p = price_df.shift(1).iloc[va_idx].values
    val_true_p = price_df.iloc[va_idx].values
    val_mu_r = y_scaler.inverse_transform(mu_v.numpy())
    val_pred_p = val_prev_p * np.exp(val_mu_r)
    val_rmse_p = float(np.sqrt(np.mean((val_true_p - val_pred_p) ** 2)))

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "tau_cal": tau,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_crps_hist": val_crps_hist,
        "val_nll_hist": val_nll_hist,
        "val_rmse_price": val_rmse_p,
        "X_te_tensor": X_te,
        "Y_te_tensor": Y_te,
        "te_idx": te_idx,
        "test_pred_ret": mu_r,
        "test_true_ret": test_true_r,
        "test_std_ret_raw": std_r_raw,
        "test_std_ret_cal": std_r_cal,
        "test_cov_ret_cal": cov_r_cal,
        "test_pred_price": pred_p_median,
        "test_true_price": test_true_p,
        "test_prev_price": test_prev_p,
        "pred_p_q025": pred_p_q025,
        "pred_p_q975": pred_p_q975,
        "pred_p_q10": pred_p_q10,
        "pred_p_q90": pred_p_q90,
        "test_nll_raw": test_nll_raw_ret,
        "test_nll_cal": test_nll_cal_ret,
        "mean_pred_corr": mean_pred_corr,
    }


def evaluate_raw_level_deepvar(
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Đối chứng: Chạy DeepVAR trực tiếp trên chuỗi mức giá gốc P_t (không sai phân log-return)
    với MinMaxScaler fit trên Train (28k-50k) để chứng minh suy thoái khi Test vượt miền (84k-139k).
    """
    set_seed(seed)
    P_arr = price_df.values.astype(np.float32)
    p_scaler = MinMaxScaler().fit(P_arr[:train_end])
    P_s = p_scaler.transform(P_arr).astype(np.float32)

    X_tr, Y_tr, _ = build_causal_windows(P_s, P_s, seq_len, seq_len, train_end)
    X_va, Y_va, _ = build_causal_windows(P_s, P_s, seq_len, train_end, val_end)
    X_te, _, te_idx = build_causal_windows(P_s, P_s, seq_len, val_end, len(P_arr))

    model = CoffeeDeepVAR(in_dim=P_arr.shape[1], hidden_dim=64, cov_hidden_dim=32, num_layers=1, out_dim=P_arr.shape[1], rank=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=16, shuffle=True)

    best_v, best_st, pat = float("inf"), None, 0
    for _ in range(50):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = model.beta_nll_loss(bx, by, beta=0.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, d_v, V_v = model(X_va)
            v_crps = float(gaussian_crps_torch(Y_va, mu_v, torch.sqrt(d_v + (V_v ** 2).sum(dim=-1))).mean().item())
        if v_crps < best_v:
            best_v = v_crps
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                break
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        mu_te, _, _ = model(X_te)
    pred_p = p_scaler.inverse_transform(mu_te.numpy())
    prev_p = price_df.shift(1).iloc[te_idx].values
    true_p = price_df.iloc[te_idx].values
    return {
        "test_pred_price": pred_p,
        "test_true_price": true_p,
        "test_pred_ret": np.log(np.maximum(pred_p, 1.0) / prev_p),
        "test_true_ret": np.log(true_p / prev_p),
    }


def diebold_mariano_loss_diff(loss_bench: np.ndarray, loss_model: np.ndarray, h: int = 1) -> Tuple[float, float]:
    """
    Kiểm định Diebold-Mariano 1 phía (HLN correction) trên chuỗi loss bất kỳ
    (ví dụ: squared error cho điểm, hoặc CRPS từng bước cho phân phối xác suất).
    """
    d = loss_bench.astype(np.float64).flatten() - loss_model.astype(np.float64).flatten()
    n = len(d)
    mean_d = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    if gamma_0 <= 1e-18:
        return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(gamma_0 / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    dm_stat_hln = dm_stat * hln_factor
    p_val = 1.0 - stats.t.cdf(dm_stat_hln, df=n - 1)
    return float(dm_stat_hln), float(p_val)


def compute_deepvar_metrics(
    run: Dict,
    bench_ret_err: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Tính toàn bộ độ đo Điểm (RMSE, MAE, MAPE, R2, DA, DM) & Xác suất (NLL, CRPS, CRPS-Sum, PICP)."""
    true_p = run["test_true_price"]
    pred_p = run["test_pred_price"]
    true_r = run["test_true_ret"]
    mu_r = run["test_pred_ret"]
    std_r_cal = run["test_std_ret_cal"]
    cov_r_cal = run["test_cov_ret_cal"]

    p_err = true_p - pred_p
    rmse_p = float(np.sqrt(np.mean(p_err ** 2)))
    mae_p = float(np.mean(np.abs(p_err)))
    mape_p = float(np.mean(np.abs(p_err / true_p)) * 100.0)
    ss_res_p = np.sum(p_err ** 2)
    ss_tot_p = np.sum((true_p - np.mean(true_p)) ** 2)
    r2_p = float(1.0 - ss_res_p / ss_tot_p) if ss_tot_p > 0 else np.nan

    r_err = true_r - mu_r
    rmse_r = float(np.sqrt(np.mean(r_err ** 2)))
    mask = true_r != 0
    da = float(np.mean(np.sign(mu_r[mask]) == np.sign(true_r[mask])) * 100.0) if np.sum(mask) > 0 else 0.0

    # CRPS biên trên log-return
    crps_mat = gaussian_crps_numpy(true_r, mu_r, std_r_cal)
    crps_ret = float(np.mean(crps_mat))

    # CRPS-Sum (tổng hợp 3 tỉnh — kiểm chứng trực tiếp khả năng ước lượng hiệp phương sai liên tỉnh)
    sum_true = true_r.sum(axis=1)
    sum_mu = mu_r.sum(axis=1)
    sum_std = np.sqrt(np.maximum(cov_r_cal.sum(axis=(1, 2)), 1e-12))
    crps_sum_vec = gaussian_crps_numpy(sum_true, sum_mu, sum_std)
    crps_sum = float(np.mean(crps_sum_vec))

    # Độ bao phủ khoảng dự báo trên giá VND (PICP 80% và 95%) và độ rộng trung bình (MPIW)
    cov80 = float(np.mean((true_p >= run["pred_p_q10"]) & (true_p <= run["pred_p_q90"])) * 100.0)
    cov95 = float(np.mean((true_p >= run["pred_p_q025"]) & (true_p <= run["pred_p_q975"])) * 100.0)
    mpiw95 = float(np.mean(run["pred_p_q975"] - run["pred_p_q025"]))

    dm_stat_r, dm_p_r = (
        diebold_mariano_loss_diff(bench_ret_err ** 2, r_err ** 2, h=1)
        if bench_ret_err is not None
        else (np.nan, np.nan)
    )

    return {
        "RMSE (VND)": rmse_p,
        "MAE (VND)": mae_p,
        "MAPE (%)": mape_p,
        "R2_price": r2_p,
        "DA (%)": da,
        "RMSE_ret": rmse_r,
        "NLL_raw": run["test_nll_raw"],
        "NLL_cal": run["test_nll_cal"],
        "CRPS_ret": crps_ret,
        "CRPS_Sum": crps_sum,
        "Cov80 (%)": cov80,
        "Cov95 (%)": cov95,
        "MPIW95 (VND)": mpiw95,
        "DM_stat_vs_RW": dm_stat_r,
        "DM_p_vs_RW": dm_p_r,
        "tau_cal": run["tau_cal"],
        "best_epoch": float(run["best_epoch"]),
    }


def compute_deepvar_permutation_importance(
    run: Dict,
    feature_names: List[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Đo tầm quan trọng đặc trưng của DeepVAR theo CẢ hai góc nhìn:
      1. Tăng RMSE điểm (Mean branch mu_t)
      2. Tăng CRPS (Joint Mean + Uncertainty branch mu_t, Sigma_t).
    """
    rng = np.random.default_rng(seed)
    model: CoffeeDeepVAR = run["model"]
    X_te: torch.Tensor = run["X_te_tensor"]
    y_scaler: StandardScaler = run["y_scaler"]
    tau = run["tau_cal"]
    prev_p = run["test_prev_price"]
    true_p = run["test_true_price"]
    true_r = run["test_true_ret"]
    scale = y_scaler.scale_

    model.eval()
    with torch.no_grad():
        mu_b, d_b, V_b = model(X_te)
        v_sum_b = torch.zeros_like(d_b) if V_b is None else (V_b ** 2).sum(dim=-1)
        std_b = torch.sqrt(d_b + v_sum_b).numpy() * tau * scale[None, :]
        mu_r_b = y_scaler.inverse_transform(mu_b.numpy())

    base_rmse = float(np.sqrt(np.mean((true_p - prev_p * np.exp(mu_r_b)) ** 2)))
    base_crps = float(np.mean(gaussian_crps_numpy(true_r, mu_r_b, std_b)))

    n_samples = X_te.shape[0]
    rows = []
    for j, fname in enumerate(feature_names):
        d_rmse, d_crps = [], []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            Xp = X_te.clone()
            Xp[:, :, j] = X_te[perm_idx, :, j]
            with torch.no_grad():
                mu_p, d_p, V_p = model(Xp)
                v_sum_p = torch.zeros_like(d_p) if V_p is None else (V_p ** 2).sum(dim=-1)
                std_p = torch.sqrt(d_p + v_sum_p).numpy() * tau * scale[None, :]
                mu_r_p = y_scaler.inverse_transform(mu_p.numpy())
            rmse_p = float(np.sqrt(np.mean((true_p - prev_p * np.exp(mu_r_p)) ** 2)))
            crps_p = float(np.mean(gaussian_crps_numpy(true_r, mu_r_p, std_p)))
            d_rmse.append(rmse_p - base_rmse)
            d_crps.append(crps_p - base_crps)
        rows.append(
            {
                "Feature": fname,
                "RMSE_Increase_VND": float(np.mean(d_rmse)),
                "CRPS_Increase": float(np.mean(d_crps)),
            }
        )

    df = pd.DataFrame(rows).sort_values("RMSE_Increase_VND", ascending=False).reset_index(drop=True)
    tot_r = df["RMSE_Increase_VND"].clip(lower=0.0).sum()
    tot_c = df["CRPS_Increase"].clip(lower=0.0).sum()
    df["RMSE_Share (%)"] = df["RMSE_Increase_VND"].clip(lower=0.0) / tot_r * 100.0 if tot_r > 0 else 0.0
    df["CRPS_Share (%)"] = df["CRPS_Increase"].clip(lower=0.0) / tot_c * 100.0 if tot_c > 0 else 0.0
    return df


# Alias tương thích hai chiều
DeepVARModel = CoffeeDeepVAR
DeepVAR = CoffeeDeepVAR

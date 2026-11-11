# -*- coding: utf-8 -*-
"""
========================================================================================
MÔ-ĐUN MÔ HÌNH DEEPVARX THẾ HỆ MỚI (TIMEXER DUAL-STREAM + GRN + LOW-RANK COVARIANCE)
Ứng dụng: Dự báo xác suất đa biến giá thu mua cà phê Robusta tại Việt Nam
========================================================================================
Tổng hợp lý thuyết chuyên sâu từ báo cáo trong dự án & nghiên cứu quốc tế mới nhất:
  1. TimeXer (Wang et al., NeurIPS 2024 - arXiv:2402.19072):
     Tách luồng nội sinh (Endogenous Temporal LSTM_Y) khỏi các chuỗi ngoại sinh được
     nhúng độc lập theo từng biến (Variate-wise Exogenous Tokenization), rồi liên kết qua
     Multi-Head Cross-Attention để lọc nhiễu ngoại sinh tự động (khắc phục suy thoái của
     LSTM 3 & DeepVAR 3 khi số lượng biến ngoại sinh tăng lên 20 biến).
  2. Temporal Fusion Transformer (Lim et al., IJF 2021):
     Khối Gated Residual Network (GRN: ELU + GLU + LayerNorm + Residual) kiểm soát dòng
     tín hiệu ngoại sinh, triệt tiêu thành phần phi tuyến thừa khi nhiễu lớn.
  3. DeepVAR Low-Rank Gaussian Process (Salinas et al., NeurIPS 2019) & Beta-NLL
     (Seitzer et al., ICLR 2022): Hiệp phương sai hạng thấp + đường chéo
     Sigma_t = diag(d_t) + V_t V_t^T (rank=2) tối ưu bằng Beta-NLL (beta=0.5) và dừng
     sớm theo CRPS + hiệu chỉnh quy mô tau trên Validation.
  4. Đối chứng thất bại của kiến trúc cũ:
     - Raw-Level DeepVARX: Bão hòa hàm kích hoạt vì giá Test (84k-139k) vượt Train (28k-50k).
     - Legacy OLS-Skip DeepVARX (benchmark_coffee_models.py): Khởi tạo nhánh tuyến tính
       bằng OLS VARX(5,2) cố định trên 2013-2022 (beta_London=0.630) khiến mô hình phản ứng
       thái quá khi độ co giãn dịch chuyển về 0.382 ở Test.

Ba phiên bản thực nghiệm (đồng bộ tập đặc trưng & 542 ngày Test với LSTM và DeepVAR):
  - DeepVARX 1 (Baseline)       : Chỉ dùng nội sinh quá khứ (3 tỉnh, M_exog = 0).
  - DeepVARX 2 (Multivariate)   : Giá 3 tỉnh + Thời tiết cơ bản + Kinh tế cơ bản (11 biến).
  - DeepVARX 3 (Full Exogenous) : Giá 3 tỉnh + Toàn bộ biến ngoại sinh phù hợp (20 biến).
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
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CoffeeDataBundle:
    """Chứa chuỗi giá gốc, chuỗi tỷ suất sinh lợi dừng I(0) và 3 tập biến DeepVARX 1/2/3."""

    raw_df: pd.DataFrame
    price_df: pd.DataFrame
    ret_df: pd.DataFrame
    dates: pd.DatetimeIndex
    train_end: int
    val_end: int
    feature_sets: Dict[str, pd.DataFrame]


def load_and_prepare_coffee_data(
    csv_path: str = "data_daily_panel_clean_v2.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> CoffeeDataBundle:
    """Nạp dữ liệu panel ngày, biến đổi dừng I(0) và tạo 3 tập đặc trưng lag >= 1."""
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

    all_econ_cols = CORE_ECON_LEVEL_COLS + EXT_ECON_LEVEL_COLS
    econ_level_df = raw_df.groupby("date")[all_econ_cols].mean().loc[dates]
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
            "DeepVARX 1": x_v1,
            "DeepVARX 2": x_v2,
            "DeepVARX 3": x_v3,
        },
    )


class GatedResidualNetwork(nn.Module):
    """Khối GRN (Lim et al., TFT 2021): ELU + Gated Linear Unit (GLU) + LayerNorm + Residual."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim * 2)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        h = self.drop(F.elu(self.fc1(x)))
        a, b = self.fc2(h).chunk(2, dim=-1)
        return self.norm(res + a * torch.sigmoid(b))


class CoffeeDeepVARX(nn.Module):
    """
    Kiến trúc TimeXer-DeepVARX xác suất đa biến:
      1. Nhánh Nội sinh (Endogenous Stream): LSTM_Y(Y_{t-L:t-1}) -> h_y in R^H.
      2. Nhánh Ngoại sinh (Variate-wise Exogenous Tokenization - TimeXer, NeurIPS 2024):
         Mỗi biến ngoại sinh m in {1..M} được nhúng độc lập trên L bước trễ thành token
         e_m in R^H mà không trộn lẫn trước khi chọn lọc.
      3. Multi-Head Cross-Attention (query=h_y, key/value=E_X) + GRN Fusion:
         Trích xuất trọng số chú ý alpha_m theo từng biến ngoại sinh (sum_m alpha_m = 1),
         tự động khuếch đại tín hiệu mạnh (ret_london, ret_arabica) và triệt tiêu nhiễu.
      4. Nhánh Hiệp phương sai Low-Rank (Salinas et al., 2019):
         LSTM_cov -> Sigma_t = diag(d_t) + V_t V_t^T (rank = 2).
    """

    def __init__(
        self,
        k_endog: int = 3,
        m_exog: int = 8,
        seq_len: int = 10,
        hidden_dim: int = 64,
        cov_hidden_dim: int = 32,
        num_heads: int = 4,
        rank: int = 2,
        dropout: float = 0.15,
        in_dropout: float = 0.10,
        min_std: float = 0.20,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.k_endog = k_endog
        self.m_exog = m_exog
        self.seq_len = seq_len
        self.rank = rank
        self.min_std = min_std
        self.use_cross_attn = use_cross_attn and (m_exog > 0)

        self.in_drop = nn.Dropout(in_dropout) if in_dropout > 0.0 else nn.Identity()

        # Nhánh nội sinh (hoặc trộn toàn bộ biến nếu use_cross_attn=False cho ablation)
        lstm_in_dim = k_endog if (self.use_cross_attn or m_exog == 0) else (k_endog + m_exog)
        self.lstm_y = nn.LSTM(lstm_in_dim, hidden_dim, num_layers=1, batch_first=True)

        # Nhánh ngoại sinh độc lập từng biến + Cross-Attention + GRN
        if self.use_cross_attn:
            self.var_proj_w = nn.Parameter(torch.empty(m_exog, seq_len, hidden_dim))
            self.var_proj_b = nn.Parameter(torch.zeros(m_exog, hidden_dim))
            nn.init.xavier_uniform_(self.var_proj_w)
            self.var_norm = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.post_attn_norm = nn.LayerNorm(hidden_dim)
            self.fusion_grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, dropout=dropout)

        self.drop_mu = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.mu_head = nn.Linear(hidden_dim, k_endog)

        # Nhánh hiệp phương sai Low-Rank tách biệt
        self.lstm_cov = nn.LSTM(k_endog + m_exog, cov_hidden_dim, num_layers=1, batch_first=True)
        self.drop_cov = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.diag_head = nn.Linear(cov_hidden_dim, k_endog)
        self.factor_head = nn.Linear(cov_hidden_dim, k_endog * rank) if rank > 0 else None

        nn.init.constant_(self.diag_head.bias, 0.0)
        if self.factor_head is not None:
            nn.init.xavier_uniform_(self.factor_head.weight, gain=0.2)
            nn.init.zeros_(self.factor_head.bias)

    def forward(
        self, x_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        attn_weights = None
        if self.use_cross_attn:
            y_seq = x_seq[:, :, : self.k_endog]
            x_exog = x_seq[:, :, self.k_endog :]
            hy_seq, _ = self.lstm_y(self.in_drop(y_seq))
            hy_last = hy_seq[:, -1, :]

            x_var = self.in_drop(x_exog).transpose(1, 2)  # (B, M, L)
            tokens = torch.einsum("bml,mlh->bmh", x_var, self.var_proj_w) + self.var_proj_b.unsqueeze(0)
            tokens = self.var_norm(F.gelu(tokens))  # (B, M, H)
            attn_out, attn_weights = self.cross_attn(
                query=hy_last.unsqueeze(1), key=tokens, value=tokens, average_attn_weights=True
            )
            ctx = self.post_attn_norm(hy_last + attn_out.squeeze(1))
            h_final = self.fusion_grn(torch.cat([hy_last, ctx], dim=-1))
        else:
            hy_seq, _ = self.lstm_y(self.in_drop(x_seq))
            h_final = hy_seq[:, -1, :]

        mu = self.mu_head(self.drop_mu(h_final))

        hc_seq, _ = self.lstm_cov(self.in_drop(x_seq))
        hc = self.drop_cov(hc_seq[:, -1, :])
        d = F.softplus(self.diag_head(hc)) + (self.min_std ** 2)
        V = self.factor_head(hc).view(-1, self.k_endog, self.rank) if self.rank > 0 else None
        return mu, d, V, attn_weights

    def joint_covariance(self, d: torch.Tensor, V: Optional[torch.Tensor]) -> torch.Tensor:
        cov = torch.diag_embed(d)
        if V is not None:
            cov = cov + torch.bmm(V, V.transpose(1, 2))
        return cov

    def log_prob(
        self, y: torch.Tensor, mu: torch.Tensor, d: torch.Tensor, V: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.rank > 0 and V is not None:
            return LowRankMultivariateNormal(loc=mu, cov_factor=V, cov_diag=d).log_prob(y)
        return -0.5 * (math.log(2.0 * math.pi) + torch.log(d) + ((y - mu) ** 2) / d).sum(dim=-1)

    def beta_nll_loss(self, x_seq: torch.Tensor, y: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        mu, d, V, _ = self.forward(x_seq)
        nll_vec = -self.log_prob(y, mu, d, V)
        if beta > 0.0:
            v_sum = torch.zeros_like(d) if V is None else (V ** 2).sum(dim=-1)
            marg_var = d + v_sum
            weight = (marg_var.mean(dim=-1).detach()) ** beta
            return (weight * nll_vec).mean()
        return nll_vec.mean()


def gaussian_crps_torch(y_true: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    s_safe = torch.clamp(sigma, min=1e-8)
    z = (y_true - mu) / s_safe
    phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return s_safe * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def gaussian_crps_numpy(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
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


def fit_single_deepvarx_seed(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    hidden_dim: int = 64,
    cov_hidden_dim: int = 32,
    rank: int = 2,
    dropout: float = 0.15,
    in_dropout: float = 0.10,
    min_std: float = 0.20,
    beta_nll: float = 0.5,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    n_mc_samples: int = 1000,
    use_cross_attn: bool = True,
    seed: int = 42,
) -> Dict:
    """Huấn luyện 1 seed DeepVARX với Beta-NLL, CRPS Early Stopping, hiệu chỉnh tau trên Val và xuất đầy đủ tensor phục vụ Conformal/Jensen."""
    set_seed(seed)
    K = Y_df.shape[1]
    M = X_df.shape[1] - K
    X_arr = X_df.values.astype(np.float32)
    Y_arr = Y_df.values.astype(np.float32)

    x_scaler = StandardScaler().fit(X_arr[:train_end])
    y_scaler = StandardScaler().fit(Y_arr[:train_end])
    X_s = x_scaler.transform(X_arr).astype(np.float32)
    Y_s = y_scaler.transform(Y_arr).astype(np.float32)

    X_tr, Y_tr, _ = build_causal_windows(X_s, Y_s, seq_len, seq_len, train_end)
    X_va, Y_va, va_idx = build_causal_windows(X_s, Y_s, seq_len, train_end, val_end)
    X_te, Y_te, te_idx = build_causal_windows(X_s, Y_s, seq_len, val_end, len(Y_arr))

    model = CoffeeDeepVARX(
        k_endog=K,
        m_exog=M,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        cov_hidden_dim=cov_hidden_dim,
        num_heads=4,
        rank=rank,
        dropout=dropout,
        in_dropout=in_dropout,
        min_std=min_std,
        use_cross_attn=use_cross_attn,
    )
    OptCls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
    opt = OptCls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    best_crps = float("inf")
    best_state = None
    best_epoch = 0
    pat_cnt = 0
    train_losses, val_crps_hist = [], []

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
            mu_v, d_v, V_v, _ = model(X_va)
            v_sum = torch.zeros_like(d_v) if V_v is None else (V_v ** 2).sum(dim=-1)
            marg_std = torch.sqrt(d_v + v_sum)
            v_crps = float(gaussian_crps_torch(Y_va, mu_v, marg_std).mean().item())
        val_crps_hist.append(v_crps)

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
        mu_v, d_v, V_v, _ = model(X_va)
        cov_v = model.joint_covariance(d_v, V_v)
        diff_v = (Y_va - mu_v).unsqueeze(-1)
        quad_v = torch.bmm(diff_v.transpose(1, 2), torch.linalg.solve(cov_v, diff_v)).squeeze()
        tau = max(1.0, float(torch.sqrt(torch.mean(quad_v) / K).item()))

        mu_t, d_t, V_t, attn_te = model(X_te)
        cov_t_raw = model.joint_covariance(d_t, V_t)
        d_t_cal = d_t * (tau ** 2)
        V_t_cal = V_t * tau if V_t is not None else None
        cov_t_cal = cov_t_raw * (tau ** 2)
        test_nll_cal_s = float((-model.log_prob(Y_te, mu_t, d_t_cal, V_t_cal)).mean().item())

        torch.manual_seed(seed + 999)
        if rank > 0 and V_t_cal is not None:
            mc_s = LowRankMultivariateNormal(loc=mu_t, cov_factor=V_t_cal, cov_diag=d_t_cal).sample((n_mc_samples,))
        else:
            mc_s = mu_t.unsqueeze(0) + torch.randn(n_mc_samples, mu_t.size(0), K) * torch.sqrt(d_t_cal).unsqueeze(0)

    scale = y_scaler.scale_
    mean_y = y_scaler.mean_
    cov_scale = np.outer(scale, scale)[None, :, :]
    mu_r = y_scaler.inverse_transform(mu_t.numpy())
    mu_r_val = y_scaler.inverse_transform(mu_v.numpy())
    cov_r_val_raw = cov_v.numpy() * cov_scale
    std_r_val_raw = np.sqrt(np.diagonal(cov_r_val_raw, axis1=1, axis2=2))
    cov_r_raw = cov_t_raw.numpy() * cov_scale
    std_r_raw = np.sqrt(np.diagonal(cov_r_raw, axis1=1, axis2=2))
    cov_r_cal = cov_t_cal.numpy() * cov_scale
    std_r_cal = np.sqrt(np.diagonal(cov_r_cal, axis1=1, axis2=2))
    test_nll_cal_ret = test_nll_cal_s + float(np.sum(np.log(scale)))

    mc_r = mc_s.numpy() * scale[None, None, :] + mean_y[None, None, :]
    test_prev_p = price_df.shift(1).iloc[te_idx].values
    test_true_p = price_df.iloc[te_idx].values
    test_true_r = Y_df.iloc[te_idx].values
    mc_prices = test_prev_p[None, :, :] * np.exp(mc_r)

    val_prev_p = price_df.shift(1).iloc[va_idx].values
    val_true_p = price_df.iloc[va_idx].values
    val_true_r = Y_df.iloc[va_idx].values
    val_pred_p = val_prev_p * np.exp(mu_r_val)
    val_rmse_p = float(np.sqrt(np.mean((val_true_p - val_pred_p) ** 2)))

    attn_np = attn_te.squeeze(1).numpy() if attn_te is not None else None  # (N_te, M)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "tau_cal": tau,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_crps_hist": val_crps_hist,
        "val_rmse_price": val_rmse_p,
        "X_te_tensor": X_te,
        "va_idx": va_idx,
        "te_idx": te_idx,
        "val_pred_ret": mu_r_val,
        "val_true_ret": val_true_r,
        "val_std_ret_raw": std_r_val_raw,
        "test_pred_ret": mu_r,
        "test_true_ret": test_true_r,
        "test_std_ret_raw": std_r_raw,
        "test_std_ret_cal": std_r_cal,
        "test_cov_ret_cal": cov_r_cal,
        "test_pred_price": test_prev_p * np.exp(mu_r),
        "test_true_price": test_true_p,
        "test_prev_price": test_prev_p,
        "mc_prices": mc_prices,
        "pred_p_q025": np.quantile(mc_prices, 0.025, axis=0),
        "pred_p_q975": np.quantile(mc_prices, 0.975, axis=0),
        "pred_p_q05": np.quantile(mc_prices, 0.05, axis=0),
        "pred_p_q95": np.quantile(mc_prices, 0.95, axis=0),
        "pred_p_q10": np.quantile(mc_prices, 0.10, axis=0),
        "pred_p_q90": np.quantile(mc_prices, 0.90, axis=0),
        "test_nll_cal": test_nll_cal_ret,
        "attn_weights_test": attn_np,
    }


class LegacyOLSHybridDeepVARX(nn.Module):
    """Tái hiện kiến trúc DeepVARX cũ (benchmark_coffee_models.py): OLS VARX(5,2) + LSTM phần dư."""

    def __init__(self, in_features: int, z_dim: int, out_features: int, hidden_dim: int, w_ols: np.ndarray) -> None:
        super().__init__()
        self.linear_branch = nn.Linear(z_dim, out_features, bias=False)
        with torch.no_grad():
            self.linear_branch.weight.copy_(torch.tensor(w_ols.T, dtype=torch.float32))
        self.lstm = nn.LSTM(in_features, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_features),
        )
        with torch.no_grad():
            self.head[-1].weight.mul_(0.1)
            self.head[-1].bias.zero_()

    def forward(self, z_linear: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        return self.linear_branch(z_linear) + self.head(out[:, -1, :])


def evaluate_legacy_ols_deepvarx(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    p_lag: int = 5,
    q_lag: int = 2,
    seed: int = 42,
) -> Dict:
    """Chạy đối chứng DeepVARX kiểu cũ (OLS VARX(5,2) + LSTM residual) trên cùng tập dữ liệu."""
    set_seed(seed)
    y_arr = Y_df.values
    x_arr = X_df.iloc[:, Y_df.shape[1] :].values
    start = max(p_lag, q_lag)
    Z_list, seq_list, Y_list, idx_list = [], [], [], []
    for t in range(start, len(y_arr)):
        yl = y_arr[t - p_lag : t][::-1].flatten()
        xl = x_arr[t - q_lag : t][::-1].flatten()
        Z_list.append(np.concatenate([[1.0], yl, xl]))
        seq_list.append(X_df.values[t - p_lag : t])
        Y_list.append(y_arr[t])
        idx_list.append(t)

    Z_arr = np.asarray(Z_list, dtype=np.float32)
    seq_arr = np.asarray(seq_list, dtype=np.float32)
    Y_arr = np.asarray(Y_list, dtype=np.float32)
    idx_arr = np.asarray(idx_list)

    tr_mask = idx_arr < train_end
    va_mask = (idx_arr >= train_end) & (idx_arr < val_end)
    te_mask = idx_arr >= val_end

    W_ols = np.linalg.lstsq(Z_arr[tr_mask], Y_arr[tr_mask], rcond=None)[0]
    sc = StandardScaler().fit(seq_arr[tr_mask].reshape(-1, X_df.shape[1]))
    seq_s = sc.transform(seq_arr.reshape(-1, X_df.shape[1])).reshape(seq_arr.shape)

    model = LegacyOLSHybridDeepVARX(X_df.shape[1], Z_arr.shape[1], Y_df.shape[1], 32, W_ols)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(Z_arr[tr_mask]), torch.tensor(seq_s[tr_mask]), torch.tensor(Y_arr[tr_mask])),
        batch_size=32,
        shuffle=True,
    )

    best_v, best_st, pat = float("inf"), None, 0
    tz_va, tx_va, ty_va = torch.tensor(Z_arr[va_mask]), torch.tensor(seq_s[va_mask]), torch.tensor(Y_arr[va_mask])
    for _ in range(150):
        model.train()
        for bz, bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bz, bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(tz_va, tx_va), ty_va).item()
        if vl < best_v:
            best_v = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 15:
                break

    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        pred_r = model(torch.tensor(Z_arr[te_mask]), torch.tensor(seq_s[te_mask])).numpy()

    te_idx = idx_arr[te_mask]
    prev_p = price_df.shift(1).iloc[te_idx].values
    true_p = price_df.iloc[te_idx].values
    return {
        "te_idx": te_idx,
        "test_pred_ret": pred_r,
        "test_true_ret": Y_arr[te_mask],
        "test_pred_price": prev_p * np.exp(pred_r),
        "test_true_price": true_p,
    }


def evaluate_raw_level_deepvarx(
    price_df: pd.DataFrame,
    exog_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    seed: int = 42,
) -> Dict:
    """Đối chứng suy thoái miền giá khi đưa trực tiếp giá gốc P_t vào TimeXer-DeepVARX."""
    set_seed(seed)
    P_arr = price_df.values.astype(np.float32)
    E_arr = exog_df.values.astype(np.float32)
    p_sc = MinMaxScaler().fit(P_arr[:train_end])
    e_sc = StandardScaler().fit(E_arr[:train_end])
    X_s = np.hstack([p_sc.transform(P_arr), e_sc.transform(E_arr)]).astype(np.float32)
    Y_s = p_sc.transform(P_arr).astype(np.float32)

    X_tr, Y_tr, _ = build_causal_windows(X_s, Y_s, seq_len, seq_len, train_end)
    X_va, Y_va, _ = build_causal_windows(X_s, Y_s, seq_len, train_end, val_end)
    X_te, _, te_idx = build_causal_windows(X_s, Y_s, seq_len, val_end, len(P_arr))

    model = CoffeeDeepVARX(k_endog=P_arr.shape[1], m_exog=E_arr.shape[1], seq_len=seq_len, dropout=0.1, in_dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
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
            mv, dv, Vv, _ = model(X_va)
            vc = float(gaussian_crps_torch(Y_va, mv, torch.sqrt(dv + (Vv ** 2).sum(dim=-1))).mean().item())
        if vc < best_v:
            best_v = vc
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= 10:
                break
    model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad():
        mt, _, _, _ = model(X_te)
    pred_p = p_sc.inverse_transform(mt.numpy())
    prev_p = price_df.shift(1).iloc[te_idx].values
    true_p = price_df.iloc[te_idx].values
    return {
        "test_pred_price": pred_p,
        "test_true_price": true_p,
        "test_pred_ret": np.log(np.maximum(pred_p, 1.0) / prev_p),
        "test_true_ret": np.log(true_p / prev_p),
    }


def format_pval(p: float, threshold: float = 1e-4) -> str:
    """Định dạng p-value khoa học, không bao giờ in 0.0000 khi p rất nhỏ."""
    if not np.isfinite(p):
        return "-"
    if p < threshold:
        return f"< {threshold:.4f}"
    return f"{p:.4f}"


def diebold_mariano_hac(
    loss_bench: np.ndarray,
    loss_model: np.ndarray,
    h: int = 1,
    max_lag: int = 5,
    aggregate_provinces: bool = True,
) -> Dict[str, float]:
    """
    Kiểm định Diebold-Mariano đa biến chuẩn theo từng ngày (N = số ngày Test)
    với hiệp phương sai dài hạn Newey-West HAC (Bartlett kernel) và hiệu chỉnh mẫu hữu hạn HLN.
    Trả về đồng thời p-value 1 phía, 2 phía và kiểm định Ljung-Box trên chuỗi chênh lệch sai số d_t.
    """
    lb = np.mean(loss_bench, axis=1) if (aggregate_provinces and loss_bench.ndim == 2) else loss_bench.reshape(-1)
    lm = np.mean(loss_model, axis=1) if (aggregate_provinces and loss_model.ndim == 2) else loss_model.reshape(-1)
    d = lb.astype(np.float64) - lm.astype(np.float64)
    n = len(d)
    d_mean = float(np.mean(d))
    dc = d - d_mean
    gamma_0 = float(np.sum(dc ** 2) / n)
    lrv = gamma_0
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        gamma_k = float(np.sum(dc[k:] * dc[:-k]) / n)
        lrv += 2.0 * w * gamma_k
    lrv = max(lrv, 1e-18)
    if gamma_0 <= 1e-18:
        return {"dm_stat": 0.0, "p_1s": 1.0, "p_2s": 1.0, "n_obs": float(n), "ac1": 0.0, "lb_p": 1.0}
    dm = d_mean / math.sqrt(lrv / n)
    hln_factor = math.sqrt((n + 1 - 2 * h + (h / n) * (h - 1)) / n)
    dm_hln = dm * hln_factor
    p_1s = float(1.0 - stats.t.cdf(dm_hln, df=n - 1))
    p_2s = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    ac1 = float(np.corrcoef(dc[1:], dc[:-1])[0, 1]) if n > 2 else 0.0
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_res = acorr_ljungbox(dc, lags=[max(1, max_lag)], return_df=True)
    lb_p = float(lb_res["lb_pvalue"].iloc[0])
    return {
        "dm_stat": float(dm_hln),
        "p_1s": p_1s,
        "p_2s": p_2s,
        "n_obs": float(n),
        "ac1": ac1,
        "lb_p": lb_p,
    }


def diebold_mariano_loss_diff(loss_bench: np.ndarray, loss_model: np.ndarray, h: int = 1) -> Tuple[float, float]:
    """Wrapper chuẩn DM đa biến ngày N=542 với Newey-West HAC (trả về DM stat và p-value 2 phía)."""
    res = diebold_mariano_hac(loss_bench, loss_model, h=h, max_lag=5, aggregate_provinces=True)
    return res["dm_stat"], res["p_2s"]


def winkler_interval_score(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Tính Winkler / Interval Score cân bằng độ rộng MPIW và hình phạt vượt biên (thấp hơn = tốt)."""
    width = hi - lo
    below = (2.0 / alpha) * np.maximum(lo - y_true, 0.0)
    above = (2.0 / alpha) * np.maximum(y_true - hi, 0.0)
    return float(np.mean(width + below + above))


def compute_deepvarx_metrics(run: Dict, bench_ret_err: Optional[np.ndarray] = None) -> Dict[str, float]:
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
    mask = true_r != 0
    da = float(np.mean(np.sign(mu_r[mask]) == np.sign(true_r[mask])) * 100.0) if np.sum(mask) > 0 else 0.0

    crps_ret = float(np.mean(gaussian_crps_numpy(true_r, mu_r, std_r_cal)))
    sum_std = np.sqrt(np.maximum(cov_r_cal.sum(axis=(1, 2)), 1e-12))
    crps_sum = float(np.mean(gaussian_crps_numpy(true_r.sum(axis=1), mu_r.sum(axis=1), sum_std)))
    cov80 = float(np.mean((true_p >= run["pred_p_q10"]) & (true_p <= run["pred_p_q90"])) * 100.0)
    cov90 = float(np.mean((true_p >= run["pred_p_q05"]) & (true_p <= run["pred_p_q95"])) * 100.0)
    cov95 = float(np.mean((true_p >= run["pred_p_q025"]) & (true_p <= run["pred_p_q975"])) * 100.0)
    mpiw80 = float(np.mean(run["pred_p_q90"] - run["pred_p_q10"]))
    mpiw90 = float(np.mean(run["pred_p_q95"] - run["pred_p_q05"]))
    mpiw95 = float(np.mean(run["pred_p_q975"] - run["pred_p_q025"]))
    wink80 = winkler_interval_score(true_p, run["pred_p_q10"], run["pred_p_q90"], alpha=0.20)
    wink90 = winkler_interval_score(true_p, run["pred_p_q05"], run["pred_p_q95"], alpha=0.10)
    wink95 = winkler_interval_score(true_p, run["pred_p_q025"], run["pred_p_q975"], alpha=0.05)

    dm_res = (
        diebold_mariano_hac(bench_ret_err ** 2, r_err ** 2, h=1, max_lag=5, aggregate_provinces=True)
        if bench_ret_err is not None
        else {"dm_stat": np.nan, "p_2s": np.nan}
    )

    return {
        "RMSE (VND)": rmse_p,
        "MAE (VND)": mae_p,
        "MAPE (%)": mape_p,
        "R2_price": r2_p,
        "DA (%)": da,
        "NLL_cal": run["test_nll_cal"],
        "CRPS_ret": crps_ret,
        "CRPS_Sum": crps_sum,
        "Cov80 (%)": cov80,
        "Cov90 (%)": cov90,
        "Cov95 (%)": cov95,
        "MPIW80 (VND)": mpiw80,
        "MPIW90 (VND)": mpiw90,
        "MPIW95 (VND)": mpiw95,
        "Winkler80": wink80,
        "Winkler90": wink90,
        "Winkler95": wink95,
        "DM_stat_vs_RW": dm_res["dm_stat"],
        "DM_p_vs_RW": dm_res["p_2s"],
        "tau_cal": run["tau_cal"],
        "best_epoch": float(run["best_epoch"]),
    }


def compute_deepvarx_permutation_importance(
    run: Dict,
    feature_names: List[str],
    n_repeats: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Tính đồng thời Permutation Feature Importance (RMSE, CRPS) và Input Gradient Saliency."""
    rng = np.random.default_rng(seed)
    model: CoffeeDeepVARX = run["model"]
    X_te: torch.Tensor = run["X_te_tensor"]
    y_scaler: StandardScaler = run["y_scaler"]
    tau = run["tau_cal"]
    prev_p = run["test_prev_price"]
    true_p = run["test_true_price"]
    true_r = run["test_true_ret"]
    scale = y_scaler.scale_

    model.eval()
    with torch.no_grad():
        mu_b, d_b, V_b, _ = model(X_te)
        v_sum_b = torch.zeros_like(d_b) if V_b is None else (V_b ** 2).sum(dim=-1)
        std_b = torch.sqrt(d_b + v_sum_b).numpy() * tau * scale[None, :]
        mu_r_b = y_scaler.inverse_transform(mu_b.numpy())
    base_rmse = float(np.sqrt(np.mean((true_p - prev_p * np.exp(mu_r_b)) ** 2)))
    base_crps = float(np.mean(gaussian_crps_numpy(true_r, mu_r_b, std_b)))

    # Input Gradient Saliency qua K=3 mục tiêu (không triệt tiêu dấu giữa các tỉnh)
    X_grad_in = X_te.clone().detach().requires_grad_(True)
    mu_g, _, _, _ = model(X_grad_in)
    abs_grads = torch.zeros_like(X_grad_in)
    k_targets = mu_g.shape[1]
    for k in range(k_targets):
        grad_k = torch.autograd.grad(mu_g[:, k].sum(), X_grad_in, retain_graph=(k < k_targets - 1))[0]
        abs_grads = abs_grads + grad_k.abs()
    grad_scores = abs_grads.mean(dim=(0, 1)).detach().numpy()
    grad_shares = grad_scores / max(float(grad_scores.sum()), 1e-12) * 100.0

    n_samples = X_te.shape[0]
    rows = []
    for j, fname in enumerate(feature_names):
        d_rmse, d_crps = [], []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            Xp = X_te.clone()
            Xp[:, :, j] = X_te[perm_idx, :, j]
            with torch.no_grad():
                mu_p, d_p, V_p, _ = model(Xp)
                v_sum_p = torch.zeros_like(d_p) if V_p is None else (V_p ** 2).sum(dim=-1)
                std_p = torch.sqrt(d_p + v_sum_p).numpy() * tau * scale[None, :]
                mu_r_p = y_scaler.inverse_transform(mu_p.numpy())
            d_rmse.append(float(np.sqrt(np.mean((true_p - prev_p * np.exp(mu_r_p)) ** 2))) - base_rmse)
            d_crps.append(float(np.mean(gaussian_crps_numpy(true_r, mu_r_p, std_p))) - base_crps)
        rows.append(
            {
                "Feature": fname,
                "RMSE_Increase_VND": float(np.mean(d_rmse)),
                "CRPS_Increase": float(np.mean(d_crps)),
                "Grad_Saliency": float(grad_scores[j]),
                "Grad_Share (%)": float(grad_shares[j]),
            }
        )

    df = pd.DataFrame(rows).sort_values("RMSE_Increase_VND", ascending=False).reset_index(drop=True)
    tot_r = df["RMSE_Increase_VND"].clip(lower=0.0).sum()
    tot_c = df["CRPS_Increase"].clip(lower=0.0).sum()
    df["RMSE_Share (%)"] = df["RMSE_Increase_VND"].clip(lower=0.0) / tot_r * 100.0 if tot_r > 0 else 0.0
    df["CRPS_Share (%)"] = df["CRPS_Increase"].clip(lower=0.0) / tot_c * 100.0 if tot_c > 0 else 0.0
    return df


def empirical_crps_from_samples(samples: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Tính CRPS chính xác O(M log M) từ tập M mẫu phân phối thực nghiệm.
    samples: (M, N, ...) ; y_true: (N, ...)
    """
    m = samples.shape[0]
    s_sorted = np.sort(samples, axis=0)
    term1 = np.mean(np.abs(s_sorted - y_true[None, ...]), axis=0)
    idx_weights = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    shape_expand = (m,) + (1,) * (samples.ndim - 1)
    term2 = np.sum(idx_weights.reshape(shape_expand) * s_sorted, axis=0) / (m * m)
    return term1 - term2


def compute_calibration_suite(
    model_label: str,
    mu_val: np.ndarray,
    std_val_raw: np.ndarray,
    true_val_r: np.ndarray,
    mu_test: np.ndarray,
    std_test_raw: np.ndarray,
    tau_val: float,
    true_test_r: np.ndarray,
    prev_p: np.ndarray,
    true_p: np.ndarray,
    cov_test_cal: Optional[np.ndarray] = None,
    rolling_window: int = 120,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Đánh giá toàn diện độ hiệu chỉnh xác suất trên 3 cơ chế hoàn toàn nhân quả:
      1) Gauss + Val tau (phương sai tỷ lệ theo Mahalanobis trên Validation)
      2) Split Conformal (Val-only, phi tham số tĩnh)
      3) Causal Rolling Asymmetric Conformal (EnbPI window=120 sai số quá khứ [t-W, t-1])
    Xuất đầy đủ Cov80/90/95, MPIW80/90/95, Winkler80/90/95, CRPS, CRPS-Sum, PIT KS và Reliability curve.
    """
    z_val = (true_val_r - mu_val) / np.maximum(std_val_raw, 1e-8)
    z_test = (true_test_r - mu_test) / np.maximum(std_test_raw, 1e-8)
    n_v, n_t, k_dim = len(z_val), len(z_test), z_test.shape[1]
    all_z = np.vstack([z_val, z_test])

    nominal_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary_rows = []
    diag_details: Dict[str, Dict] = {}

    for method in ["Gauss + Val tau", "Split Conformal (Val)", f"Causal Rolling Conformal (W={rolling_window})"]:
        rel_covs = []
        intervals_by_cov: Dict[int, Dict[str, float]] = {}

        for cov_nom in nominal_grid:
            alpha = 1.0 - cov_nom
            if method == "Gauss + Val tau":
                zq = float(stats.norm.ppf(1.0 - alpha / 2.0))
                lo_r = mu_test - zq * tau_val * std_test_raw
                hi_r = mu_test + zq * tau_val * std_test_raw
            elif method == "Split Conformal (Val)":
                lo_lvl = max(0.0, np.floor((n_v + 1) * (alpha / 2.0)) / n_v)
                hi_lvl = min(1.0, np.ceil((n_v + 1) * (1.0 - alpha / 2.0)) / n_v)
                ql = np.quantile(z_val, lo_lvl, axis=0)[None, :]
                qh = np.quantile(z_val, hi_lvl, axis=0)[None, :]
                lo_r = mu_test + ql * std_test_raw
                hi_r = mu_test + qh * std_test_raw
            else:
                ql = np.zeros_like(std_test_raw)
                qh = np.zeros_like(std_test_raw)
                for i in range(n_t):
                    pos = n_v + i
                    hist = all_z[max(0, pos - rolling_window) : pos]
                    m = len(hist)
                    lo_lvl = max(0.0, np.floor((m + 1) * (alpha / 2.0)) / m)
                    hi_lvl = min(1.0, np.ceil((m + 1) * (1.0 - alpha / 2.0)) / m)
                    ql[i] = np.quantile(hist, lo_lvl, axis=0)
                    qh[i] = np.quantile(hist, hi_lvl, axis=0)
                lo_r = mu_test + ql * std_test_raw
                hi_r = mu_test + qh * std_test_raw

            lo_p = prev_p * np.exp(lo_r)
            hi_p = prev_p * np.exp(hi_r)
            emp_cov = float(np.mean((true_p >= lo_p) & (true_p <= hi_p)) * 100.0)
            mpiw = float(np.mean(hi_p - lo_p))
            wink = winkler_interval_score(true_p, lo_p, hi_p, alpha=alpha)
            rel_covs.append(emp_cov / 100.0)
            pct_key = int(round(cov_nom * 100))
            if pct_key in (80, 90, 95):
                intervals_by_cov[pct_key] = {"cov": emp_cov, "mpiw": mpiw, "winkler": wink}

        # PIT và CRPS theo phương pháp
        if method == "Gauss + Val tau":
            std_cal = tau_val * std_test_raw
            pit_mat = stats.norm.cdf(z_test / tau_val)
            pit_vals = pit_mat.flatten()
            crps_ret = float(np.mean(gaussian_crps_numpy(true_test_r, mu_test, std_cal)))
            if cov_test_cal is not None:
                sum_std = np.sqrt(np.maximum(cov_test_cal.sum(axis=(1, 2)), 1e-12))
            else:
                sum_std = np.sqrt(np.sum(std_cal ** 2, axis=1))
            crps_sum = float(np.mean(gaussian_crps_numpy(true_test_r.sum(axis=1), mu_test.sum(axis=1), sum_std)))
        elif method == "Split Conformal (Val)":
            pit_mat = np.array(
                [[np.mean(z_val[:, k] <= z_test[i, k]) for k in range(k_dim)] for i in range(n_t)],
                dtype=np.float64,
            )
            pit_vals = pit_mat.flatten()
            samples = mu_test[None, :, :] + z_val[:, None, :] * std_test_raw[None, :, :]
            crps_ret = float(np.mean(empirical_crps_from_samples(samples, true_test_r)))
            crps_sum = float(np.mean(empirical_crps_from_samples(samples.sum(axis=2), true_test_r.sum(axis=1))))
        else:
            pit_mat = np.zeros((n_t, k_dim), dtype=np.float64)
            samples_roll = np.zeros((rolling_window, n_t, k_dim), dtype=np.float64)
            for i in range(n_t):
                pos = n_v + i
                hist = all_z[max(0, pos - rolling_window) : pos]
                samples_roll[:, i, :] = mu_test[i][None, :] + hist * std_test_raw[i][None, :]
                for k in range(k_dim):
                    pit_mat[i, k] = float(np.mean(hist[:, k] <= z_test[i, k]))
            pit_vals = pit_mat.flatten()
            crps_ret = float(np.mean(empirical_crps_from_samples(samples_roll, true_test_r)))
            crps_sum = float(np.mean(empirical_crps_from_samples(samples_roll.sum(axis=2), true_test_r.sum(axis=1))))

        ks_res = stats.kstest(pit_vals, "uniform")
        diag_details[method] = {
            "pit": pit_vals,
            "pit_matrix": pit_mat,
            "nominal_grid": np.asarray(nominal_grid),
            "empirical_grid": np.asarray(rel_covs),
            "ks_stat": float(ks_res.statistic),
            "ks_p": float(ks_res.pvalue),
        }
        summary_rows.append(
            {
                "Model": model_label,
                "Calibration_Method": method,
                "Cov80 (%)": round(intervals_by_cov[80]["cov"], 1),
                "MPIW80 (VND)": round(intervals_by_cov[80]["mpiw"], 0),
                "Winkler80": round(intervals_by_cov[80]["winkler"], 0),
                "Cov90 (%)": round(intervals_by_cov[90]["cov"], 1),
                "MPIW90 (VND)": round(intervals_by_cov[90]["mpiw"], 0),
                "Winkler90": round(intervals_by_cov[90]["winkler"], 0),
                "Cov95 (%)": round(intervals_by_cov[95]["cov"], 1),
                "MPIW95 (VND)": round(intervals_by_cov[95]["mpiw"], 0),
                "Winkler95": round(intervals_by_cov[95]["winkler"], 0),
                "CRPS (ret)": round(crps_ret, 5),
                "CRPS-Sum": round(crps_sum, 5),
                "PIT_KS_stat": round(float(ks_res.statistic), 4),
                "PIT_KS_p": format_pval(float(ks_res.pvalue)),
            }
        )

    return pd.DataFrame(summary_rows), diag_details


def compute_jensen_reconstruction_comparison(
    model_label: str,
    mu_val: np.ndarray,
    true_val_r: np.ndarray,
    mu_test: np.ndarray,
    std_test_raw: np.ndarray,
    std_test_cal: np.ndarray,
    mc_prices_pool: np.ndarray,
    prev_p: np.ndarray,
    true_p: np.ndarray,
) -> pd.DataFrame:
    """
    So sánh 4 phương pháp tái tạo kỳ vọng giá từ dự báo log-return (Jensen's Inequality):
      1) Median: P_{t-1} * exp(mu_t)
      2) Log-Normal Mean (Raw): P_{t-1} * exp(mu_t + 0.5 * sigma_raw^2)
      3) Log-Normal Mean (Calibrated): P_{t-1} * exp(mu_t + 0.5 * sigma_cal^2)
      4) MC Sample Mean: trung bình mẫu Monte Carlo của P_{t-1} * exp(r^{(s)})
      5) Duan's Non-Parametric Smearing (fit thuần trên Validation, không giả định chuẩn)
    """
    duan_factor = np.mean(np.exp(true_val_r - mu_val), axis=0)[None, :]
    variants = [
        ("Median: P_{t-1}*exp(mu)", prev_p * np.exp(mu_test)),
        ("LogNormal Mean (sigma_raw): exp(mu+0.5*s2)", prev_p * np.exp(mu_test + 0.5 * (std_test_raw ** 2))),
        ("LogNormal Mean (sigma_cal): exp(mu+0.5*s2)", prev_p * np.exp(mu_test + 0.5 * (std_test_cal ** 2))),
        ("MC Sample Mean (Ensemble Pool)", np.mean(mc_prices_pool, axis=0)),
        ("Duan Non-Parametric Smearing (Val)", prev_p * np.exp(mu_test) * duan_factor),
    ]
    med_sq_err = (true_p - variants[0][1]) ** 2
    rows = []
    for name, pred_p in variants:
        err = pred_p - true_p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        mape = float(np.mean(np.abs(err / true_p)) * 100.0)
        bias = float(np.mean(err))
        dm_r = diebold_mariano_hac(med_sq_err, err ** 2, h=1, max_lag=5, aggregate_provinces=True)
        rows.append(
            {
                "Model": model_label,
                "Reconstruction_Rule": name,
                "RMSE (VND)": round(rmse, 2),
                "MAE (VND)": round(mae, 2),
                "MAPE (%)": round(mape, 4),
                "Mean_Bias (hat-true VND)": round(bias, 2),
                "DM_vs_Median_stat": round(dm_r["dm_stat"], 3),
                "DM_vs_Median_p2s": format_pval(dm_r["p_2s"]),
            }
        )
    return pd.DataFrame(rows)


def student_t_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Tính khoảng tin cậy Student-t trên M seeds (df = M - 1)."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(n - 1, 1)))
    hw = t_crit * std_v / math.sqrt(max(n, 1))
    return mean_v, std_v, mean_v - hw, mean_v + hw


def block_bootstrap_metric_ci(
    true_p: np.ndarray,
    pred_p: np.ndarray,
    true_r: np.ndarray,
    mu_r: np.ndarray,
    crps_mat: np.ndarray,
    cov95_mat: np.ndarray,
    block_len: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Circular Moving-Block Bootstrap 95% CI trên chuỗi 542 ngày Test (bảo toàn phụ thuộc thời gian)."""
    rng = np.random.default_rng(seed)
    n_days = len(true_p)
    n_blocks = int(math.ceil(n_days / block_len))
    sq_err_p = (true_p - pred_p) ** 2
    abs_err_p = np.abs(true_p - pred_p)
    hit_da = (np.sign(mu_r) == np.sign(true_r)).astype(np.float64)
    nz_mask = (true_r != 0).astype(np.float64)

    rmses, maes, das, crpss, covs = [], [], [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n_days, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block_len) % n_days) for s in starts])[:n_days]
        rmses.append(float(np.sqrt(np.mean(sq_err_p[idx]))))
        maes.append(float(np.mean(abs_err_p[idx])))
        das.append(float(np.sum(hit_da[idx] * nz_mask[idx]) / max(np.sum(nz_mask[idx]), 1.0) * 100.0))
        crpss.append(float(np.mean(crps_mat[idx])))
        covs.append(float(np.mean(cov95_mat[idx]) * 100.0))

    return {
        "RMSE": (float(np.percentile(rmses, 2.5)), float(np.percentile(rmses, 97.5))),
        "MAE": (float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))),
        "DA": (float(np.percentile(das, 2.5)), float(np.percentile(das, 97.5))),
        "CRPS": (float(np.percentile(crpss, 2.5)), float(np.percentile(crpss, 97.5))),
        "Cov95": (float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))),
    }


def run_stationarity_diagnostics(bundle: CoffeeDataBundle) -> pd.DataFrame:
    """Chạy kiểm định đồng thời ADF (H0: có nghiệm đơn vị) & KPSS (H0: chuỗi dừng) + hiệu ứng ARCH."""
    import warnings
    from statsmodels.stats.diagnostic import het_arch
    from statsmodels.tsa.stattools import adfuller, kpss

    rows = []

    def _test_series(group: str, name: str, split: str, arr: np.ndarray, check_arch: bool = False) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_stat, adf_p, adf_lag, *_ = adfuller(arr, autolag="AIC")
            kpss_c_stat, kpss_c_p, *_ = kpss(arr, regression="c", nlags="auto")
            kpss_ct_stat, kpss_ct_p, *_ = kpss(arr, regression="ct", nlags="auto")
            arch_p = float(het_arch(arr, nlags=5)[1]) if check_arch and len(arr) > 30 else np.nan

        if adf_p < 0.05 and kpss_c_p >= 0.05:
            verdict = "Dừng mạnh I(0) (ADF bác bỏ I(1) & KPSS-c không bác bỏ I(0))"
        elif adf_p < 0.05 and kpss_c_p < 0.05:
            verdict = "Hồi quy về trung bình / mùa vụ (ADF p<0.05 nhưng KPSS phát hiện dao động dài/mùa vụ)"
        else:
            verdict = "Không dừng I(1) (ADF không bác bỏ unit root & KPSS bác bỏ I(0))"

        rows.append(
            {
                "Group": group,
                "Series": name,
                "Split": split,
                "N": len(arr),
                "ADF_stat": round(float(adf_stat), 3),
                "ADF_p": format_pval(float(adf_p)),
                "KPSS_c_stat": round(float(kpss_c_stat), 3),
                "KPSS_c_p": f">= 0.10" if kpss_c_p >= 0.10 else ("<= 0.01" if kpss_c_p <= 0.01 else f"{kpss_c_p:.3f}"),
                "KPSS_ct_stat": round(float(kpss_ct_stat), 3),
                "ARCH_LM5_p": format_pval(arch_p) if np.isfinite(arch_p) else "-",
                "Verdict": verdict,
            }
        )

    for prov in TARGET_PROVINCES:
        _test_series("Price Level P_t", prov, f"Full ({len(bundle.price_df)})", bundle.price_df[prov].values)
        _test_series("Log-Return r_t", prov, f"Full ({len(bundle.ret_df)})", bundle.ret_df[prov].values, check_arch=True)
        _test_series(
            "Log-Return r_t",
            prov,
            f"Train ({bundle.train_end})",
            bundle.ret_df[prov].iloc[: bundle.train_end].values,
            check_arch=True,
        )
        _test_series(
            "Log-Return r_t",
            prov,
            f"Val ({bundle.val_end - bundle.train_end})",
            bundle.ret_df[prov].iloc[bundle.train_end : bundle.val_end].values,
            check_arch=True,
        )
        _test_series(
            "Log-Return r_t",
            prov,
            f"Test ({len(bundle.ret_df) - bundle.val_end})",
            bundle.ret_df[prov].iloc[bundle.val_end :].values,
            check_arch=True,
        )

    exog_df = bundle.feature_sets["DeepVARX 3"].iloc[:, 3:]
    for col in exog_df.columns:
        _test_series("Exogenous V3", col, f"Full ({len(exog_df)})", exog_df[col].values)

    return pd.DataFrame(rows)


def run_correlation_diagnostics(bundle: CoffeeDataBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tính VIF, tương quan tối đa với biến ngoại sinh khác và ma trận tương quan đầy đủ 17 biến ngoại sinh."""
    exog_df = bundle.feature_sets["DeepVARX 3"].iloc[:, 3:]
    corr_mat = exog_df.corr()
    vals = exog_df.values.astype(np.float64)
    vals_z = (vals - vals.mean(axis=0)) / np.maximum(vals.std(axis=0), 1e-8)

    rows = []
    cols = list(exog_df.columns)
    for j, col in enumerate(cols):
        other = np.delete(vals_z, j, axis=1)
        y = vals_z[:, j]
        coef, *_ = np.linalg.lstsq(other, y, rcond=None)
        r2 = float(1.0 - np.sum((y - other @ coef) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
        vif = 1.0 / max(1.0 - r2, 1e-6)
        c_row = corr_mat[col].drop(col)
        top_partner = str(c_row.abs().idxmax())
        rows.append(
            {
                "Feature": col,
                "VIF": round(vif, 3),
                "Top_Correlated_Partner": top_partner,
                "Pearson_r": round(float(c_row.loc[top_partner]), 4),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    return summary_df, corr_mat


def multivariate_dm_hac(
    loss_bench: np.ndarray,
    loss_model: np.ndarray,
    max_lag: int = 5,
) -> Dict[str, float]:
    """
    Kiểm định Diebold-Mariano đa biến vector (Mariano & Preve, 2012) cho K=3 tỉnh đồng thời,
    không giả định trọng số bằng nhau hay độc lập liên tỉnh:
      W_MDM = N * d_bar^T * Omega_HAC^{-1} * d_bar ~ chi2(K), F = (N-K)/(K*(N-1)) * W_MDM ~ F(K, N-K).
    """
    D = (loss_bench - loss_model).astype(np.float64)  # (N, K)
    n, k_dim = D.shape
    d_bar = np.mean(D, axis=0)  # (K,)
    Dc = D - d_bar[None, :]
    omega = (Dc.T @ Dc) / n
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        gamma_k = (Dc[lag:].T @ Dc[:-lag]) / n
        omega = omega + w * (gamma_k + gamma_k.T)
    omega = omega + 1e-12 * np.eye(k_dim)
    wald = float(n * (d_bar.T @ np.linalg.solve(omega, d_bar)))
    f_stat = float(((n - k_dim) / (k_dim * max(n - 1, 1))) * wald)
    p_chi2 = float(1.0 - stats.chi2.cdf(wald, df=k_dim))
    p_f = float(1.0 - stats.f.cdf(f_stat, dfn=k_dim, dfd=max(n - k_dim, 1)))
    return {
        "MDM_Wald_chi2": wald,
        "MDM_F_stat": f_stat,
        "MDM_p_chi2": p_chi2,
        "MDM_p_F": p_f,
    }


def pit_independence_aware_tests(pit_1d: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
    """
    Kiểm định PIT cho từng chuỗi N=542 đơn lẻ (không gộp 3 tỉnh) có tính đến tự tương quan:
      1) KS đơn biến trên N=542
      2) Ljung-Box Q(5) trên (u_t - 0.5) & (u_t - 0.5)^2 (Diebold, Gunther & Tay, 1998)
      3) Berkowitz (2001) AR(1) Likelihood-Ratio LR3 ~ chi2(3) trên z_t = Phi^{-1}(u_t),
         kiểm định đồng thời (mu=0, sigma^2=1, rho_AR1=0).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    u = np.asarray(pit_1d, dtype=np.float64)
    ks_res = stats.kstest(u, "uniform")
    lb_mean = float(acorr_ljungbox(u - 0.5, lags=[max_lag], return_df=True)["lb_pvalue"].iloc[0])
    lb_var = float(acorr_ljungbox((u - 0.5) ** 2, lags=[max_lag], return_df=True)["lb_pvalue"].iloc[0])

    z_b = stats.norm.ppf(np.clip(u, 1e-5, 1.0 - 1e-5))
    y_ar = z_b[1:]
    x_ar = np.column_stack([np.ones(len(y_ar)), z_b[:-1]])
    coef, *_ = np.linalg.lstsq(x_ar, y_ar, rcond=None)
    resid = y_ar - x_ar @ coef
    sigma2_hat = max(float(np.mean(resid ** 2)), 1e-12)
    lr3 = max(0.0, float(np.sum(y_ar ** 2) - len(y_ar) * (1.0 + math.log(sigma2_hat))))
    berk_p = float(1.0 - stats.chi2.cdf(lr3, df=3))

    return {
        "PIT_Mean": float(np.mean(u)),
        "PIT_Std": float(np.std(u)),
        "KS_stat_N542": float(ks_res.statistic),
        "KS_p_N542": float(ks_res.pvalue),
        "AR1_rho": float(coef[1]),
        "Berkowitz_LR3": lr3,
        "Berkowitz_p": berk_p,
        "LB_Q5_mean_p": lb_mean,
        "LB_Q5_var_p": lb_var,
    }


def compute_province_calibration_table(
    model_specs: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]],
    rolling_window: int = 120,
) -> pd.DataFrame:
    """
    Báo cáo hiệu chỉnh xác suất RIÊNG TỪNG TỈNH (đắk lắk, lâm đồng, gia lai; N=542 mỗi tỉnh)
    với đủ Cov 80/90/95%, MPIW, Winkler Score, CRPS, PIT KS (N=542), Berkowitz AR(1) LR và Ljung-Box.
    """
    rows = []
    for m_label, mu_val, std_val_raw, true_val_r, mu_te, std_te_raw, tau_val, true_te_r, prev_p, true_p in model_specs:
        for p_idx, prov in enumerate(TARGET_PROVINCES):
            df_p, diag_p = compute_calibration_suite(
                model_label=f"{m_label} | {prov}",
                mu_val=mu_val[:, p_idx : p_idx + 1],
                std_val_raw=std_val_raw[:, p_idx : p_idx + 1],
                true_val_r=true_val_r[:, p_idx : p_idx + 1],
                mu_test=mu_te[:, p_idx : p_idx + 1],
                std_test_raw=std_te_raw[:, p_idx : p_idx + 1],
                tau_val=tau_val,
                true_test_r=true_te_r[:, p_idx : p_idx + 1],
                prev_p=prev_p[:, p_idx : p_idx + 1],
                true_p=true_p[:, p_idx : p_idx + 1],
                cov_test_cal=None,
                rolling_window=rolling_window,
            )
            for method_name in ["Gauss + Val tau", f"Causal Rolling Conformal (W={rolling_window})"]:
                r_m = df_p[df_p["Calibration_Method"] == method_name].iloc[0]
                pit_1d = diag_p[method_name]["pit_matrix"][:, 0]
                pt = pit_independence_aware_tests(pit_1d, max_lag=5)
                rows.append(
                    {
                        "Model": m_label,
                        "Province": prov,
                        "Calibration_Method": method_name,
                        "Cov80 (%)": r_m["Cov80 (%)"],
                        "MPIW80": r_m["MPIW80 (VND)"],
                        "Cov90 (%)": r_m["Cov90 (%)"],
                        "MPIW90": r_m["MPIW90 (VND)"],
                        "Cov95 (%)": r_m["Cov95 (%)"],
                        "MPIW95": r_m["MPIW95 (VND)"],
                        "Winkler95": r_m["Winkler95"],
                        "CRPS (ret)": r_m["CRPS (ret)"],
                        "KS_p (N=542)": format_pval(pt["KS_p_N542"]),
                        "AR1_rho": round(pt["AR1_rho"], 3),
                        "Berkowitz_p": format_pval(pt["Berkowitz_p"]),
                        "LB_mean_p": format_pval(pt["LB_Q5_mean_p"]),
                        "LB_var_p": format_pval(pt["LB_Q5_var_p"]),
                    }
                )
    return pd.DataFrame(rows)


def _compute_rolling_conformal_band_and_pit(
    mu_val: np.ndarray,
    std_val_raw: np.ndarray,
    true_val_r: np.ndarray,
    mu_test: np.ndarray,
    std_test_raw: np.ndarray,
    true_test_r: np.ndarray,
    prev_p: np.ndarray,
    rolling_window: int,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hàm lõi tính khoảng dự báo Causal Rolling Conformal & PIT theo từng bước nhân quả."""
    z_val = (true_val_r - mu_val) / np.maximum(std_val_raw, 1e-8)
    z_te = (true_test_r - mu_test) / np.maximum(std_test_raw, 1e-8)
    all_z = np.vstack([z_val, z_te])
    n_v, n_t, k_dim = len(z_val), len(z_te), z_te.shape[1]
    ql = np.zeros_like(std_test_raw)
    qh = np.zeros_like(std_test_raw)
    pit_mat = np.zeros((n_t, k_dim), dtype=np.float64)
    max_hist_pos = np.zeros(n_t, dtype=np.int64)
    for i in range(n_t):
        pos = n_v + i
        hist = all_z[max(0, pos - rolling_window) : pos]
        max_hist_pos[i] = pos - 1
        m = len(hist)
        ql[i] = np.quantile(hist, max(0.0, np.floor((m + 1) * (alpha / 2.0)) / m), axis=0)
        qh[i] = np.quantile(hist, min(1.0, np.ceil((m + 1) * (1.0 - alpha / 2.0)) / m), axis=0)
        for k in range(k_dim):
            pit_mat[i, k] = float(np.mean(hist[:, k] <= z_te[i, k]))
    lo_p = prev_p * np.exp(mu_test + ql * std_test_raw)
    hi_p = prev_p * np.exp(mu_test + qh * std_test_raw)
    return lo_p, hi_p, pit_mat, max_hist_pos


def verify_rolling_conformal_zero_leakage(
    model_label: str,
    mu_val: np.ndarray,
    std_val_raw: np.ndarray,
    true_val_r: np.ndarray,
    mu_test: np.ndarray,
    std_test_raw: np.ndarray,
    true_test_r: np.ndarray,
    prev_p: np.ndarray,
    true_p: np.ndarray,
    dates_all: pd.DatetimeIndex,
    train_end: int,
    rolling_windows: List[int] = [60, 90, 120, 180],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Chứng minh nhân quả tuyệt đối (Zero Leakage) bằng kiểm thử đối kháng Future-Poisoning:
      - Xác minh mọi chỉ số lịch sử sử dụng cho ngày Test t thỏa điều kiện timestamp <= t - 1.
      - Nhiễu hóa toàn bộ thực tế tương lai true_test_r[t_check:] -> +500.0 và xác nhận
        khoảng dự báo [lo[t_check], hi[t_check]] bất biến tuyệt đối (sai khác = 0.000000).
      - Kiểm tra độ nhạy trên 4 độ dài cửa sổ W = {60, 90, 120, 180}.
    """
    n_v, n_t = len(true_val_r), len(true_test_r)
    lo_orig, hi_orig, _, max_hist_pos = _compute_rolling_conformal_band_and_pit(
        mu_val, std_val_raw, true_val_r, mu_test, std_test_raw, true_test_r, prev_p, rolling_window=120, alpha=0.05
    )
    target_positions = np.arange(n_v, n_v + n_t, dtype=np.int64)
    causal_gap_min = int(np.min(target_positions - max_hist_pos))
    assert causal_gap_min == 1, f"VI PHẠM NHÂN QUẢ: khoảng cách thời gian lịch sử tối thiểu = {causal_gap_min}"

    test_dates_slice = dates_all[train_end + n_v :]
    val_and_test_dates = dates_all[train_end:]
    assert all(val_and_test_dates[max_hist_pos[i]] < test_dates_slice[i] for i in range(n_t))

    max_poison_diff = 0.0
    for t_chk in [0, 120, 350, n_t - 1]:
        poison_r = true_test_r.copy()
        poison_r[t_chk:] = 500.0
        lo_p, hi_p, _, _ = _compute_rolling_conformal_band_and_pit(
            mu_val, std_val_raw, true_val_r, mu_test, std_test_raw, poison_r, prev_p, rolling_window=120, alpha=0.05
        )
        diff = max(
            float(np.max(np.abs(lo_p[t_chk] - lo_orig[t_chk]))),
            float(np.max(np.abs(hi_p[t_chk] - hi_orig[t_chk]))),
        )
        max_poison_diff = max(max_poison_diff, diff)
    assert max_poison_diff == 0.0, f"PHÁT HIỆN RÒ RỈ TƯƠNG LAI: max_diff = {max_poison_diff}"

    # Độ nhạy cửa sổ W = {60, 90, 120, 180}
    win_rows = []
    for w in rolling_windows:
        df_w, diag_w = compute_calibration_suite(
            model_label=model_label,
            mu_val=mu_val, std_val_raw=std_val_raw, true_val_r=true_val_r,
            mu_test=mu_test, std_test_raw=std_test_raw, tau_val=1.0,
            true_test_r=true_test_r, prev_p=prev_p, true_p=true_p, rolling_window=w,
        )
        rw = df_w[df_w["Calibration_Method"] == f"Causal Rolling Conformal (W={w})"].iloc[0]
        pit_dl = pit_independence_aware_tests(diag_w[f"Causal Rolling Conformal (W={w})"]["pit_matrix"][:, 0])
        win_rows.append(
            {
                "Model": model_label,
                "Window_W": w,
                "Cov80 (%)": rw["Cov80 (%)"],
                "Cov90 (%)": rw["Cov90 (%)"],
                "Cov95 (%)": rw["Cov95 (%)"],
                "MPIW95 (VND)": rw["MPIW95 (VND)"],
                "Winkler95": rw["Winkler95"],
                "CRPS (ret)": rw["CRPS (ret)"],
                "DakLak_KS_p": format_pval(pit_dl["KS_p_N542"]),
                "DakLak_Berkowitz_p": format_pval(pit_dl["Berkowitz_p"]),
                "ZeroLeak_Diff": f"{max_poison_diff:.1e}",
            }
        )

    audit_info = {
        "causal_gap_days": causal_gap_min,
        "max_poison_diff": max_poison_diff,
        "status": "PASS (Strict t-1 causal history & Future-Poison invariant)",
    }
    return audit_info, pd.DataFrame(win_rows)


def paired_block_bootstrap_diagnostics(
    true_p: np.ndarray,
    preds_map: Dict[str, Dict],
    exog_london_lag1: np.ndarray,
    block_lengths: List[int] = [5, 10, 20],
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Circular Moving-Block Bootstrap ĐỒNG BỘ theo ngày t trên tất cả 3 tỉnh & biến ngoại sinh trễ:
      1) Định lượng mức độ bảo toàn ma trận tương quan liên tỉnh & phụ thuộc sốc ngoại sinh.
      2) Báo cáo cả Marginal RMSE CI lẫn Paired Differential CI (Delta RMSE, Delta CRPS vs Naive),
         giải thích lý do CI biên rộng do nhạy chế độ biến động thị trường 2024-2026.
    """
    rng = np.random.default_rng(seed)
    n_days = len(true_p)

    # Kiểm chứng bảo toàn tương quan đa tỉnh & tương quan sốc ngoại sinh trên Naive error & DVX3
    ref_err = true_p - preds_map["DeepVARX 3"]["pred_p"]
    orig_corr_dl_ld = float(np.corrcoef(ref_err[:, 0], ref_err[:, 1])[0, 1])
    orig_corr_dl_gl = float(np.corrcoef(ref_err[:, 0], ref_err[:, 2])[0, 1])
    orig_corr_exog = float(np.corrcoef(np.abs(ref_err[:, 0]), np.abs(exog_london_lag1))[0, 1])

    struct_rows = []
    paired_rows = []

    naive_sq = (true_p - preds_map["Naive Prob"]["pred_p"]) ** 2
    naive_crps = preds_map["Naive Prob"]["crps_mat"]

    for b_len in block_lengths:
        n_blocks = int(math.ceil(n_days / b_len))
        b_corr_dl_ld, b_corr_dl_gl, b_corr_exog = [], [], []
        idx_draws = []
        for _ in range(n_boot):
            starts = rng.integers(0, n_days, size=n_blocks)
            idx = np.concatenate([(np.arange(s, s + b_len) % n_days) for s in starts])[:n_days]
            idx_draws.append(idx)
            b_corr_dl_ld.append(float(np.corrcoef(ref_err[idx, 0], ref_err[idx, 1])[0, 1]))
            b_corr_dl_gl.append(float(np.corrcoef(ref_err[idx, 0], ref_err[idx, 2])[0, 1]))
            b_corr_exog.append(float(np.corrcoef(np.abs(ref_err[idx, 0]), np.abs(exog_london_lag1[idx]))[0, 1]))

        struct_rows.append(
            {
                "Block_Len_B": b_len,
                "Orig_Corr_DL_LD": round(orig_corr_dl_ld, 4),
                "Boot_Mean_Corr_DL_LD": round(float(np.mean(b_corr_dl_ld)), 4),
                "Orig_Corr_DL_GL": round(orig_corr_dl_gl, 4),
                "Boot_Mean_Corr_DL_GL": round(float(np.mean(b_corr_dl_gl)), 4),
                "Orig_Corr_|Err|_vs_|LondonLag1|": round(orig_corr_exog, 4),
                "Boot_Mean_Corr_|Err|_vs_|LondonLag1|": round(float(np.mean(b_corr_exog)), 4),
            }
        )

        for mname, d_m in preds_map.items():
            sq_m = (true_p - d_m["pred_p"]) ** 2
            crps_m = d_m["crps_mat"]
            marg_rmse, d_rmse_vs_rw, d_crps_vs_rw = [], [], []
            for idx in idx_draws:
                r_m = float(np.sqrt(np.mean(sq_m[idx])))
                r_rw = float(np.sqrt(np.mean(naive_sq[idx])))
                marg_rmse.append(r_m)
                d_rmse_vs_rw.append(r_rw - r_m)
                d_crps_vs_rw.append(float(np.mean(naive_crps[idx]) - np.mean(crps_m[idx])))
            paired_rows.append(
                {
                    "Block_Len_B": b_len,
                    "Model": mname,
                    "Point_RMSE": round(float(np.sqrt(np.mean(sq_m))), 2),
                    "Marginal_RMSE_95CI": f"[{np.percentile(marg_rmse, 2.5):.2f}, {np.percentile(marg_rmse, 97.5):.2f}]",
                    "Paired_DeltaRMSE_vs_Naive_Mean": round(float(np.mean(d_rmse_vs_rw)), 2),
                    "Paired_DeltaRMSE_vs_Naive_95CI": f"[{np.percentile(d_rmse_vs_rw, 2.5):.2f}, {np.percentile(d_rmse_vs_rw, 97.5):.2f}]",
                    "Paired_DeltaCRPS_vs_Naive_95CI": f"[{np.percentile(d_crps_vs_rw, 2.5):.5f}, {np.percentile(d_crps_vs_rw, 97.5):.5f}]",
                }
            )

    return pd.DataFrame(struct_rows), pd.DataFrame(paired_rows)

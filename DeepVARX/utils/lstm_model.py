# -*- coding: utf-8 -*-
"""
========================================================================================
MÔ-ĐUN MÔ HÌNH LSTM DỰ BÁO GIÁ CÀ PHÊ ROBUSTA VIỆT NAM (3 TỈNH TÂY NGUYÊN)
========================================================================================
Xây dựng dựa trên tổng hợp cơ sở lý thuyết và thực nghiệm từ các nguồn tài liệu:
  1. Hochreiter & Schmidhuber (1997): Kiến trúc LSTM giải quyết vanishing gradient.
  2. MachineLearningMastery (PyTorch LSTM for Time Series): Thiết kế sliding-window
     nn.LSTM(batch_first=True) + nn.Linear trên bước cuối out[:, -1, :].
  3. Beck, Dovern & Vogl (2025) - "Mind the naive forecast!": Chứng minh chuỗi giá tài
     chính/nông sản biến động mạnh rất khó vượt Naive (P_hat_t = P_{t-1}) nếu chỉ dùng
     giá quá khứ hoặc mất tính dừng.
  4. Kadir et al. (Forecasting, 2026): Đánh giá leak-free walk-forward trên Robusta,
     khẳng định vai trò thống trị của độ trễ bậc 1 và thiết lập kiến trúc 1 lớp LSTM
     (64 units, tanh + linear, batch=16, Adam lr=1e-3, epochs=30-50, không quá sâu).
  5. Mai, Shakur & Cassells (2018): Cơ chế truyền dẫn giá dọc (vertical price
     transmission) giữa sàn xuất khẩu/London và giá nông hộ Đắk Lắk diễn ra trong
     vòng vài ngày (độ trễ T+1 rất mạnh).
  6. O'Brien (2007) & Vatcheva et al. (2016): Kiểm soát đa cộng tuyến VIF khi tuyển
     chọn tập biến ngoại sinh phù hợp cho mô hình đa biến.

Ba phiên bản thực nghiệm chính (dự báo log-return dừng neo trên Naive P_{t-1},
theo chiến lược Holdout-trained Rolling One-step-ahead h=1):
  - LSTM 1 (Baseline)        : Chỉ dùng giá cà phê quá khứ (3 tỉnh).
  - LSTM 2 (Multivariate)    : Giá quá khứ + Thời tiết cơ bản + Kinh tế cơ bản.
  - LSTM 3 (Full Exogenous)  : Giá quá khứ + Toàn bộ biến ngoại sinh được chọn theo kiến thức
                               miền & kiểm chứng chẩn đoán trên Train (loại temp_max/temp_min
                               đa cộng tuyến cao, sai phân bậc 1 soil_moisture_top, log-return
                               chuỗi I(1); lưu ý cos_day_of_year có VIF=10.05 > 10 do tương quan
                               với chỉ báo mùa vụ và dữ liệu vào qua cửa sổ nhân quả lag >= 1).
========================================================================================
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
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
    """Cố định toàn bộ seed ngẫu nhiên để đảm bảo khả năng tái lập thực nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class CoffeeDataBundle:
    """Cấu trúc chứa dữ liệu đã đồng bộ, chuỗi dừng, và các tập đặc trưng LSTM 1/2/3."""

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
    Nạp dữ liệu panel và xây dựng chuỗi dừng + tập đặc trưng cho LSTM 1, 2, 3.
    Mọi biến tại dòng thời gian s chỉ dùng thông tin ngày s; khi tạo cửa sổ trượt dự báo
    ngày t, mô hình chỉ nhận [t - seq_len, ..., t - 1] nên tuyệt đối không rò rỉ tương lai.
    """
    raw_df = pd.read_csv(csv_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    price_raw = (
        raw_df[raw_df["province_slug"].isin(TARGET_PROVINCES)]
        .pivot(index="date", columns="province_slug", values="price_vnd_kg")[TARGET_PROVINCES]
        .dropna()
        .sort_index()
    )

    # Biến mục tiêu dừng I(0): tỷ suất sinh lợi logarit r_t = ln(P_t) - ln(P_{t-1})
    ret_df = np.log(price_raw).diff().dropna()
    dates = ret_df.index
    price_df = price_raw.loc[dates]

    # Biến kinh tế/vĩ mô: chuyển chuỗi giá/tỷ giá I(1) sang log-return dừng I(0)
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

    # Biến thời tiết trung bình vùng trồng (loại bỏ temp_max/temp_min đa cộng tuyến cao,
    # sai phân bậc 1 soil_moisture_top vì chuỗi gốc không dừng ở mức 5%, ADF Train p=0.0614)
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

    # Biến mùa vụ, lịch thu hoạch và chỉ báo ngày sàn quốc tế đóng cửa
    cal_df = raw_df.groupby("date")[CALENDAR_FRICTION_COLS].mean().loc[dates]

    # Định nghĩa 3 tập đầu vào tương ứng với 3 phiên bản LSTM
    x_lstm1 = ret_df.copy()

    x_lstm2 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS],
            econ_ret_df[["ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd"]],
        ],
        axis=1,
    )

    x_lstm3 = pd.concat(
        [
            ret_df,
            weather_mean_df[CORE_WEATHER_COLS + EXT_WEATHER_COLS],
            econ_ret_df,
            cal_df,
        ],
        axis=1,
    )

    exog_unscreened = pd.concat(
        [
            weather_mean_df[raw_weather_cols],
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
            "LSTM 1": x_lstm1,
            "LSTM 2": x_lstm2,
            "LSTM 3": x_lstm3,
        },
        exog_unscreened=exog_unscreened,
    )


def compute_diagnostics(bundle: CoffeeDataBundle) -> Dict[str, pd.DataFrame]:
    """
    Thực hiện các kiểm định chẩn đoán dữ liệu CHỈ trên tập Train (không chạm Validation/Test):
      1. Thống kê phân bổ giá qua Train / Val / Test (minh họa dịch chuyển phân phối).
      2. Kiểm định tính dừng ADF cho cả 3 tỉnh mục tiêu và toàn bộ biến ngoại sinh LSTM 3
         (lưu ý các biến mùa vụ sin/cos/harvest là hàm tất định theo lịch, không phải chuỗi I(1)).
      3. Kiểm định đa cộng tuyến VIF trước và sau sàng lọc biến ngoại sinh (báo cáo minh bạch
         cờ VIF_Below_10 thay vì mặc định mọi biến đều < 10).
      4. Tương quan dẫn dắt/đồng thời/trước-sau (same-day t, lag t-1, t-2, lead t+1) trên Train
         để kiểm tra tính nhân quả theo thời gian.

    Lưu ý: Các chẩn đoán ADF/VIF ở đây dùng để kiểm chứng và báo cáo tính hợp lệ của bộ biến
    đã thiết kế sẵn trên tập Train, không thực hiện lọc biến tự động trên Validation/Test.
    """
    tr_end = bundle.train_end
    va_end = bundle.val_end

    # 1. Phân bổ giá qua 3 tập
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
                    "Start": str(p_slice.index.min().date()),
                    "End": str(p_slice.index.max().date()),
                    "Price_Min": float(p_slice[prov].min()),
                    "Price_Max": float(p_slice[prov].max()),
                    "Price_Mean": float(p_slice[prov].mean()),
                    "Return_Std": float(r_slice[prov].std()),
                    "Return_AC1": float(r_slice[prov].autocorr(1)),
                }
            )
    split_stats_df = pd.DataFrame(split_rows)

    # 2. Kiểm định ADF trên Train (đầy đủ 3 tỉnh mục tiêu + toàn bộ ngoại sinh)
    dl = (
        bundle.raw_df[bundle.raw_df["province_slug"] == "dak_lak"]
        .sort_values("date")
        .set_index("date")
        .loc[bundle.dates]
    )
    train_dl = dl.iloc[:tr_end]
    train_exog_clean = bundle.feature_sets["LSTM 3"].iloc[:tr_end]

    adf_candidates = []
    for prov in TARGET_PROVINCES:
        adf_candidates.append((f"price_vnd_kg ({prov})", bundle.price_df[prov].iloc[:tr_end], "Target_Level"))
        adf_candidates.append((f"ret_price ({prov})", bundle.ret_df[prov].iloc[:tr_end], "Target_Return"))

    adf_candidates.extend(
        [
            ("london_price_usd_ton", train_dl["london_price_usd_ton"], "Exog_Level"),
            ("ret_london", train_exog_clean["ret_london"], "Exog_Return"),
            ("arabica_price_usd_lb", train_dl["arabica_price_usd_lb"], "Exog_Level"),
            ("ret_arabica", train_exog_clean["ret_arabica"], "Exog_Return"),
            ("brent_price_usd_barrel", train_dl["brent_price_usd_barrel"], "Exog_Level"),
            ("ret_brent", train_exog_clean["ret_brent"], "Exog_Return"),
            ("dxy_index", train_dl["dxy_index"], "Exog_Level"),
            ("ret_dxy", train_exog_clean["ret_dxy"], "Exog_Return"),
            ("usd_vnd_rate", train_dl["usd_vnd_rate"], "Exog_Level"),
            ("ret_usd_vnd", train_exog_clean["ret_usd_vnd"], "Exog_Return"),
            ("usd_brl_rate", train_dl["usd_brl_rate"], "Exog_Level"),
            ("ret_usd_brl", train_exog_clean["ret_usd_brl"], "Exog_Return"),
            ("temp_mean", train_exog_clean["temp_mean"], "Weather"),
            ("precip_nasa", train_exog_clean["precip_nasa"], "Weather"),
            ("humidity_air", train_exog_clean["humidity_air"], "Weather"),
            ("soil_moisture_top (raw)", bundle.exog_unscreened["soil_moisture_top"].iloc[:tr_end], "Weather_Raw"),
            ("d_soil_moisture (diff)", train_exog_clean["d_soil_moisture"], "Weather_Diff"),
            ("solar_radiation", train_exog_clean["solar_radiation"], "Weather"),
            ("wind_speed_2m", train_exog_clean["wind_speed_2m"], "Weather"),
            ("sin_day_of_year", train_exog_clean["sin_day_of_year"], "Calendar_Deterministic"),
            ("cos_day_of_year", train_exog_clean["cos_day_of_year"], "Calendar_Deterministic"),
            ("is_harvest_season_vn", train_exog_clean["is_harvest_season_vn"], "Calendar_Deterministic"),
            ("is_harvest_season_brazil", train_exog_clean["is_harvest_season_brazil"], "Calendar_Deterministic"),
            ("exog_stale_days", train_exog_clean["exog_stale_days"], "Market_Friction"),
        ]
    )
    adf_rows = []
    for label, series, role in adf_candidates:
        stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
        adf_rows.append(
            {
                "Variable": label,
                "Role": role,
                "ADF_Stat": round(float(stat), 3),
                "p_value": round(float(pval), 4),
                "Stationary_5pct": bool(pval < 0.05),
            }
        )
    adf_df = pd.DataFrame(adf_rows)

    # 3. Kiểm định VIF trước và sau sàng lọc trên Train
    def _calc_vif(df_in: pd.DataFrame) -> Dict[str, float]:
        mat = np.column_stack([np.ones(len(df_in)), df_in.values])
        return {col: float(variance_inflation_factor(mat, i + 1)) for i, col in enumerate(df_in.columns)}

    vif_before = _calc_vif(bundle.exog_unscreened.iloc[:tr_end])
    exog_screened = bundle.feature_sets["LSTM 3"].drop(columns=TARGET_PROVINCES).iloc[:tr_end]
    vif_after = _calc_vif(exog_screened)
    all_vif_cols = list(dict.fromkeys(list(vif_before.keys()) + list(vif_after.keys())))
    vif_rows = []
    for c in all_vif_cols:
        v_aft = round(vif_after[c], 2) if c in vif_after else np.nan
        vif_rows.append(
            {
                "Exogenous_Feature": c,
                "VIF_Before_Screening": round(vif_before[c], 2) if c in vif_before else np.nan,
                "VIF_After_Screening": v_aft,
                "VIF_Below_10": bool(v_aft < 10.0) if pd.notna(v_aft) else False,
                "Kept_in_LSTM3": c in vif_after,
            }
        )
    vif_df = pd.DataFrame(vif_rows)

    # 4. Tương quan trễ T+1 (và t, t-2, lead t+1 kiểm tra rò rỉ) trên Train (Đắk Lắk)
    y_dl_tr = bundle.ret_df["dak_lak"].iloc[:tr_end]
    x_tr = exog_screened
    corr_rows = []
    for c in x_tr.columns:
        corr_rows.append(
            {
                "Feature": c,
                "Corr_Lead1_t_plus_1": round(float(y_dl_tr.corr(x_tr[c].shift(-1))), 4),
                "Corr_SameDay_t": round(float(y_dl_tr.corr(x_tr[c])), 4),
                "Corr_Lag1_t_minus_1": round(float(y_dl_tr.corr(x_tr[c].shift(1))), 4),
                "Corr_Lag2_t_minus_2": round(float(y_dl_tr.corr(x_tr[c].shift(2))), 4),
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    return {
        "split_stats": split_stats_df,
        "adf": adf_df,
        "vif": vif_df,
        "lead_lag_corr": corr_df,
    }


class CoffeeLSTM(nn.Module):
    """
    Mạng LSTM dự báo chuỗi thời gian đa biến theo chuẩn PyTorch:
      Input (B, seq_len, in_dim) -> [Input Dropout] -> LSTM (tanh gates)
      -> last time step out[:, -1, :] -> [Output Dropout] -> Linear(hidden_dim, out_dim).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        out_dim: int = 3,
        dropout: float = 0.0,
        in_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout) if in_dropout > 0.0 else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_drop(x)
        out, _ = self.lstm(x)
        last_hidden = self.out_drop(out[:, -1, :])
        return self.head(last_hidden)


def build_causal_windows(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    seq_len: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Tạo cặp (cửa sổ đầu vào, nhãn mục tiêu) trên đoạn chỉ số [start_idx, end_idx).
    Với mỗi ngày mục tiêu t, cửa sổ đầu vào luôn là [t - seq_len, ..., t - 1] (lag >= 1).
    """
    effective_start = max(start_idx, seq_len)
    xs, ys, indices = [], [], []
    for t in range(effective_start, end_idx):
        xs.append(x_scaled[t - seq_len : t])
        ys.append(y_scaled[t])
        indices.append(t)
    return (
        torch.tensor(np.asarray(xs), dtype=torch.float32),
        torch.tensor(np.asarray(ys), dtype=torch.float32),
        np.asarray(indices, dtype=np.int64),
    )


def fit_single_lstm_seed(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    hidden_dim: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    in_dropout: float = 0.0,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 10,
    scaler_type: str = "standard",
    scale_target: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Huấn luyện 1 lượt (1 seed) cho mô hình LSTM trên chuỗi dừng, sử dụng Early Stopping
    trên Validation và đánh giá cả trên Validation lẫn Test (không rò rỉ dữ liệu).
    """
    set_seed(seed)
    X_arr = X_df.values.astype(np.float32)
    Y_arr = Y_df.values.astype(np.float32)

    ScalerCls = StandardScaler if scaler_type == "standard" else MinMaxScaler
    x_scaler = ScalerCls().fit(X_arr[:train_end])
    X_s = x_scaler.transform(X_arr).astype(np.float32)

    if scale_target:
        y_scaler = ScalerCls().fit(Y_arr[:train_end])
        Y_s = y_scaler.transform(Y_arr).astype(np.float32)
    else:
        y_scaler = None
        Y_s = Y_arr

    X_tr, Y_tr, _ = build_causal_windows(X_s, Y_s, seq_len, seq_len, train_end)
    X_va, Y_va, va_idx = build_causal_windows(X_s, Y_s, seq_len, train_end, val_end)
    X_te, Y_te, te_idx = build_causal_windows(X_s, Y_s, seq_len, val_end, len(Y_arr))

    model = CoffeeLSTM(
        in_dim=X_df.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=Y_df.shape[1],
        dropout=dropout,
        in_dropout=in_dropout,
    )
    OptCls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
    optimizer = OptCls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    pat_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        train_losses.append(ep_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_pred_s = model(X_va)
            v_loss = loss_fn(val_pred_s, Y_va).item()
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
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
        val_pred_raw = model(X_va).numpy()
        test_pred_raw = model(X_te).numpy()

    val_pred_ret = y_scaler.inverse_transform(val_pred_raw) if y_scaler is not None else val_pred_raw
    test_pred_ret = y_scaler.inverse_transform(test_pred_raw) if y_scaler is not None else test_pred_raw

    # Khôi phục giá 1-step ahead neo trên giá hôm trước P_{t-1}: P_hat_t = P_{t-1} * exp(r_hat_t)
    val_prev_p = price_df.shift(1).iloc[va_idx].values
    val_true_p = price_df.iloc[va_idx].values
    val_pred_p = val_prev_p * np.exp(val_pred_ret)

    test_prev_p = price_df.shift(1).iloc[te_idx].values
    test_true_p = price_df.iloc[te_idx].values
    test_pred_p = test_prev_p * np.exp(test_pred_ret)

    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "va_idx": va_idx,
        "te_idx": te_idx,
        "X_te_tensor": X_te,
        "val_pred_ret": val_pred_ret,
        "val_true_ret": Y_df.iloc[va_idx].values,
        "val_pred_price": val_pred_p,
        "val_true_price": val_true_p,
        "test_pred_ret": test_pred_ret,
        "test_true_ret": Y_df.iloc[te_idx].values,
        "test_pred_price": test_pred_p,
        "test_true_price": test_true_p,
        "test_prev_price": test_prev_p,
    }


def evaluate_raw_price_level_lstm(
    price_df: pd.DataFrame,
    train_end: int,
    val_end: int,
    seq_len: int = 10,
    hidden_dim: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    scaler_type: str = "minmax",
    seed: int = 42,
) -> Dict:
    """
    Thực nghiệm đối chứng: Áp dụng trực tiếp quy trình MachineLearningMastery PyTorch
    trên giá mức danh nghĩa P_t (KHÔNG sai phân/log-return) để kiểm chứng nguyên nhân
    mất ổn định do dịch chuyển miền giá (Train 28k-50k vs Test 84k-139k) khiến Naive
    vượt xa mô hình LSTM thô.
    """
    set_seed(seed)
    P_arr = price_df.values.astype(np.float32)
    ScalerCls = MinMaxScaler if scaler_type == "minmax" else StandardScaler
    p_scaler = ScalerCls().fit(P_arr[:train_end])
    P_s = p_scaler.transform(P_arr).astype(np.float32)

    X_tr, Y_tr, _ = build_causal_windows(P_s, P_s, seq_len, seq_len, train_end)
    X_va, Y_va, va_idx = build_causal_windows(P_s, P_s, seq_len, train_end, val_end)
    X_te, Y_te, te_idx = build_causal_windows(P_s, P_s, seq_len, val_end, len(P_arr))

    model = CoffeeLSTM(in_dim=P_arr.shape[1], hidden_dim=hidden_dim, num_layers=1, out_dim=P_arr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    best_val, best_state, pat = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(X_va), Y_va).item()
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_p = p_scaler.inverse_transform(model(X_te).numpy())

    test_prev_p = price_df.shift(1).iloc[te_idx].values
    test_true_p = price_df.iloc[te_idx].values
    # Suy ngược log-return ngụ ý so với P_{t-1} để tính DA nhất quán
    implied_pred_ret = np.log(np.maximum(test_pred_p, 1.0) / test_prev_p)
    test_true_ret = np.log(test_true_p / test_prev_p)

    return {
        "test_pred_price": test_pred_p,
        "test_true_price": test_true_p,
        "test_pred_ret": implied_pred_ret,
        "test_true_ret": test_true_ret,
    }


def select_hac_lag(n: int, h: int = 1) -> int:
    """
    Chọn độ trễ cắt (truncation lag) cho ước lượng phương sai dài hạn HAC/Newey-West:
    Quy tắc Newey-West (1994) / Stock-Watson (2007): L = max(h - 1, floor(4 * (n / 100)^(2/9))).
    Với n = 542 và h = 1, cho L = 5.
    """
    if n <= 1:
        return 0
    sw_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(h - 1, min(sw_lag, n - 1))


def diebold_mariano_from_loss_diff(
    d: np.ndarray,
    h: int = 1,
    max_lag: Optional[int] = None,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """
    Kiểm định Diebold-Mariano (1995) trên chuỗi 1D chênh lệch tổn thất theo thời gian
    d_t = L(e_bench, t) - L(e_model, t), kết hợp:
      1. Phương sai dài hạn HAC / Newey-West (1987) với nhân Bartlett w_k = 1 - k / (L + 1)
         nhằm xử lý tự tương quan chuỗi d_t (công thức hiệp phương sai chia cho n đảm bảo LRV >= 0).
      2. Hiệu chỉnh mẫu hữu hạn Harvey-Leybourne-Newbold (1997).

    Giả thuyết với d_t = e_bench^2 - e_model^2:
      - alternative="greater" (1 phía):
          H0: E[d_t] <= 0  (MSE_model >= MSE_bench)
          H1: E[d_t] >  0  (MSE_model <  MSE_bench, mô hình tốt hơn benchmark)
      - alternative="two-sided" (2 phía):
          H0: E[d_t] == 0 (MSE_model == MSE_bench)
          H1: E[d_t] != 0 (hai mô hình có độ chính xác khác nhau)
      - alternative="less" (1 phía ngược):
          H0: E[d_t] >= 0  vs  H1: E[d_t] < 0 (MSE_model > MSE_bench)

    Lưu ý: Không kết luận mô hình tốt hơn chỉ dựa trên p-value nhỏ; luôn kiểm tra đồng thời
    dấu của DM_stat (DM_stat > 0 => MSE_model < MSE_bench) và độ lớn chênh lệch sai số.
    """
    d_arr = np.asarray(d, dtype=np.float64)
    if d_arr.ndim != 1:
        raise ValueError(f"Loss differential phải là chuỗi 1D theo thời gian, nhận shape={d_arr.shape}.")
    n = len(d_arr)
    if n < 2:
        return 0.0, 1.0

    mean_d = float(np.mean(d_arr))
    dc = d_arr - mean_d
    gamma_0 = float(np.dot(dc, dc) / n)
    if gamma_0 <= 1e-18:
        return 0.0, 1.0

    L = select_hac_lag(n, h=h) if max_lag is None else max(0, min(int(max_lag), n - 1))
    lrv = gamma_0
    for k in range(1, L + 1):
        wk = 1.0 - k / (L + 1.0)
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / n)
        lrv += 2.0 * wk * gamma_k

    if lrv <= 1e-18:
        lrv = gamma_0
    if lrv <= 1e-18:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(lrv / n)
    hln_inner = (n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n
    hln_factor = np.sqrt(max(hln_inner, 1e-12))
    dm_stat_hln = float(dm_stat * hln_factor)

    if alternative == "greater":
        p_val = float(stats.t.sf(dm_stat_hln, df=n - 1))
    elif alternative == "less":
        p_val = float(stats.t.cdf(dm_stat_hln, df=n - 1))
    elif alternative == "two-sided":
        p_val = float(2.0 * stats.t.sf(abs(dm_stat_hln), df=n - 1))
    else:
        raise ValueError(f"alternative không hợp lệ: {alternative}")

    return dm_stat_hln, p_val


def diebold_mariano_hln(
    e_bench: np.ndarray,
    e_model: np.ndarray,
    h: int = 1,
    max_lag: Optional[int] = None,
    alternative: str = "greater",
) -> Tuple[float, float]:
    """
    Kiểm định Diebold-Mariano (HAC/Newey-West + hiệu chỉnh HLN) trên sai số 1D của một chuỗi.
    Chặn truyền trực tiếp ma trận nhiều tỉnh để tránh lỗi flatten() phá vỡ cấu trúc thời gian.
    """
    eb = np.asarray(e_bench, dtype=np.float64)
    em = np.asarray(e_model, dtype=np.float64)
    if eb.ndim == 2 and eb.shape[1] == 1:
        eb = eb[:, 0]
    if em.ndim == 2 and em.shape[1] == 1:
        em = em[:, 0]
    if eb.ndim != 1 or em.ndim != 1:
        raise ValueError(
            f"diebold_mariano_hln chỉ nhận chuỗi 1D (hoặc N x 1); nhận {eb.shape} và {em.shape}. "
            "Với dữ liệu nhiều tỉnh (N, P), hãy dùng diebold_mariano_panel_hln."
        )
    if len(eb) != len(em):
        raise ValueError(f"Độ dài không khớp: {len(eb)} vs {len(em)}.")

    d = (eb ** 2) - (em ** 2)
    return diebold_mariano_from_loss_diff(d, h=h, max_lag=max_lag, alternative=alternative)


def diebold_mariano_panel_hln(
    e_bench: np.ndarray,
    e_model: np.ndarray,
    h: int = 1,
    max_lag: Optional[int] = None,
    alternative: str = "greater",
    province_names: Optional[List[str]] = None,
) -> Dict:
    """
    Kiểm định Diebold-Mariano cho dữ liệu panel dạng (N, P) mà KHÔNG flatten thành chuỗi dài N*P:
      - Phương án tổng hợp theo thời gian: d_bar_t = (1/P) * sum_p (e_bench_{t,p}^2 - e_model_{t,p}^2),
        bảo toàn đúng cấu trúc lịch t = 1..N (N = 542) và tính tương quan đồng thời giữa các tỉnh.
      - Phương án riêng từng tỉnh: kiểm định DM-HLN-HAC độc lập trên từng cột tỉnh p = 1..P.
    """
    eb = np.asarray(e_bench, dtype=np.float64)
    em = np.asarray(e_model, dtype=np.float64)
    if eb.ndim == 1:
        eb = eb[:, None]
    if em.ndim == 1:
        em = em[:, None]
    if eb.shape != em.shape:
        raise ValueError(f"Kích thước panel không khớp: {eb.shape} vs {em.shape}.")

    n, n_prov = eb.shape
    prov_labels = province_names if province_names is not None else [f"prov_{i}" for i in range(n_prov)]
    d_mat = (eb ** 2) - (em ** 2)
    d_bar = np.mean(d_mat, axis=1)

    L = select_hac_lag(n, h=h) if max_lag is None else max(0, min(int(max_lag), n - 1))
    agg_stat, agg_p = diebold_mariano_from_loss_diff(d_bar, h=h, max_lag=L, alternative=alternative)
    _, agg_p_2sided = diebold_mariano_from_loss_diff(d_bar, h=h, max_lag=L, alternative="two-sided")

    per_province = {}
    for j, pname in enumerate(prov_labels):
        st_j, p_j = diebold_mariano_from_loss_diff(d_mat[:, j], h=h, max_lag=L, alternative=alternative)
        _, p2_j = diebold_mariano_from_loss_diff(d_mat[:, j], h=h, max_lag=L, alternative="two-sided")
        per_province[pname] = {
            "dm_stat": st_j,
            "p_value": p_j,
            "p_two_sided": p2_j,
        }

    return {
        "agg_stat": agg_stat,
        "agg_p": agg_p,
        "agg_p_two_sided": agg_p_2sided,
        "per_province": per_province,
        "hac_lag": L,
        "n": n,
    }


def compute_metrics(
    true_price: np.ndarray,
    pred_price: np.ndarray,
    true_ret: np.ndarray,
    pred_ret: np.ndarray,
    bench_ret_err: Optional[np.ndarray] = None,
    bench_price_err: Optional[np.ndarray] = None,
    dm_max_lag: Optional[int] = None,
    dm_alternative: str = "greater",
) -> Dict[str, float]:
    """
    Tính đầy đủ các độ đo trên cả thang Giá (VND/kg) và thang Log-Return.
    Khi dữ liệu là panel nhiều tỉnh (N, P với P > 1), kiểm định DM sử dụng chuỗi
    loss differential tổng hợp theo từng ngày d_bar_t (độ dài N), KHÔNG dùng flatten().
    """
    p_err = true_price - pred_price
    rmse_p = float(np.sqrt(np.mean(p_err ** 2)))
    mae_p = float(np.mean(np.abs(p_err)))
    mape_p = float(np.mean(np.abs(p_err / true_price)) * 100.0)
    ss_res_p = np.sum(p_err ** 2)
    ss_tot_p = np.sum((true_price - np.mean(true_price)) ** 2)
    r2_p = float(1.0 - ss_res_p / ss_tot_p) if ss_tot_p > 0 else np.nan

    r_err = true_ret - pred_ret
    rmse_r = float(np.sqrt(np.mean(r_err ** 2)))
    mae_r = float(np.mean(np.abs(r_err)))
    ss_res_r = np.sum(r_err ** 2)
    ss_tot_r = np.sum((true_ret - np.mean(true_ret)) ** 2)
    r2_r = float(1.0 - ss_res_r / ss_tot_r) if ss_tot_r > 0 else np.nan

    non_zero_mask = true_ret != 0
    if np.sum(non_zero_mask) > 0:
        da = float(np.mean(np.sign(pred_ret[non_zero_mask]) == np.sign(true_ret[non_zero_mask])) * 100.0)
    else:
        da = 0.0

    if bench_ret_err is not None:
        if r_err.ndim == 2 and r_err.shape[1] > 1:
            dm_r_res = diebold_mariano_panel_hln(
                bench_ret_err, r_err, h=1, max_lag=dm_max_lag, alternative=dm_alternative
            )
            dm_stat_r = dm_r_res["agg_stat"]
            dm_p_r = dm_r_res["agg_p"]
            dm_p_r_2s = dm_r_res["agg_p_two_sided"]
        else:
            dm_stat_r, dm_p_r = diebold_mariano_hln(
                bench_ret_err, r_err, h=1, max_lag=dm_max_lag, alternative=dm_alternative
            )
            _, dm_p_r_2s = diebold_mariano_hln(
                bench_ret_err, r_err, h=1, max_lag=dm_max_lag, alternative="two-sided"
            )
    else:
        dm_stat_r, dm_p_r, dm_p_r_2s = np.nan, np.nan, np.nan

    if bench_price_err is not None:
        if p_err.ndim == 2 and p_err.shape[1] > 1:
            dm_p_res = diebold_mariano_panel_hln(
                bench_price_err, p_err, h=1, max_lag=dm_max_lag, alternative=dm_alternative
            )
            dm_stat_p = dm_p_res["agg_stat"]
            dm_p_p = dm_p_res["agg_p"]
            dm_p_p_2s = dm_p_res["agg_p_two_sided"]
        else:
            dm_stat_p, dm_p_p = diebold_mariano_hln(
                bench_price_err, p_err, h=1, max_lag=dm_max_lag, alternative=dm_alternative
            )
            _, dm_p_p_2s = diebold_mariano_hln(
                bench_price_err, p_err, h=1, max_lag=dm_max_lag, alternative="two-sided"
            )
    else:
        dm_stat_p, dm_p_p, dm_p_p_2s = np.nan, np.nan, np.nan

    return {
        "RMSE (VND)": rmse_p,
        "MAE (VND)": mae_p,
        "MAPE (%)": mape_p,
        "R2_price": r2_p,
        "R2_ret": r2_r,
        "DA (%)": da,
        "RMSE_ret": rmse_r,
        "MAE_ret": mae_r,
        "DM_stat_ret": dm_stat_r,
        "DM_p_ret": dm_p_r,
        "DM_p_ret_2sided": dm_p_r_2s,
        "DM_stat_price": dm_stat_p,
        "DM_p_price": dm_p_p,
        "DM_p_price_2sided": dm_p_p_2s,
    }


def compute_permutation_importance(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_scaler: StandardScaler,
    prev_prices: np.ndarray,
    true_prices: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tính độ quan trọng đặc trưng đơn biến bằng Permutation Importance trên tập Test:
    Hoán vị từng kênh biến đầu vào j giữa các sample windows và đo mức tăng RMSE (VND).
    Lưu ý: Với các biến có tương quan chéo (như 3 tỉnh, ret_london/ret_arabica, sin/cos),
    hoán vị đơn lẻ có thể tạo tổ hợp phi thực tế; nên kết hợp với Group Permutation
    Importance và Ablation Study.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        raw_out = model(X_tensor).numpy()
        base_pred_r = y_scaler.inverse_transform(raw_out) if y_scaler is not None else raw_out
    base_pred_p = prev_prices * np.exp(base_pred_r)
    base_rmse = float(np.sqrt(np.mean((true_prices - base_pred_p) ** 2)))

    n_samples = X_tensor.shape[0]
    rows = []
    for j, fname in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            X_perm = X_tensor.clone()
            X_perm[:, :, j] = X_tensor[perm_idx, :, j]
            with torch.no_grad():
                p_out = model(X_perm).numpy()
                p_r = y_scaler.inverse_transform(p_out) if y_scaler is not None else p_out
            p_p = prev_prices * np.exp(p_r)
            perm_rmse = float(np.sqrt(np.mean((true_prices - p_p) ** 2)))
            deltas.append(perm_rmse - base_rmse)
        rows.append(
            {
                "Feature": fname,
                "RMSE_Increase_VND": float(np.mean(deltas)),
                "RMSE_Increase_Std": float(np.std(deltas)),
            }
        )

    imp_df = pd.DataFrame(rows).sort_values("RMSE_Increase_VND", ascending=False).reset_index(drop=True)
    total_pos = imp_df["RMSE_Increase_VND"].clip(lower=0.0).sum()
    imp_df["Relative_Share (%)"] = (
        (imp_df["RMSE_Increase_VND"].clip(lower=0.0) / total_pos * 100.0) if total_pos > 0 else 0.0
    )
    return imp_df


# Alias tương thích hai chiều
LSTMModel = CoffeeLSTM
LSTM = CoffeeLSTM


def compute_group_permutation_importance(
    model: nn.Module,
    X_tensor: torch.Tensor,
    y_scaler: StandardScaler,
    prev_prices: np.ndarray,
    true_prices: np.ndarray,
    feature_names: List[str],
    feature_groups: Dict[str, List[str]],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tính Group Permutation Importance trên tập Test:
    Hoán vị ĐỒNG THỜI cả nhóm biến tương quan (Target, Weather, Economic, Calendar)
    cùng chỉ số perm_idx để bảo toàn cấu trúc phụ thuộc nội nhóm giữa các biến.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        raw_out = model(X_tensor).numpy()
        base_pred_r = y_scaler.inverse_transform(raw_out) if y_scaler is not None else raw_out
    base_pred_p = prev_prices * np.exp(base_pred_r)
    base_rmse = float(np.sqrt(np.mean((true_prices - base_pred_p) ** 2)))

    n_samples = X_tensor.shape[0]
    rows = []
    for gname, cols in feature_groups.items():
        idx_list = [feature_names.index(c) for c in cols if c in feature_names]
        if not idx_list:
            continue
        deltas = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(n_samples)
            X_perm = X_tensor.clone()
            X_perm[:, :, idx_list] = X_tensor[perm_idx][:, :, idx_list]
            with torch.no_grad():
                p_out = model(X_perm).numpy()
                p_r = y_scaler.inverse_transform(p_out) if y_scaler is not None else p_out
            p_p = prev_prices * np.exp(p_r)
            perm_rmse = float(np.sqrt(np.mean((true_prices - p_p) ** 2)))
            deltas.append(perm_rmse - base_rmse)
        rows.append(
            {
                "Group": gname,
                "Num_Features": len(idx_list),
                "Features": ", ".join([feature_names[i] for i in idx_list]),
                "RMSE_Increase_VND": float(np.mean(deltas)),
                "RMSE_Increase_Std": float(np.std(deltas)),
            }
        )

    imp_df = pd.DataFrame(rows).sort_values("RMSE_Increase_VND", ascending=False).reset_index(drop=True)
    total_pos = imp_df["RMSE_Increase_VND"].clip(lower=0.0).sum()
    imp_df["Relative_Share (%)"] = (
        (imp_df["RMSE_Increase_VND"].clip(lower=0.0) / total_pos * 100.0) if total_pos > 0 else 0.0
    )
    return imp_df


# Alias tương thích hai chiều
LSTMModel = CoffeeLSTM
LSTM = CoffeeLSTM

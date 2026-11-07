from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Core Model Components (Embedding, Attention, Forecasters)
# --------------------------------------------------------------------------- #
class ProvinceEmbedding(nn.Module):
    """Mã hoá tỉnh thành vector học được, nối vào đầu vào ở mọi bước thời gian."""

    def __init__(self, n_prov: int = 0, dim: int = 0):
        super().__init__()
        self.emb = nn.Embedding(n_prov, dim) if n_prov > 0 and dim > 0 else None
        self.dim = dim if self.emb is not None else 0

    def forward(self, x: torch.Tensor, pid=None):
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
    """LSTM chuẩn: chuỗi (B, L, V) [+ embedding tỉnh] -> giá trị kế tiếp của Y. Có hỗ trợ Residual Learning."""

    has_attention = False

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 n_prov: int = 0, emb_dim: int = 0, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.embed = ProvinceEmbedding(n_prov, emb_dim)
        self.lstm = nn.LSTM(n_in + self.embed.dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)
        if self.residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, pid=None, return_attention: bool = False):
        h, _ = self.lstm(self.embed(x, pid))
        delta = self.head(h[:, -1])
        return (x[:, -1, 0:1] + delta) if self.residual else delta


class LSTMAttentionForecaster(nn.Module):
    """LSTM kết hợp Temporal Attention: (B, L, V) [+ embedding tỉnh] -> context vector -> Y. Có hỗ trợ Residual Learning."""

    has_attention = True

    def __init__(self, n_in: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1,
                 n_prov: int = 0, emb_dim: int = 0, residual: bool = True):
        super().__init__()
        self.residual = residual
        self.embed = ProvinceEmbedding(n_prov, emb_dim)
        self.lstm = nn.LSTM(n_in + self.embed.dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.temporal = TemporalAttention(hidden)
        self.head = nn.Linear(2 * hidden, 1)
        if self.residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, pid=None, return_attention: bool = False):
        h, _ = self.lstm(self.embed(x, pid))
        ctx, beta = self.temporal(h)
        delta = self.head(torch.cat([ctx, h[:, -1]], dim=-1))
        out = (x[:, -1, 0:1] + delta) if self.residual else delta
        if return_attention:
            return out, None, beta
        return out

SCHEMA_VERSION = 3
AGGS = ("mean", "last", "first", "median")
EVAL_MODES = ("rolling", "blocks", "fixed")


# --------------------------------------------------------------------------- #
# Config (cấu hình khởi đầu đề xuất cho thí nghiệm tuần; CHƯA phải cấu hình tối ưu)
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    csv: str = "data/coffee_price_cleaned.csv"
    date_col: str = "date"
    province_col: str = "province"
    target_col: str = "coffee_price_vnd_kg"
    # --- ngoại sinh: ưu tiên KHAI BÁO RÕ ---
    exog_cols: list | None = None     # None -> tự lấy mọi cột số (có cảnh báo)
    shared_exog: list | None = None   # cột dùng chung mọi tỉnh (USD/VND, futures...)
    province_exog: list | None = None # cột khác nhau theo tỉnh (thời tiết...)
    provinces: list | None = None
    # --- tần suất & tổng hợp ---
    freq: str | None = "W"
    target_agg: str = "mean"
    exog_agg: str = "mean"
    # None = TỰ SUY RA từ tần suất thật của dữ liệu gốc trong đoạn train:
    #   min_obs = max(1, ceil(0.6 * trung vị số quan sát/kỳ của các kỳ có dữ liệu)).
    # Giá trị cuối cùng được ghi vào info.min_obs_per_period (+ min_obs_source).
    min_obs_per_period: int | None = None
    exog_ffill_limit: int = 0
    # kỳ cuối chưa hoàn tất (vd. dữ liệu mới đến thứ Tư): drop_incomplete | keep | error
    last_period_policy: str = "drop_incomplete"
    last_period_tolerance_days: int = 2     # cho phép thiếu cuối kỳ (thứ 7/CN không có giá)
    # số ngày tối thiểu có >= 2 tỉnh cùng có giá trị để kết luận ngoại sinh dùng chung/riêng
    exog_min_overlap_dates: int = 20
    # None = tự phát hiện ngoại sinh trùng nội dung TRÊN TẬP TRAIN; list = dùng đúng danh sách này
    drop_exog_cols: list | None = None
    allow_irregular: bool = False
    min_coverage: float = 0.5
    # --- cửa sổ & chia tập ---
    lookback: int = 26
    horizon: int = 7
    val_frac: float = 0.15
    test_frac: float = 0.15
    eval_mode: str = "blocks"
    log_target: bool = True
    # --- mô hình ---
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    grad_clip: float = 1.0
    seed: int = 42
    outdir: str = "outputs_multivar"
    plot_province: str | None = None

    def __post_init__(self):
        e = []
        for n in ("lookback", "horizon", "batch_size", "epochs", "patience", "hidden_size", "num_layers"):
            if not (isinstance(getattr(self, n), int) and getattr(self, n) >= 1):
                e.append(f"{n} phải là số nguyên >= 1 (đang là {getattr(self, n)!r})")
        if not (0 < self.val_frac < 1 and 0 < self.test_frac < 1):
            e.append("val_frac và test_frac phải trong (0, 1)")
        elif self.val_frac + self.test_frac > 0.6:
            e.append("val_frac + test_frac phải <= 0.6 (cần đủ dữ liệu train)")
        if not (0 <= self.dropout < 1):
            e.append("dropout phải trong [0, 1)")
        if self.lr <= 0 or self.grad_clip <= 0:
            e.append("lr và grad_clip phải > 0")
        if self.weight_decay < 0:
            e.append("weight_decay phải >= 0")
        if self.min_obs_per_period is not None and not (
                isinstance(self.min_obs_per_period, int) and self.min_obs_per_period >= 1):
            e.append("min_obs_per_period phải là None hoặc số nguyên >= 1")
        if self.exog_ffill_limit < 0:
            e.append("exog_ffill_limit phải >= 0")
        if self.last_period_policy not in ("drop_incomplete", "keep", "error"):
            e.append("last_period_policy phải thuộc drop_incomplete | keep | error")
        if self.last_period_tolerance_days < 0 or self.exog_min_overlap_dates < 1:
            e.append("last_period_tolerance_days >= 0 và exog_min_overlap_dates >= 1")
        if not (0 < self.min_coverage <= 1):
            e.append("min_coverage phải trong (0, 1]")
        if self.target_agg not in AGGS or self.exog_agg not in AGGS:
            e.append(f"target_agg/exog_agg phải thuộc {AGGS}")
        if self.eval_mode not in EVAL_MODES:
            e.append(f"eval_mode phải thuộc {EVAL_MODES}")
        if self.freq is not None and not isinstance(self.freq, str):
            e.append("freq phải là chuỗi (vd. 'W') hoặc None")
        both = set(self.shared_exog or []) & set(self.province_exog or [])
        if both:
            e.append(f"cột vừa shared vừa province: {sorted(both)}")
        if e:
            raise ValueError("Config không hợp lệ:\n - " + "\n - ".join(e))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data: panel (dài) -> dạng rộng (kỳ x [giá các tỉnh | ngoại sinh]), KHÔNG điền giá
# --------------------------------------------------------------------------- #
def _resample(w: pd.DataFrame, freq: str, how: str, min_obs: int) -> pd.DataFrame:
    r = w.resample(freq)
    return getattr(r, how)().where(r.count() >= min_obs)   # thiếu quan sát -> NaN


_RIGHT_LABELED = {"W", "M", "ME", "BM", "BME", "Q", "QE", "BQ", "BQE", "A", "Y", "YE", "BA", "BY", "BYE", "SM", "SME"}


def _period_end(label, freq: str) -> pd.Timestamp:
    """Ngày cuối của kỳ mà nhãn `label` đại diện (W/ME/QE... nhãn ở cuối kỳ; D/MS/QS... ở đầu kỳ)."""
    off = pd.tseries.frequencies.to_offset(freq)
    lab = pd.Timestamp(label)
    if off.name.split("-")[0] in _RIGHT_LABELED:
        return lab.normalize()
    return (lab + off - pd.Timedelta(days=1)).normalize()


def _trim_to_price(wide: pd.DataFrame, provs: list) -> pd.DataFrame:
    has = wide[provs].notna().any(axis=1)
    if not has.any():
        raise ValueError("Không có giá hợp lệ nào.")
    return wide.loc[has.idxmax(): has[::-1].idxmax()].copy()


def _longest_gap(isna: np.ndarray) -> int:
    run = best = 0
    for v in isna:
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def _exog_evidence(df_tr: pd.DataFrame, c: str, d: str):
    """Bằng chứng về cấu trúc: số ngày có >= 2 tỉnh CÙNG có giá trị, và độ lệch lớn nhất giữa
    các tỉnh trong những ngày đó. std = NaN khi < 2 tỉnh có dữ liệu -> KHÔNG tính là 'giống nhau'."""
    sd = df_tr.groupby(d)[c].std()
    n_overlap = int(sd.notna().sum())
    spread = float(sd.max()) if n_overlap else float("nan")
    vals = df_tr[c].to_numpy(float)
    mean_abs = float(np.nanmean(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    return n_overlap, spread, 1e-9 * max(1.0, mean_abs)


def _classify_exog(df, df_tr, cfg: Config, provs):
    """Trả về (exog, shared, per_prov, nguồn cấu trúc). Khai báo > bằng chứng từ TRAIN.
    Không đủ bằng chứng chồng lấp giữa các tỉnh -> LỖI, yêu cầu khai báo (không đoán)."""
    d, p, y = cfg.date_col, cfg.province_col, cfg.target_col
    shared_decl, prov_decl = list(cfg.shared_exog or []), list(cfg.province_exog or [])
    declared = shared_decl + prov_decl
    if declared:
        if cfg.exog_cols is not None and set(cfg.exog_cols) != set(declared):
            raise ValueError("exog_cols mâu thuẫn với shared_exog + province_exog.")
        exog, source = declared, "declared"
    elif cfg.exog_cols is not None:
        exog, source = list(cfg.exog_cols), "auto_from_train"
    else:
        exog = [c for c in df.select_dtypes(include=[np.number]).columns if c != y]
        source = "auto_from_train"
        print(f"[warn] không khai báo --exog: tự lấy mọi cột số {exog}. "
              f"Nên khai báo rõ để tránh đưa nhầm cột phái sinh/định danh.")

    if len(set(exog)) != len(exog):
        raise ValueError("Danh sách ngoại sinh có cột bị lặp tên.")
    bad = [c for c in exog if c in (y, d, p)]
    if bad:
        raise ValueError(f"Cột {bad} không được dùng làm ngoại sinh (trùng target/ngày/tỉnh).")
    if set(exog) & set(provs):
        raise ValueError(f"Tên cột ngoại sinh trùng tên tỉnh: {sorted(set(exog) & set(provs))}")
    for c in exog:
        if c not in df.columns:
            raise KeyError(f"Thiếu cột ngoại sinh '{c}'")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"Cột ngoại sinh '{c}' không phải số.")
        if np.isinf(df[c].to_numpy(float)).any():
            raise ValueError(f"Cột ngoại sinh '{c}' chứa Inf.")

    n_prov, need = len(provs), cfg.exog_min_overlap_dates
    shared, per_prov, undetermined = [], [], {}
    for c in exog:
        n_ov, spread, tol = _exog_evidence(df_tr, c, d)
        enough = n_ov >= need
        same = enough and spread <= tol
        if c in shared_decl:
            if enough and not same:
                raise ValueError(f"'{c}' khai báo shared nhưng khác nhau giữa các tỉnh (trong train).")
            if not enough and n_prov > 1:
                print(f"[warn] '{c}' khai báo shared nhưng chỉ có {n_ov} ngày có >= 2 tỉnh cùng có giá trị "
                      f"(cần >= {need}) nên không kiểm chứng được; tin vào khai báo.")
            kind = "shared"
        elif c in prov_decl:
            kind = "province"
        elif n_prov == 1:
            kind = "shared"                      # 1 tỉnh: không có gì để phân biệt
        elif not enough:
            undetermined[c] = n_ov
            continue
        else:
            kind = "shared" if same else "province"
        if kind == "shared":
            n_f, sp_f, tol_f = _exog_evidence(df, c, d)
            if n_f >= need and sp_f > tol_f:
                print(f"[warn] '{c}' dùng chung trong train nhưng khác nhau giữa tỉnh ở giai đoạn sau.")
            shared.append(c)
        else:
            per_prov.append(c)
    if undetermined:
        raise ValueError(
            f"Không đủ dữ liệu chồng lấp giữa các tỉnh để kết luận dùng chung hay theo tỉnh "
            f"(ngày có >= 2 tỉnh cùng có giá trị, cần >= {need}): {undetermined}. "
            f"Hãy khai báo --shared-exog / --province-exog cho các cột này.")
    return exog, shared, per_prov, source


def load_wide(cfg: Config):
    d, p, y = cfg.date_col, cfg.province_col, cfg.target_col
    df = pd.read_csv(cfg.csv, parse_dates=[d])
    for c in (d, p, y):
        if c not in df.columns:
            raise KeyError(f"Thiếu cột '{c}' trong {cfg.csv}")
    if df[d].isna().any() or df[p].isna().any():
        raise ValueError("Có ngày hoặc tỉnh bị trống.")
    df[p] = df[p].astype(str)
    if cfg.provinces:
        want = list(map(str, cfg.provinces))
        miss = set(want) - set(df[p])
        if miss:
            raise ValueError(f"Không có tỉnh: {sorted(miss)}")
        df = df[df[p].isin(want)]
    if df.empty:
        raise ValueError("Dữ liệu rỗng sau khi lọc.")
    if df.duplicated([d, p]).any():
        raise ValueError(f"Có dòng trùng ({d}, {p}). Lọc 1 loại cà phê / gộp trước khi chạy.")

    yv = pd.to_numeric(df[y], errors="coerce")
    if yv.isna().sum() > df[y].isna().sum():
        raise ValueError(f"Cột {y} có giá trị không phải số.")
    if np.isinf(yv.to_numpy(float)).any():
        raise ValueError(f"Cột {y} chứa Inf.")
    if (yv.dropna() <= 0).any():
        raise ValueError(f"Cột {y} có giá <= 0; giá phải > 0 (kiểm tra nguồn, KHÔNG sửa bằng điền số).")
    df[y] = yv

    provs = sorted(df[p].unique())
    wide_y = df.pivot(index=d, columns=p, values=y).reindex(columns=provs).sort_index()

    # cấu trúc ngoại sinh: khai báo hoặc xác định từ đoạn train xấp xỉ
    dates_u = np.sort(df[d].unique())
    n_tr = int(len(dates_u) * (1 - cfg.val_frac - cfg.test_frac))
    cutoff = dates_u[max(n_tr - 1, 0)]
    df_tr = df[df[d] <= cutoff]
    exog, shared, per_prov, structure_source = _classify_exog(df, df_tr, cfg, provs)

    parts = []
    for c in exog:
        if c in shared:
            parts.append(df.groupby(d)[c].first().rename(c).to_frame())
        else:
            w = df.pivot(index=d, columns=p, values=c).reindex(columns=provs)
            w.columns = [f"{c}@{k}" for k in w.columns]
            parts.append(w)
    wide_x = pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame(index=wide_y.index)

    # ngưỡng số quan sát tối thiểu/kỳ: theo cấu hình, hoặc suy ra từ tần suất thật trong TRAIN
    min_obs, min_obs_src, typical = cfg.min_obs_per_period, "config", None
    if not cfg.freq:
        min_obs, min_obs_src = 1, "không áp dụng (freq=None)"
    elif min_obs is None:
        cnt0 = wide_y.loc[:pd.Timestamp(cutoff)].resample(cfg.freq).count().to_numpy().ravel()
        cnt0 = cnt0[cnt0 > 0]
        typical = float(np.median(cnt0)) if len(cnt0) else 1.0
        min_obs = max(1, int(np.ceil(0.6 * typical)))
        min_obs_src = f"auto: trung vị {typical:g} quan sát/kỳ (train) -> ceil(0.6 x trung vị)"
    print(f"min_obs_per_period = {min_obs} ({min_obs_src})")

    if cfg.freq:
        wy = _resample(wide_y, cfg.freq, cfg.target_agg, min_obs)
        wx = _resample(wide_x, cfg.freq, cfg.exog_agg, 1) if wide_x.shape[1] else wide_x
    else:
        wy, wx = wide_y, wide_x
    wide = pd.concat([wy, wx], axis=1).sort_index()
    if not cfg.freq:
        gaps = wide.index.to_series().diff().dropna()
        if gaps.nunique() > 1 and not cfg.allow_irregular:
            raise ValueError(
                f"Trục thời gian không đều ({gaps.nunique()} độ dài khoảng khác nhau). Đặt --freq "
                f"(vd. W), hoặc --allow-irregular nếu chấp nhận lookback/horizon là số quan sát.")

    if not wide[provs].notna().any().any():
        raise ValueError(
            f"Không còn kỳ nào có giá sau khi gộp theo freq={cfg.freq!r} với min_obs_per_period={min_obs}. "
            f"Dữ liệu gốc có tối đa {int(wide_y.resample(cfg.freq).count().to_numpy().max()) if cfg.freq else 0} "
            f"quan sát/kỳ: hãy hạ min_obs_per_period (hoặc để None để tự suy ra).")
    wide = _trim_to_price(wide, provs)

    # kỳ cuối chưa hoàn tất (vd. dữ liệu mới đến thứ Tư) không được coi là một kỳ đầy đủ
    last_period = {"policy": cfg.last_period_policy, "complete": True, "action": "none"}
    if cfg.freq:
        data_end = pd.Timestamp(df.loc[df[y].notna(), d].max()).normalize()
        label = wide.index[-1]
        pend = _period_end(label, cfg.freq)
        complete = data_end >= pend - pd.Timedelta(days=cfg.last_period_tolerance_days)
        last_period.update({"label": str(label.date()), "period_end": str(pend.date()),
                            "data_end": str(data_end.date()), "complete": bool(complete)})
        if not complete:
            msg = (f"Kỳ cuối {label.date()} (kết thúc {pend.date()}) chưa hoàn tất: dữ liệu giá mới đến "
                   f"{data_end.date()}.")
            if cfg.last_period_policy == "error":
                raise ValueError(msg)
            if cfg.last_period_policy == "drop_incomplete":
                wide = _trim_to_price(wide.iloc[:-1], provs)
                last_period["action"] = "dropped"
                print(f"[warn] {msg} Đã loại kỳ này; kỳ cuối dùng là {wide.index[-1].date()}.")
            else:
                last_period["action"] = "kept_incomplete"
                print(f"[warn] {msg} Giữ lại theo last_period_policy=keep: giá kỳ cuối chỉ là trung bình một phần kỳ.")
    xcols = [c for c in wide.columns if c not in provs]

    # ngoại sinh trùng nội dung: quyết định CHỈ dựa trên đoạn train (không nhìn val/test)
    dropped_dup = []
    if cfg.drop_exog_cols is not None:
        dropped_dup = [c for c in cfg.drop_exog_cols if c in wide.columns]
    elif xcols:
        n_train = split_bounds(len(wide), cfg.val_frac, cfg.test_frac)["train"][1]
        trn, kept = wide[xcols].iloc[:n_train], []
        for c in xcols:
            twin = next((k for k in kept if trn[c].equals(trn[k])), None)
            if twin is None:
                kept.append(c)
            else:
                dropped_dup.append(c)
                if not wide[c].equals(wide[twin]):
                    print(f"[warn] '{c}' trùng '{twin}' trong train nhưng khác nhau ở val/test; vẫn bỏ vì "
                          f"quyết định dựa trên train. Khai báo --exog để giữ cả hai nếu cần.")
    if dropped_dup:
        print(f"[warn] bỏ ngoại sinh trùng nội dung (xác định trên train): {dropped_dup}")
        wide = wide.drop(columns=dropped_dup)
        xcols = [c for c in xcols if c not in dropped_dup]

    filled = 0
    if cfg.exog_ffill_limit > 0 and xcols:
        before = int(wide[xcols].isna().sum().sum())
        wide[xcols] = wide[xcols].ffill(limit=cfg.exog_ffill_limit)
        filled = before - int(wide[xcols].isna().sum().sum())

    if np.isinf(wide.to_numpy(float)).any():
        raise ValueError("Có Inf sau tiền xử lý.")

    cov = wide[provs].notna().mean()
    low = cov[cov < cfg.min_coverage]
    if len(low):
        raise ValueError(f"Tỉnh thiếu dữ liệu (< {cfg.min_coverage:.0%} kỳ có giá): {low.round(3).to_dict()}")

    quality = {
        "n_periods": int(len(wide)), "start": str(wide.index[0].date()), "end": str(wide.index[-1].date()),
        "price_coverage": {k: float(cov[k]) for k in provs},
        "price_longest_gap_periods": {k: _longest_gap(wide[k].isna().to_numpy()) for k in provs},
        "exog_missing_frac": {c: float(wide[c].isna().mean()) for c in xcols},
    }
    if cfg.freq:   # phân bố số quan sát/kỳ -> chọn min_obs_per_period theo dữ liệu thật
        cnt = wide_y.resample(cfg.freq).count().reindex(wide.index).fillna(0)
        bins = {"0": cnt == 0, "1": cnt == 1, "2": cnt == 2, "3": cnt == 3, "4": cnt == 4, "5+": cnt >= 5}
        quality["obs_per_period"] = {k: {b: int(m[k].sum()) for b, m in bins.items()} for k in provs}
        quality["periods_excluded_by_min_obs"] = {
            k: int(((cnt[k] > 0) & (cnt[k] < min_obs)).sum()) for k in provs}
        for k in provs:
            ex = quality["periods_excluded_by_min_obs"][k]
            if ex > 0.1 * len(wide):
                print(f"[warn] {k}: {ex} kỳ bị loại vì < {min_obs} quan sát/kỳ "
                      f"({ex / len(wide):.0%}). Xem quality.obs_per_period; cân nhắc min_obs_per_period.")
    for k in provs:
        if quality["price_coverage"][k] < 0.9:
            print(f"[warn] {k}: chỉ {quality['price_coverage'][k]:.1%} kỳ có giá -> nhiều cửa sổ bị loại.")

    if cfg.log_target:
        wide[provs] = np.log1p(wide[provs])

    info = {
        "provinces": provs, "shared_exog": shared, "per_province_exog": per_prov,
        "exog_structure_source": structure_source, "dropped_duplicate_exog": dropped_dup,
        "freq": cfg.freq, "target_agg": cfg.target_agg, "exog_agg": cfg.exog_agg,
        "target_definition": (f"{cfg.target_agg} theo kỳ {cfg.freq}" if cfg.freq else "giá gốc theo quan sát"),
        "min_obs_per_period": min_obs, "min_obs_source": min_obs_src,
        "typical_obs_per_period_train": typical,
        "last_period_policy": cfg.last_period_policy,
        "last_period_tolerance_days": cfg.last_period_tolerance_days,
        "last_period": last_period,
        "exog_ffill_limit": cfg.exog_ffill_limit, "exog_cells_filled": filled,
        "target_filled": False, "uses_future_exog": False,
        "data_quality": quality,
    }
    return wide, info


def split_bounds(n: int, val_frac: float, test_frac: float) -> dict:
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    n_train = n - n_val - n_test
    return {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}


def make_windows(X: np.ndarray, n_out: int, lookback: int, horizon: int, lo: int, hi: int, cols: list):
    """Input X[i-L:i,:], nhãn X[i:i+H,:n_out]; lo<=i, i+H<=hi.
    Cửa sổ có NaN/Inf ở đầu vào hoặc nhãn bị LOẠI; trả thêm thống kê nguyên nhân."""
    F = X.shape[1]
    cs = np.vstack([np.zeros((1, F), np.int64), np.cumsum(np.isfinite(X).astype(np.int64), 0)])
    cand = np.arange(max(lo, lookback), hi - horizon + 1)
    if len(cand) == 0:
        raise ValueError("Đoạn dữ liệu quá ngắn: giảm lookback/horizon hoặc tăng dữ liệu.")
    in_bad = (cs[cand] - cs[cand - lookback]) < lookback              # (n,F) thiếu ở đầu vào
    lab_bad = (cs[cand + horizon, :n_out] - cs[cand, :n_out]) < horizon  # (n,P) thiếu ở nhãn

    price_in = in_bad[:, :n_out].any(1)
    exog_in = in_bad[:, n_out:].any(1) if F > n_out else np.zeros(len(cand), bool)
    label = lab_bad.any(1)
    ok = ~(price_in | exog_in | label)
    idx = cand[ok]
    if len(idx) == 0:
        raise ValueError("Không còn cửa sổ hợp lệ (quá nhiều NaN). Xem data_quality / window report.")

    col_bad = in_bad.copy()
    col_bad[:, :n_out] |= lab_bad
    sole = (col_bad.sum(1) == 1)                                       # chỉ 1 cột gây loại
    stats = {
        "candidates": int(len(cand)), "kept": int(ok.sum()), "dropped": int((~ok).sum()),
        "dropped_reasons_overlapping": {
            "missing_price_in_input": int(price_in.sum()),
            "missing_price_in_label": int(label.sum()),
            "missing_exog_in_input": int(exog_in.sum()),
        },
        "dropped_only_because_of_exog": int((exog_in & ~price_in & ~label).sum()),
        "by_province": {cols[j]: {"missing_in_input": int(in_bad[:, j].sum()),
                                  "missing_in_label": int(lab_bad[:, j].sum()),
                                  "sole_cause": int((sole & col_bad[:, j]).sum())} for j in range(n_out)},
        "by_exog": {cols[j]: {"missing_in_input": int(in_bad[:, j].sum()),
                              "sole_cause": int((sole & col_bad[:, j]).sum())} for j in range(n_out, F)},
    }
    xs = np.stack([X[i - lookback: i] for i in idx]).astype(np.float32)
    ys = np.stack([X[i: i + horizon, :n_out] for i in idx]).astype(np.float32)
    assert np.isfinite(xs).all() and np.isfinite(ys).all()
    return idx, xs, ys, stats


def build_datasets(wide: pd.DataFrame, info: dict, cfg: Config):
    n_out = len(info["provinces"])
    vals = wide.to_numpy(np.float64)
    bounds = split_bounds(len(wide), cfg.val_frac, cfg.test_frac)

    a, b = bounds["train"]                         # scaler chỉ fit trên train, bỏ qua NaN
    tr = vals[a:b]
    if np.isnan(tr).all(0).any():
        raise ValueError("Có cột hoàn toàn NaN trong đoạn train.")
    mean, std = np.nanmean(tr, 0), np.nanstd(tr, 0)
    std[std < 1e-8] = 1.0
    X = (vals - mean) / std

    cols, dates = list(wide.columns), wide.index.to_numpy()
    data, stats = {}, {}
    for s, (lo, hi) in bounds.items():
        idx, xs, ys, st = make_windows(X, n_out, cfg.lookback, cfg.horizon, lo, hi, cols)
        data[s] = {"x": xs, "y": ys, "idx": idx,
                   "input_start": dates[idx - cfg.lookback], "input_end": dates[idx - 1],
                   "date": dates[idx], "date_end": dates[idx + cfg.horizon - 1]}   # date = forecast_start
        stats[s] = st
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "cols": cols,
              "n_out": n_out, "log_target": cfg.log_target}
    return data, scaler, X, stats


def print_window_report(stats: dict) -> None:
    print("\n=== Cửa sổ giữ / ứng viên và nguyên nhân bị loại ===")
    for s, st in stats.items():
        r = st["dropped_reasons_overlapping"]
        print(f"[{s}] {st['kept']}/{st['candidates']} giữ ({st['kept'] / st['candidates']:.0%}) | "
              f"thiếu giá đầu vào {r['missing_price_in_input']}, thiếu giá nhãn {r['missing_price_in_label']}, "
              f"thiếu ngoại sinh {r['missing_exog_in_input']} (chỉ vì ngoại sinh: {st['dropped_only_because_of_exog']})")
        offenders = sorted(
            [(k, v["sole_cause"], v["missing_in_input"]) for k, v in {**st["by_province"], **st["by_exog"]}.items()],
            key=lambda t: (-t[1], -t[2]))[:3]
        offenders = [o for o in offenders if o[2] > 0]
        if offenders:
            print("      cột gây mất nhiều nhất (sole_cause / thiếu ở input): " +
                  ", ".join(f"{k} ({a}/{b})" for k, a, b in offenders))
        if st["kept"] / st["candidates"] < 0.7:
            print(f"[warn] tập {s} mất > 30% cửa sổ vì thiếu dữ liệu; kết quả có thể thiên lệch.")


def to_loader(d: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(d["x"]), torch.from_numpy(d["y"]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MultivariateLSTM(nn.Module):
    """Shared LSTM encoder -> FC -> (horizon x n_provinces), direct multi-step."""

    def __init__(self, n_features, n_out, horizon, hidden_size, num_layers, dropout):
        super().__init__()
        self.horizon, self.n_out = horizon, n_out
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon * n_out),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).view(-1, self.horizon, self.n_out)


# --------------------------------------------------------------------------- #
# Train / predict
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, loss_fn, device, opt=None, clip=1.0) -> float:
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


def fit(model, train_loader, val_loader, cfg: Config, device):
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, cfg.patience // 3))
    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, opt, cfg.grad_clip)
        va = run_epoch(model, val_loader, loss_fn, device)
        if not (np.isfinite(tr) and np.isfinite(va)):
            raise RuntimeError(f"Loss không hữu hạn ở epoch {ep} (train={tr}, val={va}). "
                               f"Giảm lr / kiểm tra dữ liệu đầu vào.")
        sched.step(va)
        hist["train"].append(tr)
        hist["val"].append(va)
        print(f"epoch {ep:03d} | train {tr:.5f} | val {va:.5f} | lr {opt.param_groups[0]['lr']:.2e}")
        if va < best - 1e-6:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stopping tại epoch {ep} (best val {best:.5f})")
                break
    if best_state is None:
        raise RuntimeError("Không có epoch nào cải thiện val loss; không có trọng số tốt nhất để nạp.")
    model.load_state_dict(best_state)
    return hist, best


@torch.no_grad()
def predict(model, x: np.ndarray, device, batch_size=512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i: i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(x), batch_size)]
    return np.concatenate(out)


def inverse(a: np.ndarray, scaler: dict) -> np.ndarray:
    """a: (..., n_out) đã chuẩn hoá -> giá VND/kg."""
    k = scaler["n_out"]
    v = a * np.asarray(scaler["std"][:k]) + np.asarray(scaler["mean"][:k])
    return np.expm1(v) if scaler["log_target"] else v


# --------------------------------------------------------------------------- #
# Metrics & evaluation protocols
# --------------------------------------------------------------------------- #
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_%": float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-8)) * 100),
        "sMAPE_%": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-8)) * 100),
    }


def select_origins(idx: np.ndarray, mode: str, horizon: int) -> np.ndarray:
    if mode == "rolling":
        return np.arange(len(idx))
    if mode == "fixed":
        return np.array([0])
    if mode == "blocks":
        sel, last = [0], idx[0]
        for k in range(1, len(idx)):
            if idx[k] - last >= horizon:
                sel.append(k)
                last = idx[k]
        return np.array(sel)
    raise ValueError(f"eval_mode không hợp lệ: {mode}")


def summarize(y_true, y_pred, y_naive, provs) -> dict:
    H = y_true.shape[1]
    res = {"n_origins": int(len(y_true)),
           "pooled_note": "RMSE/MAE gộp các tỉnh phụ thuộc quy mô giá; ưu tiên per_province và macro_avg",
           "pooled": metrics(y_true, y_pred), "naive_pooled": metrics(y_true, y_naive)}
    per_p, skills = {}, []
    for j, k in enumerate(provs):
        m, n = metrics(y_true[..., j], y_pred[..., j]), metrics(y_true[..., j], y_naive[..., j])
        sk = 100 * (1 - m["RMSE"] / n["RMSE"]) if n["RMSE"] > 0 else float("nan")
        skills.append(sk)
        per_p[k] = {"model": m, "naive": n, "skill_vs_naive_RMSE_%": sk}
    res["per_province"] = per_p
    res["per_horizon"] = {
        f"t+{h + 1}": {"model": metrics(y_true[:, h], y_pred[:, h]), "naive": metrics(y_true[:, h], y_naive[:, h])}
        for h in range(H)}
    # ma trận MAE tỉnh x horizon (độc lập)
    mae_mat = lambda yp: {k: [float(np.mean(np.abs(yp[:, h, j] - y_true[:, h, j]))) for h in range(H)]
                          for j, k in enumerate(provs)}
    res["mae_matrix_province_horizon"] = mae_mat(y_pred)
    res["mae_matrix_province_horizon_naive"] = mae_mat(y_naive)
    # macro: tính metric riêng từng tỉnh rồi lấy trung bình
    mac = lambda who: {key: float(np.mean([per_p[k][who][key] for k in provs]))
                       for key in ("RMSE", "MAE", "MAPE_%", "sMAPE_%")}
    res["macro_avg"] = {"model": mac("model"), "naive": mac("naive"),
                        "skill_vs_naive_RMSE_%": float(np.nanmean(skills))}
    return res


def evaluate(model, d: dict, scaler: dict, provs: list, cfg: Config, device):
    y_true = inverse(d["y"], scaler)                                  # (N, H, P)
    y_pred = inverse(predict(model, d["x"], device), scaler)
    last = inverse(d["x"][:, -1, : scaler["n_out"]], scaler)
    y_naive = np.repeat(last[:, None, :], y_true.shape[1], axis=1)
    by = {}
    for m in EVAL_MODES:
        s = select_origins(d["idx"], m, cfg.horizon)
        by[m] = summarize(y_true[s], y_pred[s], y_naive[s], provs)
    return {"eval_mode": cfg.eval_mode, "primary": by[cfg.eval_mode], "by_scheme": by}, y_true, y_pred, y_naive


def print_report(res: dict, provs: list) -> None:
    pr = res["primary"]
    print(f"\n=== Test | eval_mode={res['eval_mode']} | n_origins={pr['n_origins']} ===")
    print(f"{'tỉnh':<16}{'MAE':>10}{'MAPE':>8}{'sMAPE':>8}{'naiveMAPE':>11}{'skill%':>8}")
    for k in provs:
        m = pr["per_province"][k]
        print(f"{k:<16}{m['model']['MAE']:>10.0f}{m['model']['MAPE_%']:>8.2f}{m['model']['sMAPE_%']:>8.2f}"
              f"{m['naive']['MAPE_%']:>11.2f}{m['skill_vs_naive_RMSE_%']:>8.1f}")
    ma = pr["macro_avg"]
    print(f"{'macro-avg':<16}{ma['model']['MAE']:>10.0f}{ma['model']['MAPE_%']:>8.2f}{ma['model']['sMAPE_%']:>8.2f}"
          f"{ma['naive']['MAPE_%']:>11.2f}{ma['skill_vs_naive_RMSE_%']:>8.1f}")
    print("macro skill vs naive theo giao thức: " + ", ".join(
        f"{m}={res['by_scheme'][m]['macro_avg']['skill_vs_naive_RMSE_%']:.1f} (n={res['by_scheme'][m]['n_origins']})"
        for m in res["by_scheme"]))


def save_tables(res: dict, provs: list, d: dict, cfg: Config, outdir: Path) -> None:
    H = cfg.horizon
    cols = [f"t+{h + 1}" for h in range(H)]
    for tag, key in (("model", "mae_matrix_province_horizon"), ("naive", "mae_matrix_province_horizon_naive")):
        pd.DataFrame(res["primary"][key], index=cols).T.rename_axis("province").to_csv(
            outdir / f"mae_matrix_{cfg.eval_mode}_{tag}.csv")
    for m in EVAL_MODES:     # origin dùng chung cho mọi mô hình khác
        s = select_origins(d["idx"], m, H)
        pd.DataFrame({"grid_index": d["idx"][s], "input_start": d["input_start"][s],
                      "input_end": d["input_end"][s], "forecast_start": d["date"][s],
                      "forecast_end": d["date_end"][s]}).to_csv(outdir / f"test_origins_{m}.csv", index=False)


def predictions_frame(d: dict, y_true, y_pred, y_naive, provs: list, dates_all, cfg: Config) -> pd.DataFrame:
    """Bảng dài (mỗi dòng = 1 origin x 1 horizon x 1 tỉnh) để đối chiếu giữa các mô hình.
    Chứa MỌI origin rolling; cờ in_blocks / in_fixed đánh dấu tập con của hai giao thức còn lại."""
    N, H, P = y_true.shape
    o = np.repeat(np.arange(N), H * P)
    h = np.tile(np.repeat(np.arange(H), P), N)
    j = np.tile(np.arange(P), N * H)
    idx = d["idx"][o]
    in_blocks = np.isin(np.arange(N), select_origins(d["idx"], "blocks", H))
    in_fixed = np.isin(np.arange(N), select_origins(d["idx"], "fixed", H))
    return pd.DataFrame({
        "window_id": o, "grid_index": idx,
        "input_start": d["input_start"][o], "input_end": d["input_end"][o],
        "forecast_start": d["date"][o], "forecast_end": d["date_end"][o],
        "province": np.asarray(provs)[j], "horizon": h + 1, "target_date": dates_all[idx + h],
        "y_true": y_true.reshape(-1), "y_pred": y_pred.reshape(-1), "y_naive": y_naive.reshape(-1),
        "in_rolling": True, "in_blocks": in_blocks[o], "in_fixed": in_fixed[o],
    })


# --------------------------------------------------------------------------- #
# Plots (theo eval_mode) + dự báo tương lai
# --------------------------------------------------------------------------- #
def save_plots(hist, d, y_true, y_pred, provs, plot_province, eval_mode, horizon, dates_all, outdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150); plt.close()

    j = provs.index(plot_province) if plot_province in provs else 0
    sel = select_origins(d["idx"], eval_mode, horizon)
    plt.figure(figsize=(10, 4))
    if eval_mode == "rolling":
        plt.plot(d["date"][sel], y_true[sel, 0, j], label="actual", color="black")
        plt.plot(d["date"][sel], y_pred[sel, 0, j], label="forecast t+1", alpha=0.8)
        plt.plot(d["date_end"][sel], y_pred[sel, horizon - 1, j], label=f"forecast t+{horizon} (vẽ tại kỳ đích)", alpha=0.8)
    else:   # blocks / fixed: vẽ nguyên đường dự báo H kỳ từ mỗi origin đã chọn
        for n, s in enumerate(sel):
            i = d["idx"][s]
            td = dates_all[i: i + horizon]
            plt.plot(td, y_true[s, :, j], color="black", label="actual" if n == 0 else None)
            plt.plot(td, y_pred[s, :, j], color="C0", marker="o", ms=3, label=f"forecast ({horizon} kỳ)" if n == 0 else None)
    plt.title(f"Test ({eval_mode}, {len(sel)} origin) - {provs[j]}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"forecast_plot_{eval_mode}.png", dpi=150); plt.close()


def forecast_latest(model, X_all, wide, scaler, provs, cfg, device) -> pd.DataFrame:
    """Dự báo H kỳ TIẾP THEO sau kỳ cuối của dữ liệu (tương lai thật)."""
    if list(wide.columns) != scaler["cols"]:
        raise ValueError("Thứ tự/tên cột khác với lúc huấn luyện.")
    win = wide.iloc[-cfg.lookback:]
    bad = win.isna()
    if bad.any().any():
        cols = bad.columns[bad.any()].tolist()
        detail = {c: [str(t.date()) for t in win.index[bad[c].to_numpy()]] for c in cols[:6]}
        raise ValueError(
            f"Cửa sổ mới nhất ({cfg.lookback} kỳ, đến {win.index[-1].date()}) thiếu dữ liệu ở "
            f"{len(cols)} cột: {detail}. Nguyên nhân thường gặp: kỳ có < min_obs_per_period="
            f"{cfg.min_obs_per_period} quan sát giá, hoặc ngoại sinh chưa cập nhật. Không dự báo từ số bịa.")
    pred = inverse(predict(model, X_all[-cfg.lookback:][None].astype(np.float32), device), scaler)[0]
    last = wide.index[-1]
    if cfg.freq:
        dates = pd.date_range(last, periods=cfg.horizon + 1, freq=cfg.freq)[1:]
    else:
        dates = [last + wide.index.to_series().diff().dropna().median() * (h + 1) for h in range(cfg.horizon)]
    return pd.DataFrame(pred, index=pd.DatetimeIndex(dates, name=cfg.date_col), columns=provs)


def _load_ckpt(path, device):
    try:                                   # PyTorch >= 1.13 có tham số weights_only
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:                      # PyTorch cũ
        return torch.load(path, map_location=device)


def forecast_from_checkpoint(ckpt_path: str, csv: str | None = None, device=None) -> pd.DataFrame:
    """Dự báo từ checkpoint; xử lý dữ liệu lại ĐÚNG như lúc huấn luyện và đối chiếu thông số."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = _load_ckpt(ckpt_path, device)
    if ck.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Checkpoint schema {ck.get('schema_version')} != {SCHEMA_VERSION}; huấn luyện lại bằng bản này.")
    saved = ck["info"]
    cfg = Config(**ck["config"])
    if csv:
        cfg.csv = csv
    # cố định tỉnh + cấu trúc ngoại sinh theo lúc huấn luyện (không tự nhận diện lại)
    cfg.provinces, cfg.shared_exog, cfg.province_exog, cfg.exog_cols = (
        saved["provinces"], saved["shared_exog"], saved["per_province_exog"], None)
    cfg.min_obs_per_period = saved["min_obs_per_period"]          # dùng đúng ngưỡng đã chốt
    cfg.drop_exog_cols = saved["dropped_duplicate_exog"]          # dùng đúng danh sách cột đã bỏ
    wide, info = load_wide(cfg)
    diff = {k: (saved.get(k), info.get(k)) for k in
            ("freq", "target_agg", "exog_agg", "min_obs_per_period", "exog_ffill_limit",
             "last_period_policy", "last_period_tolerance_days", "dropped_duplicate_exog",
             "provinces", "shared_exog", "per_province_exog") if saved.get(k) != info.get(k)}
    if diff:
        raise ValueError(f"Thông số xử lý dữ liệu khác lúc huấn luyện (đã lưu, mới): {diff}")
    sc = ck["scaler"]
    if list(wide.columns) != sc["cols"]:
        raise ValueError(f"Cột dữ liệu mới khác lúc huấn luyện:\nmới: {list(wide.columns)}\ncũ : {sc['cols']}")
    X = (wide.to_numpy(np.float64) - np.asarray(sc["mean"])) / np.asarray(sc["std"])
    model = MultivariateLSTM(**ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return forecast_latest(model, X, wide, sc, info["provinces"], cfg, device)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wide, info = load_wide(cfg)
    cfg.min_obs_per_period = info["min_obs_per_period"]            # ghi giá trị đã chốt vào checkpoint
    cfg.drop_exog_cols = info["dropped_duplicate_exog"]
    provs = info["provinces"]
    data, scaler, X_all, wstats = build_datasets(wide, info, cfg)
    print(f"mục tiêu: {info['target_definition']} | lookback={cfg.lookback}, horizon={cfg.horizon} kỳ | "
          f"giá bị điền: không | ngoại sinh bị điền: {info['exog_cells_filled']} ô | min_obs/kỳ={cfg.min_obs_per_period}")
    print(f"tỉnh: {provs}\nngoại sinh dùng chung: {info['shared_exog']} | theo tỉnh: {info['per_province_exog']} "
          f"| cấu trúc: {info['exog_structure_source']}")
    print_window_report(wstats)

    model_kwargs = dict(n_features=wide.shape[1], n_out=len(provs), horizon=cfg.horizon,
                        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, dropout=cfg.dropout)
    model = MultivariateLSTM(**model_kwargs).to(device)
    hist, best_val = fit(model, to_loader(data["train"], cfg.batch_size, True),
                         to_loader(data["val"], cfg.batch_size, False), cfg, device)

    res, y_true, y_pred, y_naive = evaluate(model, data["test"], scaler, provs, cfg, device)
    res.update({
        "best_val_mse_scaled": best_val, "windows": wstats, "inputs": info,
        "protocol": {
            "eval_mode": cfg.eval_mode, "lookback_periods": cfg.lookback, "horizon_periods": cfg.horizon,
            "freq": cfg.freq, "retrain_between_origins": False, "refit_on_train_plus_val": False,
            "uses_future_exog": False,
            "test_target_span": [str(pd.Timestamp(data["test"]["date"][0]).date()),
                                 str(pd.Timestamp(data["test"]["date_end"][-1]).date())],
            "note": "rolling/blocks/fixed là 3 giao thức khác nhau; mọi mô hình so sánh phải dùng "
                    "cùng tập test, cùng test_origins_<mode>.csv, cùng horizon và quy tắc cập nhật thông tin.",
        },
    })
    print_report(res, provs)

    torch.save({"schema_version": SCHEMA_VERSION, "state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                "config": asdict(cfg), "scaler": scaler, "info": info, "torch_version": torch.__version__},
               outdir / "best_multivar_lstm.pt")
    (outdir / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (outdir / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    save_tables(res, provs, data["test"], cfg, outdir)
    predictions_frame(data["test"], y_true, y_pred, y_naive, provs, wide.index.to_numpy(), cfg).to_csv(
        outdir / "predictions_test.csv", index=False)
    try:
        forecast_latest(model, X_all, wide, scaler, provs, cfg, device).to_csv(outdir / "forecast_next.csv")
    except ValueError as e:
        print("[warn] không xuất forecast_next.csv:", e)
    try:
        save_plots(hist, data["test"], y_true, y_pred, provs, cfg.plot_province, cfg.eval_mode,
                   cfg.horizon, wide.index.to_numpy(), outdir)
    except Exception as e:
        print("[warn] bỏ qua vẽ biểu đồ:", e)
    return res


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--csv", default=d.csv)
    p.add_argument("--date-col", default=d.date_col)
    p.add_argument("--province-col", default=d.province_col)
    p.add_argument("--target", dest="target_col", default=d.target_col)
    p.add_argument("--exog", dest="exog_cols", nargs="*", default=None)
    p.add_argument("--shared-exog", nargs="*", default=None, help="ngoại sinh dùng chung mọi tỉnh")
    p.add_argument("--province-exog", nargs="*", default=None, help="ngoại sinh khác nhau theo tỉnh")
    p.add_argument("--provinces", nargs="*", default=None)
    p.add_argument("--freq", default=d.freq, help='W, D, MS... hoặc "native" (phải đều)')
    p.add_argument("--target-agg", default=d.target_agg, choices=AGGS)
    p.add_argument("--exog-agg", default=d.exog_agg, choices=AGGS)
    p.add_argument("--min-obs-per-period", type=int, default=d.min_obs_per_period)
    p.add_argument("--exog-ffill-limit", type=int, default=d.exog_ffill_limit)
    p.add_argument("--last-period-policy", default=d.last_period_policy, choices=["drop_incomplete", "keep", "error"])
    p.add_argument("--last-period-tolerance-days", type=int, default=d.last_period_tolerance_days)
    p.add_argument("--exog-min-overlap-dates", type=int, default=d.exog_min_overlap_dates)
    p.add_argument("--drop-exog", dest="drop_exog_cols", nargs="*", default=None)
    p.add_argument("--allow-irregular", action="store_true")
    p.add_argument("--min-coverage", type=float, default=d.min_coverage)
    p.add_argument("--lookback", type=int, default=d.lookback)
    p.add_argument("--horizon", type=int, default=d.horizon)
    p.add_argument("--val-frac", type=float, default=d.val_frac)
    p.add_argument("--test-frac", type=float, default=d.test_frac)
    p.add_argument("--eval-mode", default=d.eval_mode, choices=EVAL_MODES)
    p.add_argument("--no-log-target", dest="log_target", action="store_false")
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--num-layers", type=int, default=d.num_layers)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--outdir", default=d.outdir)
    p.add_argument("--plot-province", default=None)
    a = vars(p.parse_args())
    if str(a["freq"]).lower() in ("native", "none"):
        a["freq"] = None
    return Config(**a)


if __name__ == "__main__":
    run(parse_args())
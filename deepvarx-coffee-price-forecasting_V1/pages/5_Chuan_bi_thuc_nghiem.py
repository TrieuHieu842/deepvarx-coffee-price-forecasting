# -*- coding: utf-8 -*-
"""
DeepVARX — Trang 13: Chuẩn bị thực nghiệm (Step 1–6 + Data Leakage Audit)
Câu hỏi của trang: "Thực nghiệm đã được thiết kế sạch chưa?"
Nút sang Huấn luyện chỉ mở khi hộp kiểm toán rò rỉ đạt toàn bộ.
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
META_COLS = {"date", "province", "coffee_type", "lat", "lon", "coffee_price_vnd_kg_raw"}
TARGET_COL = "coffee_price_vnd_kg"
PROVINCE_COL = "province"
COFFEE_TYPE_COL = "coffee_type"

st.set_page_config(page_title="Chuẩn bị thực nghiệm", page_icon="☕", layout="wide")


def _streamlit_version():
    return tuple(int(re.sub(r"\D", "", p) or 0) for p in st.__version__.split(".")[:2])


STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}

X_GROUPS = {
    "Market": {"exact": {"market", "volume", "export", "exports", "ice"}, "prefix": ("market", "export", "futur")},
    "Weather": {"exact": {"mua", "mưa"}, "prefix": ("temp", "rain", "precip", "humid", "weather", "wind", "solar")},
    "Macro": {"exact": {"usd", "dxy", "fx", "cpi", "inflation", "gdp", "rate", "vnd", "interest"},
              "prefix": ("exchange", "inflat")},
    "Calendar": {"exact": {"month", "year", "quarter", "week", "weekday", "dayofweek", "day", "season", "holiday"},
                 "prefix": ("calendar", "season", "holiday")},
}


def name_group(name):
    norm = re.sub(r"[^0-9a-zà-ỹ]+", " ", str(name).lower()).strip()
    tokens = set(norm.split())
    for g, kw in X_GROUPS.items():
        if tokens & kw["exact"] or any(t.startswith(p) for t in tokens for p in kw["prefix"]):
            return g
    return "Khác"


# ============================================================
# HELPERS — DỰNG CHUỖI & TIỀN XỬ LÝ
# ============================================================
def build_series(df, date_col, cfg):
    """Y: mỗi loại cà phê một chuỗi đại diện (hoặc các chuỗi tỉnh×loại được chọn). X: trung bình theo ngày."""
    ycol = cfg["ycol"]
    has_t = COFFEE_TYPE_COL in df.columns
    if cfg["ymode"] == "combo" and cfg["ycombos"]:
        cols = {}
        for lab in cfg["ycombos"]:
            prov, typ = [s.strip() for s in lab.split("|")]
            sub = df[(df[PROVINCE_COL].astype(str) == prov) & (df[COFFEE_TYPE_COL].astype(str) == typ)]
            cols[lab] = sub.groupby(date_col)[ycol].mean()
        Y = pd.DataFrame(cols)
    elif has_t:
        Y = df.pivot_table(index=date_col, columns=COFFEE_TYPE_COL, values=ycol, aggfunc="mean")
        Y.columns = [str(c) for c in Y.columns]
    else:
        Y = df.groupby(date_col)[ycol].mean().to_frame(ycol)
    Y = Y.sort_index()
    xc = [c for c in cfg["xcols"] if c in df.columns]
    X = df.groupby(date_col)[xc].mean().sort_index() if xc else pd.DataFrame(index=Y.index)
    return Y, X.reindex(Y.index.union(X.index)), Y.index.union(X.index)


def regularize(ts, freq):
    """Đưa về lưới thời gian đều để cửa sổ trễ có nghĩa (ngày thiếu thành NaN rồi xử lý theo chính sách khuyết)."""
    try:
        if freq == "Daily":
            return ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="D"))
        if freq == "Weekly":
            return ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="7D"))
        if freq == "Monthly":
            alias = "MS" if (ts.index.day == 1).all() else "ME"
            try:
                return ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq=alias))
            except ValueError:
                return ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="M"))
    except Exception:
        pass
    return ts


def ndiff(s, d):
    for _ in range(d):
        s = s.diff()
    return s


def make_scaler(kind):
    if kind == "StandardScaler":
        return StandardScaler() if SKLEARN_AVAILABLE else _NpStd()
    if kind == "MinMaxScaler":
        return MinMaxScaler() if SKLEARN_AVAILABLE else _NpMinMax()
    return None


class _NpStd:
    def fit(self, a):
        self.mean_ = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0)
        self.scale_ = np.where(sd == 0, 1.0, sd)
        return self

    def transform(self, a):
        return (a - self.mean_) / self.scale_


class _NpMinMax:
    def fit(self, a):
        self.data_min_, self.data_max_ = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
        return self

    def transform(self, a):
        return (a - self.data_min_) / np.where(self.data_max_ == self.data_min_, 1.0, self.data_max_ - self.data_min_)


def run_prepare(df, date_col, cfg):
    """Missing → log → differencing → scaling (fit trên TRAIN). Trả về dict kết quả hoặc {'error': ...}."""
    P = {"warnings": []}
    Y, X, _ = build_series(df, date_col, cfg)
    if Y.shape[1] == 0 or Y.notna().sum().sum() == 0:
        return {"error": "Chưa có chuỗi Y hợp lệ.", "warnings": []}
    ycols, xcols = list(Y.columns), list(X.columns)
    ts = pd.concat([Y, X], axis=1).sort_index()
    ts = regularize(ts, cfg["freq"])
    live = [c for c in ts.columns if ts[c].notna().any()]
    ts = ts[live]
    ycols, xcols = [c for c in ycols if c in live], [c for c in xcols if c in live]
    ts = ts.loc[max(ts[c].first_valid_index() for c in live):]

    dY, dX = cfg["dY"], (cfg["dX"] if xcols else 0)
    dmax = max(dY, dX)
    N = len(ts) - dmax
    H, purge = cfg["H"], cfg["purge"]
    n_tr, n_val = int(N * cfg["train_r"]), int(N * cfg["val_r"])
    n_te = N - n_tr - n_val - 2 * purge
    if min(n_tr, n_val, n_te) <= 0:
        return {"error": f"Không đủ mẫu để chia (N={N}, purge={purge}). Giảm purge hoặc đổi tỷ lệ.", "warnings": [], "N": N}
    seg = {"TRAIN": (0, n_tr), "VAL": (n_tr + purge, n_tr + purge + n_val),
           "TEST": (n_tr + n_val + 2 * purge, N)}
    train_end_ts = dmax + n_tr

    # --- missing (mặc định nhân quả) ---
    lim = cfg["fill_limit"]
    pol = cfg["miss_policy"]
    if pol == "ffill":
        filled = ts.ffill(limit=lim)
    elif pol == "interp_train":
        head = ts.iloc[:train_end_ts].interpolate(method="linear", limit_area="inside", limit=lim)
        filled = pd.concat([head, ts.iloc[train_end_ts:]]).ffill(limit=lim)
    else:
        filled = ts.copy()
    fill_mask = filled.notna() & ts.isna()

    # --- log → differencing ---
    tr = filled.copy()
    for c in cfg["log_vars"]:
        if c in tr.columns:
            if (tr[c].dropna() > 0).all():
                tr[c] = np.log(tr[c])
            else:
                P["warnings"].append(f"Không áp log cho `{c}` vì có giá trị ≤ 0.")
    for c in ycols:
        tr[c] = ndiff(tr[c], dY)
    for c in xcols:
        tr[c] = ndiff(tr[c], dX)
    tr, fill_mask, ts_cut, filled_cut = tr.iloc[dmax:], fill_mask.iloc[dmax:], ts.iloc[dmax:], filled.iloc[dmax:]

    # --- scaling: fit CHỈ trên TRAIN ---
    a_tr = tr.iloc[:n_tr].to_numpy(float)
    scaler = make_scaler(cfg["scaler"])
    if scaler is not None:
        scaler.fit(a_tr)
        scaled = scaler.transform(tr.to_numpy(float))
    else:
        scaled = tr.to_numpy(float)

    P.update({"Y": Y, "X": X, "ts": ts_cut, "filled": filled_cut, "fill_mask": fill_mask, "final": tr,
              "scaled": scaled, "scaler": scaler, "N": N, "seg": seg, "ycols": ycols, "xcols": xcols,
              "dates": tr.index, "n_tr": n_tr})
    return P


# ============================================================
# HELPERS — CỬA SỔ & MÔ HÌNH TUYẾN TÍNH TRỰC TIẾP
# ============================================================
def seg_origins(a, b, L, H):
    """Cửa sổ nằm trọn trong đoạn: s ∈ [a+L, b−H]; N_w = T_đoạn − L − H + 1."""
    return np.arange(a + L, b - H + 1)


def valid_mask(bad_rows, s, L, H):
    cum = np.r_[0, np.cumsum(bad_rows)]
    return (cum[s + H] - cum[s - L]) == 0


def design(A, s, K, p, q, use_x):
    parts = [np.ones((len(s), 1))] + [A[s - i, :K] for i in range(1, p + 1)]
    if use_x:
        parts += [A[s - j, K:] for j in range(1, q + 1)]
    return np.hstack(parts)


def eval_linear(A, K, seg_tr, seg_va, p, q, use_x, H):
    """Dự báo trực tiếp h = 1..H bằng OLS trên TRAIN, sai số RMSE trên VAL (đơn vị đã chuẩn hóa)."""
    M = A.shape[1] - K
    use_x = bool(use_x and M > 0)
    L = max(p, q if use_x else 0)
    bad = np.isnan(A if use_x else A[:, :K]).any(1)

    def origins(sg):
        s = seg_origins(sg[0], sg[1], L, H)
        return s[valid_mask(bad, s, L, H)] if len(s) else s

    s_tr, s_va = origins(seg_tr), origins(seg_va)
    nfeat = 1 + K * p + (M * q if use_x else 0)
    if len(s_tr) < nfeat + 10 or len(s_va) < 5:
        return None
    Ztr, Zva = design(A, s_tr, K, p, q, use_x), design(A, s_va, K, p, q, use_x)
    rm = []
    for h in range(1, H + 1):
        B = np.linalg.lstsq(Ztr, A[s_tr + h - 1, :K], rcond=None)[0]
        rm.append(float(np.sqrt(np.mean((A[s_va + h - 1, :K] - Zva @ B) ** 2))))
    return {"rmse_mean": float(np.mean(rm)), "rmse_H": rm[-1], "n_train": len(s_tr), "n_val": len(s_va), "c": nfeat}


def var_ic(Ytr, plist):
    Ytr = Ytr[~np.isnan(Ytr).any(1)]
    pmax = max(plist)
    out = []
    for p in plist:
        T, K = Ytr.shape
        Z = np.hstack([np.ones((T - pmax, 1))] + [Ytr[pmax - i:T - i] for i in range(1, p + 1)])
        Yt = Ytr[pmax:]
        if len(Yt) <= Z.shape[1] + 5:
            continue
        E = Yt - Z @ np.linalg.lstsq(Z, Yt, rcond=None)[0]
        ld = np.linalg.slogdet(E.T @ E / len(E))[1]
        n = len(E)
        out.append({"p": p, "AIC": ld + 2 * K * Z.shape[1] / n, "BIC": ld + np.log(n) * K * Z.shape[1] / n})
    return pd.DataFrame(out)


# ============================================================
# HELPERS — KIỂM TOÁN RÒ RỈ
# ============================================================
def check_builtin_features(df, cols, date_col):
    """#7: biến lag/rolling dựng sẵn phải dịch ≥ 1 và không dùng cửa sổ trung tâm."""
    sus = [c for c in cols if re.search(r"(^|_)(lag|shift|roll|rolling|ma|sma|ewm|ewma)(_|\d|$)", str(c).lower())]
    if not sus:
        return True, "không có biến lag/rolling dựng sẵn trong các biến đã chọn"
    gcols = [c for c in (PROVINCE_COL, COFFEE_TYPE_COL) if c in df.columns]
    d = df.sort_values(gcols + [date_col]) if date_col in df.columns else df
    grp = next((g for _, g in (d.groupby(gcols) if gcols else [(0, d)]) if len(g) >= 80), d)
    notes, ok_all = [], True
    for c in sus:
        cands = [b for b in df.columns if b != c and str(b).lower() in str(c).lower()]
        base = max(cands, key=len) if cands else None
        nums = re.findall(r"\d+", str(c))
        k = int(nums[-1]) if nums else 7
        verdict = None
        if base is not None and pd.api.types.is_numeric_dtype(grp[base]):
            x, b = grp[c].astype(float), grp[base].astype(float)

            def match(v):
                both = x.notna() & v.notna()
                return both.sum() > 30 and np.isclose(x[both], v[both], rtol=1e-6, atol=1e-9).mean() >= 0.99
            if re.search(r"lag|shift", str(c).lower()):
                for sh in range(0, 61):
                    if match(b.shift(sh)):
                        verdict = (sh >= 1, f"dịch {sh}")
                        break
            else:
                for fn in ("mean", "std"):
                    for sh in range(0, 4):
                        if match(getattr(b.rolling(k), fn)().shift(sh)):
                            verdict = (sh >= 1, f"rolling {fn} cửa sổ {k}, dịch {sh}")
                            break
                    if verdict:
                        break
                    if match(getattr(b.rolling(k, center=True), fn)()):
                        verdict = (False, f"cửa sổ TRUNG TÂM ({fn}, {k})")
                        break
        if verdict is None:
            ok_all = False
            notes.append(f"{c}: không xác minh được (cần kiểm tra thủ công)")
        else:
            ok_all &= verdict[0]
            notes.append(f"{c}: {verdict[1]} → {'OK' if verdict[0] else 'RÒ RỈ'}")
    return ok_all, "; ".join(notes)


def leakage_audit(P, cfg, df, date_col, prov, scopes, counts, pq_scope):
    rows = []

    def add(name, ok, measured):
        rows.append({"#": len(rows) + 1, "Kiểm tra": name, "Giá trị đo": measured, "Kết quả": "PASS" if ok else "FAIL"})

    if "error" in P:
        for i, nm in enumerate(["Chia theo thời gian", "Scaler chỉ fit trên train", "Chọn biến/kiểm định chỉ trên train",
                                "Điền khuyết nhân quả", "Không có Y tương lai trong đầu vào",
                                "Không có X chưa biết trước trong đầu vào", "Biến dựng sẵn (lag, rolling)",
                                "Test chỉ dùng một lần", "Không chồng cửa sổ giữa các đoạn"]):
            add(nm, False, P["error"])
        return rows
    H, p, q = cfg["H"], cfg["p"], cfg["q"]
    seg, dates, N = P["seg"], P["dates"], P["N"]
    K = len(P["ycols"])

    # 1
    tr, va, te = seg["TRAIN"], seg["VAL"], seg["TEST"]
    ok1 = dates[tr[1] - 1] < dates[va[0]] and dates[va[1] - 1] < dates[te[0]] and \
        (va[0] - tr[1]) >= H and (te[0] - va[1]) >= H
    add("Chia theo thời gian", ok1,
        f"max(train)={dates[tr[1]-1].date()} < min(val)={dates[va[0]].date()}; max(val)={dates[va[1]-1].date()} < "
        f"min(test)={dates[te[0]].date()}; khoảng cách {va[0]-tr[1]}/{te[0]-va[1]} ≥ H={H}")

    # 2
    sc = P["scaler"]
    A_final = P["final"].to_numpy(float)
    a_tr, a_all = A_final[:tr[1]], A_final
    if sc is None:
        add("Scaler chỉ fit trên train", True, "không chuẩn hóa (N/A)")
    elif cfg["scaler"] == "StandardScaler":
        err_tr = float(np.nanmax(np.abs(sc.mean_ - np.nanmean(a_tr, 0)))) + \
            float(np.nanmax(np.abs(sc.scale_ - np.nanstd(a_tr, 0))))
        diff_all = float(np.nanmax(np.abs(sc.mean_ - np.nanmean(a_all, 0))))
        add("Scaler chỉ fit trên train", err_tr < 1e-8 and diff_all > 1e-12,
            f"sai số so với mean/std train = {err_tr:.2e} (< 1e-8); lệch so với toàn bộ = {diff_all:.3g} (> 0)")
    else:
        err_tr = float(np.nanmax(np.abs(sc.data_min_ - np.nanmin(a_tr, 0)))) + \
            float(np.nanmax(np.abs(sc.data_max_ - np.nanmax(a_tr, 0))))
        add("Scaler chỉ fit trên train", err_tr < 1e-8, f"sai số so với min/max train = {err_tr:.2e}")

    # 3
    sc_list = dict(scopes)
    ok3, notes3 = True, []
    train_last = dates[tr[1] - 1]
    for nm, last in sc_list.items():
        try:
            lastd = pd.Timestamp(last)
            good = lastd <= train_last
        except Exception:
            good = False
        ok3 &= good
        notes3.append(f"{nm}: đến {last} ({'trong' if good else 'NGOÀI'} train)")
    if not sc_list:
        ok3, notes3 = False, ["chưa có metadata phạm vi dữ liệu của Trang 4–5 (chạy Trang 4 với 'Chỉ đoạn TRAIN')"]
    add("Chọn biến/kiểm định chỉ trên train", ok3, "; ".join(notes3) + f"; train đến {train_last.date()}")

    # 4
    fm, filled, ts = P["fill_mask"], P["filled"], P["ts"]
    ref = ts.ffill(limit=cfg["fill_limit"])
    bad_cells = 0
    for nm in ("VAL", "TEST"):
        a, b = seg[nm]
        m = fm.iloc[a:b]
        diff = (filled.iloc[a:b][m] != ref.iloc[a:b][m]) & m
        bad_cells += int(diff.sum().sum())
    risky = []
    if isinstance(prov, pd.DataFrame) and "Variable" in prov.columns:
        used = set(P["ycols"]) | set(P["xcols"]) | {cfg["ycol"]}
        for _, r in prov[prov["Variable"].isin(used)].iterrows():
            if r.get("Type") in ("Back-filled", "Synthetic") or str(r.get("Processing scope")) == "Vắt qua ranh giới train/val/test":
                risky.append(f"{r['Variable']} ({r.get('Type')}/{r.get('Processing scope')})")
    n_fill = {k: int(fm.iloc[a:b].sum().sum()) for k, (a, b) in seg.items()}
    add("Điền khuyết nhân quả", bad_cells == 0 and not risky,
        f"ô VAL/TEST điền không nhân quả = {bad_cells}; số ô được điền {n_fill}; chính sách = {cfg['miss_policy']}"
        + (f"; biến rủi ro ở Provenance: {', '.join(risky)}" if risky else ""))

    # 5, 6, 9
    all_ok5, all_ok6, targets, nw = True, True, {}, {}
    A = P["scaled"]
    L = max(p, q if P["xcols"] else 0)
    bad = np.isnan(A).any(1)
    for nm, (a, b) in seg.items():
        s = seg_origins(a, b, L, H)
        s = s[valid_mask(bad, s, L, H)] if len(s) else s
        nw[nm] = len(s)
        if len(s):
            all_ok5 &= bool((((s[:, None] + np.arange(-p, 0)).max(1)) <= s - 1).all())
            if P["xcols"]:
                all_ok6 &= bool((((s[:, None] + np.arange(-q, 0)).max(1)) <= s - 1).all())
            targets[nm] = set(np.unique((s[:, None] + np.arange(0, H)).ravel()).tolist())
        else:
            targets[nm] = set()
    add("Không có Y tương lai trong đầu vào", all_ok5 and sum(nw.values()) > 0,
        f"chỉ số lớn nhất của cửa sổ Y ≤ t−1 tại mọi gốc; số cửa sổ hợp lệ {nw}")
    groups = {}
    if isinstance(prov, pd.DataFrame) and "Known at forecast time" in prov.columns:
        groups = dict(zip(prov["Variable"], prov["Known at forecast time"]))
    kf = [x for x in P["xcols"] if groups.get(x) == "Known-future"]
    add("Không có X chưa biết trước trong đầu vào", all_ok6,
        f"cửa sổ X kết thúc tại t−1 cho {len(P['xcols'])} biến (mọi nhóm khả dụng đều dùng cửa sổ quá khứ); "
        f"{len(kf)} biến known-future chưa được dùng phần tương lai (thận trọng)"
        + ("" if groups else "; chưa có Provenance nên coi mọi X là past-only"))

    # 7
    cols7 = list(dict.fromkeys(list(cfg["xcols"]) + [cfg["ycol"]]))
    ok7, d7 = check_builtin_features(df, cols7, date_col)
    add("Biến dựng sẵn (lag, rolling) không rò rỉ", ok7, d7)

    # 8
    over = {k: v for k, v in counts.items() if v > 1}
    add("Test chỉ dùng một lần", not over and pq_scope in (None, "VAL"),
        f"số lần đánh giá TEST: {counts or 'chưa có'}; phạm vi chọn siêu tham số = {pq_scope or 'chưa chạy'}")

    # 9
    names = list(targets)
    inter = {f"{a}∩{b}": len(targets[a] & targets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    add("Không chồng cửa sổ giữa các đoạn", all(v == 0 for v in inter.values()), f"giao chỉ số đích: {inter}")
    return rows


# ============================================================
# SIDEBAR + KIỂM TRA DỮ LIỆU
# ============================================================
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
st.title("Trang 13 — Chuẩn bị thực nghiệm")
st.markdown("**Câu hỏi của trang:** *Thực nghiệm đã được thiết kế sạch chưa?* Nút sang Huấn luyện chỉ mở khi hộp kiểm toán rò rỉ đạt.")

proc, raw = st.session_state.get("processed_data"), st.session_state.get("raw_data")
df_all = proc if proc is not None else raw
if df_all is None:
    st.warning("Chưa có dữ liệu. Vui lòng tải dữ liệu ở Trang 1 trước.")
    if st.button("Trang 1"):
        st.switch_page("pages/1_Tai_du_lieu.py")
    st.stop()

date_col = st.session_state.get("date_col", "date")
if date_col not in df_all.columns:
    st.error(f"Không tìm thấy cột ngày `{date_col}`.")
    st.stop()
df_all = df_all.copy()
df_all[date_col] = pd.to_datetime(df_all[date_col], errors="coerce")
df_all = df_all.dropna(subset=[date_col])

num_all = [c for c in df_all.select_dtypes(include="number").columns if c not in META_COLS]
cs = st.session_state.get("corr_settings") or {}
short = st.session_state.get("candidate_shortlist")
vdict = st.session_state.get("variable_dictionary")
prov = st.session_state.get("provenance")
freq_detected = st.session_state.get("data_frequency") or "Daily"

tab1, tab2 = st.tabs(["Step 1–6 · Chuẩn bị & kiểm toán", "Độ nhạy p,q (VAR / VARX, trên VAL)"])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    # ── Step 1 ──────────────────────────────────────────────
    st.header("Step 1 — Target")
    s1a, s1b = st.columns([1, 2])
    y_def = cs.get("y") if cs.get("y") in num_all else (TARGET_COL if TARGET_COL in num_all else num_all[0])
    ycol = s1a.selectbox("Biến mục tiêu Y", num_all, index=num_all.index(y_def))
    ymode = s1b.radio("Chuỗi đích", ["Đại diện theo loại cà phê (mean các tỉnh)", "Chọn chuỗi cụ thể (tỉnh | loại)"],
                      horizontal=True)
    ycombos = []
    if ymode.startswith("Chọn"):
        if PROVINCE_COL in df_all.columns and COFFEE_TYPE_COL in df_all.columns:
            labels = sorted((df_all[PROVINCE_COL].astype(str) + " | " + df_all[COFFEE_TYPE_COL].astype(str)).unique())
            ycombos = st.multiselect("Chuỗi (tỉnh | loại)", labels, default=labels[:2])
        else:
            st.warning("Cần cột province và coffee_type để chọn chuỗi cụ thể.")

    # ── Step 2 ──────────────────────────────────────────────
    st.header("Step 2 — Covariates")
    x_opts = [c for c in num_all if c != ycol]
    short_map = short.set_index("Biến X").to_dict("index") if isinstance(short, pd.DataFrame) and "Biến X" in short.columns else {}
    grp_map = {}
    if isinstance(vdict, pd.DataFrame) and {"Variable", "Nhóm X"} <= set(vdict.columns):
        grp_map = {r["Variable"]: r["Nhóm X"] for _, r in vdict.iterrows() if r["Nhóm X"]}
    known_map = dict(zip(prov["Variable"], prov["Known at forecast time"])) if isinstance(prov, pd.DataFrame) and "Known at forecast time" in prov.columns else {}
    risk_map = dict(zip(prov["Variable"], prov["Look-ahead risk"])) if isinstance(prov, pd.DataFrame) and "Look-ahead risk" in prov.columns else {}
    reasons = st.session_state.get("covariate_reasons", {})
    prev_sel = [c for c in (st.session_state.get("corr_selected_vars") or st.session_state.get("exogenous_vars") or []) if c in x_opts]

    sig_opts = hashlib.md5(json.dumps([x_opts, bool(short_map), sorted(prev_sel)]).encode()).hexdigest()
    if "p13_cov_editor" not in st.session_state or st.session_state.get("p13_cov_sig") != sig_opts:
        rows = []
        for x in x_opts:
            sm = short_map.get(x, {})
            rows.append({"Chọn": x in prev_sel, "Biến X": x, "Nhóm": grp_map.get(x) or name_group(x),
                         "CCF lag đề xuất": sm.get("CCF lag đề xuất (Diff, tiền trắng hóa)", ""),
                         "p_Holm": sm.get("p_Holm (Granger)", np.nan), "VIF (Diff)": sm.get("VIF (Diff)", np.nan),
                         "Nhóm khả dụng (Tab D)": known_map.get(x, ""), "Look-ahead risk": risk_map.get(x, ""),
                         "Lý do / nguồn tài liệu": reasons.get(x, "")})
        st.session_state["p13_cov_base"] = pd.DataFrame(rows)
        st.session_state["p13_cov_sig"] = sig_opts
    cov_ed = st.data_editor(
        st.session_state["p13_cov_base"], hide_index=True, key="p13_cov_editor", num_rows="fixed", **STRETCH,
        disabled=["Biến X", "Nhóm", "CCF lag đề xuất", "p_Holm", "VIF (Diff)", "Nhóm khả dụng (Tab D)", "Look-ahead risk"],
        column_config={"Chọn": st.column_config.CheckboxColumn(),
                       "p_Holm": st.column_config.NumberColumn(format="%.4g"),
                       "VIF (Diff)": st.column_config.NumberColumn(format="%.2f"),
                       "Lý do / nguồn tài liệu": st.column_config.TextColumn(
                           help="Bắt buộc với mọi biến được chọn: bằng chứng ở Trang 4 hoặc lý do miền có nguồn tài liệu.")})
    chosen = cov_ed[cov_ed["Chọn"]]
    xcols = chosen["Biến X"].tolist()
    st.session_state["covariate_reasons"] = dict(zip(cov_ed["Biến X"], cov_ed["Lý do / nguồn tài liệu"]))
    no_reason = [r["Biến X"] for _, r in chosen.iterrows() if not str(r["Lý do / nguồn tài liệu"]).strip()]
    no_evid = [r["Biến X"] for _, r in chosen.iterrows()
               if str(r["CCF lag đề xuất"]).strip() in ("", "nan") and not (isinstance(r["p_Holm"], float) and r["p_Holm"] < 0.05)]
    step2_ok = not no_reason
    if no_reason:
        st.error("Ô lý do không được để trống. Còn thiếu: " + ", ".join(map(str, no_reason)))
    if no_evid:
        st.warning("Chưa có bằng chứng thống kê (CCF/Granger ở Trang 4) cho: " + ", ".join(map(str, no_evid)) +
                   " — phải dựa vào lý do miền có nguồn tài liệu.")
    st.caption("Mỗi biến chọn phải kèm ít nhất một bằng chứng hoặc một lý do miền có nguồn tài liệu. "
               "Nhóm: Market / Weather / Macro (và Calendar nếu có).")

    # ── Step 3 ──────────────────────────────────────────────
    st.header("Step 3 — Lag & Horizon")
    l1, l2, l3, l4 = st.columns(4)
    p_val = l1.number_input("Lookback Y (p)", 1, 120, int(st.session_state.get("lookback_p", 30)))
    q_val = l2.number_input("Lookback X (q)", 1, 120, int(st.session_state.get("lookback_q", 30)))
    H_val = l3.number_input("Horizon H", 1, 90, int(st.session_state.get("horizon_h", 7)))
    freq_opts = ["Daily", "Weekly", "Monthly", "Khác"]
    freq = l4.selectbox("Tần suất dữ liệu", freq_opts, index=freq_opts.index(freq_detected) if freq_detected in freq_opts else 3,
                        help=f"Phát hiện từ Trang 1: {freq_detected}")
    unit = {"Daily": "ngày", "Weekly": "tuần", "Monthly": "tháng"}.get(freq, "bước")
    fig = go.Figure()
    for name, x0, x1, col in [("Y đầu vào", -p_val, 0, "#3b82f6"), ("X đầu vào (past-only)", -q_val, 0, "#f59e0b"),
                              ("Mục tiêu", 0, H_val, "#22c55e")]:
        fig.add_trace(go.Bar(y=[name], x=[x1 - x0], base=[x0], orientation="h", marker_color=col, showlegend=False,
                             text=f"[{x0}, {x1 - 1 if x1 <= 0 else x1}]", textposition="inside"))
    fig.update_layout(title="Sơ đồ cửa sổ quanh điểm gốc t (t = bước dự báo đầu tiên)", height=220,
                      xaxis_title=f"Bước ({unit}) so với t", margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, **STRETCH)
    if freq != "Daily":
        st.warning(f"Dữ liệu theo {unit}: '30 ngày' và '7 ngày' không còn đúng; các giá trị trên là số bước {unit}.")

    # ── Step 4 ──────────────────────────────────────────────
    st.header("Step 4 — Transformation")
    t1, t2, t3 = st.columns(3)
    pol_lab = t1.radio("Missing handling", ["Forward-fill có giới hạn (nhân quả) — mặc định",
                                            "Time interpolation chỉ trong TRAIN (+ forward-fill cho VAL/TEST)",
                                            "Không điền"])
    miss_policy = "ffill" if pol_lab.startswith("Forward") else ("interp_train" if pol_lab.startswith("Time") else "none")
    fill_limit = t1.number_input("Giới hạn điền (số bước)", 1, 60, 5)
    scaler_kind = t2.radio("Scaling (fit trên TRAIN)", ["StandardScaler", "MinMaxScaler", "Không chuẩn hóa"],
                           index=["StandardScaler", "MinMaxScaler", "Không chuẩn hóa"].index(
                               st.session_state.get("scaler_type", "StandardScaler")))
    dY = t3.number_input("Bậc sai phân Y (d)", 0, 2, int(st.session_state.get("diff_order", 0) or 0),
                         help="Gợi ý từ Trang 5 (bậc tích hợp).")
    dX = t3.number_input("Bậc sai phân X (d)", 0, 2, 0)
    if miss_policy == "interp_train":
        st.warning("Nội suy dùng giá trị tương lai để điền quá khứ nên chỉ được áp dụng trong đoạn TRAIN; "
                   "VAL/TEST chỉ dùng forward-fill. Hãy ghi lý do lựa chọn.")
    miss_reason = st.text_input("Lý do chọn phương pháp xử lý khuyết/biến đổi", st.session_state.get("prep_reason", ""))

    # ── Step 5 ──────────────────────────────────────────────
    st.header("Step 5 — Temporal split")
    c1, c2, c3, c4 = st.columns(4)
    sp = st.session_state.get("split_config", {"train_ratio": 0.72, "val_ratio": 0.08})
    train_r = c1.number_input("Train (%)", 50, 90, int(sp["train_ratio"] * 100)) / 100
    val_r = c2.number_input("Validation (%)", 5, 30, int(sp["val_ratio"] * 100)) / 100
    purge = c3.number_input("Purge giữa các đoạn (≥ H)", int(H_val), 500, int(H_val),
                            help="Số quan sát loại bỏ giữa TRAIN|VAL|TEST để cửa sổ đích không chồng sang đoạn sau.")
    regime_thr = c4.number_input("Ngưỡng cảnh báo σ_test/σ_train", 1.1, 5.0, 1.5, 0.1, help="Tham số ứng dụng.")

    cfg = {"ycol": ycol, "ymode": "combo" if ycombos else "type", "ycombos": ycombos, "xcols": xcols,
           "freq": freq, "miss_policy": miss_policy, "fill_limit": int(fill_limit), "dY": int(dY), "dX": int(dX),
           "scaler": scaler_kind, "H": int(H_val), "p": int(p_val), "q": int(q_val), "purge": int(purge),
           "train_r": train_r, "val_r": val_r, "log_vars": []}

    # gợi ý log từ Spearman(rolling mean, rolling std) — chỉ trên TRAIN
    P0 = run_prepare(df_all, date_col, cfg)
    log_sel = []
    if "error" not in P0:
        a_tr = P0["seg"]["TRAIN"][1] + max(int(dY), int(dX) if xcols else 0)
        rows_l = []
        for c in P0["ts"].columns:
            s = P0["filled"][c].iloc[: P0["seg"]["TRAIN"][1]]
            if (s.dropna() > 0).all() and len(s.dropna()) > 80:
                rm, rs = s.rolling(30).mean(), s.rolling(30).std()
                rho = rm.corr(rs, method="spearman")
                rows_l.append({"Biến": c, "Spearman(rolling mean, rolling std)": rho,
                               "Gợi ý log": bool(rho > 0.5) if pd.notna(rho) else False})
        if rows_l:
            lg = pd.DataFrame(rows_l)
            st.markdown("**Gợi ý log** (tính trên TRAIN; ngưỡng ρ > 0.5 là tham số ứng dụng)")
            st.dataframe(lg.style.format({"Spearman(rolling mean, rolling std)": "{:.3f}"}), hide_index=True, **STRETCH)
            log_sel = st.multiselect("Biến áp log", lg["Biến"].tolist(), default=lg.loc[lg["Gợi ý log"], "Biến"].tolist())
        cfg["log_vars"] = log_sel
    P = run_prepare(df_all, date_col, cfg)

    if "error" in P:
        st.error(P["error"])
    else:
        for w in P["warnings"]:
            st.warning(w)
        K_, M_, N_ = len(P["ycols"]), len(P["xcols"]), P["N"]
        st.markdown(f"**Y ∈ ℝ^{{T×K}}: T = {N_:,}, K = {K_}** ({', '.join(P['ycols'])}) · "
                    f"**X ∈ ℝ^{{T×M}}: T = {N_:,}, M = {M_}**")
        ev_id = "P4.1-" + hashlib.md5(json.dumps([cs.get("y"), cs.get("agg"), cs.get("method"), cs.get("scope"),
                                                   cs.get("last_train_date")], default=str).encode()).hexdigest()[:8] if cs else None
        st.caption(("ID bằng chứng cho quyết định rút gọn chuỗi đích (Trang 4.1): **" + ev_id + "**") if ev_id else
                   "Chưa có ID bằng chứng cho quyết định rút gọn số tỉnh: chạy Trang 4.1 trước.")
        conf_k = st.checkbox(f"[XÁC NHẬN] Số chuỗi đích K = {K_} (luận văn: 2 chuỗi đại diện theo loại cà phê; bản phác thảo cũ: 4 tỉnh)")
        conf_f = st.checkbox(f"[XÁC NHẬN] Tần suất dữ liệu là '{freq}' (đơn vị bước = {unit})")

        # Step 3 tiếp: số cửa sổ
        L_eff = max(p_val, q_val if M_ else 0)
        seg = P["seg"]
        names = {"TRAIN": "TRAIN", "VAL": "VAL", "TEST": "TEST"}
        split_tbl = []
        A_all = P["scaled"]
        bad_all = np.isnan(A_all).any(1)
        for nm, (a, b) in seg.items():
            s = seg_origins(a, b, L_eff, H_val)
            nv = int(valid_mask(bad_all, s, L_eff, H_val).sum()) if len(s) else 0
            split_tbl.append({"Đoạn": nm, "Ngày đầu": P["dates"][a].date(), "Ngày cuối": P["dates"][b - 1].date(),
                              "n": b - a, "N_w = T − L − H + 1": max(b - a - L_eff - H_val + 1, 0), "Cửa sổ hợp lệ (không NaN)": nv})
        st.dataframe(pd.DataFrame(split_tbl), hide_index=True, **STRETCH)
        st.caption(f"L = max(p, q) = {L_eff}. Ví dụ số học: T_đoạn = 1000, L = 30, H = 7 → 964 cửa sổ.")

        # Step 4: số liệu scaler
        a_tr_f = P["final"].to_numpy(float)[: seg["TRAIN"][1]]
        a_all_f = P["final"].to_numpy(float)
        if cfg["scaler"] == "StandardScaler":
            sdf = pd.DataFrame({"Biến": P["final"].columns, "μ_train": np.nanmean(a_tr_f, 0), "σ_train": np.nanstd(a_tr_f, 0),
                                "|μ_toàn_bộ − μ_train| / σ_train": np.abs(np.nanmean(a_all_f, 0) - np.nanmean(a_tr_f, 0)) /
                                np.where(np.nanstd(a_tr_f, 0) == 0, np.nan, np.nanstd(a_tr_f, 0))})
            st.markdown("**Scaler: z = (x − μ_train)/σ_train** — cột cuối là mức rò rỉ đã tránh được.")
            st.dataframe(sdf.style.format({"μ_train": "{:.4g}", "σ_train": "{:.4g}", "|μ_toàn_bộ − μ_train| / σ_train": "{:.3f}"}),
                         hide_index=True, **STRETCH)
        elif cfg["scaler"] == "MinMaxScaler":
            zs = P["scaled"]
            out_frac = [float(np.nanmean((zs[seg[k][0]:seg[k][1]] < 0) | (zs[seg[k][0]:seg[k][1]] > 1))) for k in ("VAL", "TEST")]
            st.markdown(f"MinMax fit trên TRAIN: tỷ lệ giá trị VAL/TEST nằm ngoài [0,1] = {out_frac[0]:.3%} / {out_frac[1]:.3%}.")
        fm = P["fill_mask"]
        st.caption("Số ô được điền theo đoạn: " + str({k: int(fm.iloc[a:b].sum().sum()) for k, (a, b) in seg.items()}))

        # Step 5: chỉ số chế độ + walk-forward
        dy = P["Y"].reindex(P["dates"]).diff()
        reg = pd.DataFrame({"Y": P["ycols"], "σ_train(ΔY)": [dy[c].iloc[: seg["TRAIN"][1]].std() for c in P["ycols"]],
                            "σ_test(ΔY)": [dy[c].iloc[seg["TEST"][0]: seg["TEST"][1]].std() for c in P["ycols"]]})
        reg["σ_test/σ_train"] = reg["σ_test(ΔY)"] / reg["σ_train(ΔY)"]
        st.markdown("**Chỉ số chế độ biến động**")
        st.dataframe(reg.style.format({"σ_train(ΔY)": "{:.4g}", "σ_test(ΔY)": "{:.4g}", "σ_test/σ_train": "{:.3f}"}), hide_index=True, **STRETCH)
        if ((reg["σ_test/σ_train"] > regime_thr) | (reg["σ_test/σ_train"] < 1 / regime_thr)).any():
            st.warning("σ_test/σ_train lệch xa 1: đoạn test thuộc chế độ biến động khác, cần ghi chú khi diễn giải kết quả.")
        wf_on = st.checkbox("Expanding-window walk-forward (nhiều điểm gốc, Tashman 2000)")
        wf = None
        if wf_on:
            w1, w2 = st.columns(2)
            F_ = w1.number_input("Số fold F", 2, 20, 5)
            W_ = w2.number_input("Độ dài đoạn kiểm tra W", int(L_eff + H_val + 5), 2000, max(int(L_eff + H_val + 20), 60))
            t1_ = seg["TRAIN"][1]
            frows = []
            for f in range(int(F_)):
                tf = t1_ + f * int(W_)
                a, b = tf + int(purge), tf + int(purge) + int(W_)
                if b > P["N"]:
                    st.error(f"Fold {f + 1} vượt quá dữ liệu: giảm F hoặc W.")
                    break
                frows.append({"Fold": f + 1, "Train đến": P["dates"][tf - 1].date(), "Kiểm tra từ": P["dates"][a].date(),
                              "Kiểm tra đến": P["dates"][b - 1].date(), "N_w": max(int(W_) - L_eff - H_val + 1, 0)})
            if frows:
                st.dataframe(pd.DataFrame(frows), hide_index=True, **STRETCH)
                st.caption("Báo cáo mean ± std qua các fold ở Trang 16.")
                wf = {"F": int(F_), "W": int(W_)}

        # Step 3: độ nhạy theo L
        st.subheader("Độ nhạy validation theo L")
        L_grid = [l for l in [5, 10, 20, 30, 45, 60] if l + H_val + 10 < seg["TRAIN"][1]]
        if st.button("Tính độ nhạy theo L (OLS trực tiếp, trên VAL)"):
            K0 = len(P["ycols"])
            res, ic = [], var_ic(P["scaled"][: seg["TRAIN"][1], :K0], L_grid[:6] or [5])
            for L in L_grid:
                r1 = eval_linear(P["scaled"], K0, seg["TRAIN"], seg["VAL"], L, L, False, int(H_val))
                r2 = eval_linear(P["scaled"], K0, seg["TRAIN"], seg["VAL"], L, L, True, int(H_val)) if M_ else None
                res.append({"L": L, "RMSE VAR": r1["rmse_mean"] if r1 else np.nan, "RMSE VARX": r2["rmse_mean"] if r2 else np.nan})
            st.session_state["p13_sens"] = (pd.DataFrame(res), ic)
        sens = st.session_state.get("p13_sens")
        if sens is not None and len(sens[0]):
            sd, ic = sens
            fg = go.Figure()
            for c in ["RMSE VAR", "RMSE VARX"]:
                fg.add_trace(go.Scatter(x=sd["L"], y=sd[c], mode="lines+markers", name=c))
            fg.update_layout(title="RMSE validation (trung bình h = 1..H, đã chuẩn hóa) theo L", xaxis_title="L", height=300,
                             margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fg, **STRETCH)
            if len(ic):
                st.caption(f"AIC/BIC trên TRAIN (VAR): p tối ưu theo AIC = {int(ic.loc[ic['AIC'].idxmin(), 'p'])}, "
                           f"theo BIC = {int(ic.loc[ic['BIC'].idxmin(), 'p'])} (đường cong chính thức ở Trang 5.4). "
                           "Chọn L dựa trên AIC/BIC và độ nhạy validation. Mô hình tuyến tính trực tiếp chỉ để so độ nhạy.")

        # ── Step 6: audit ───────────────────────────────────
        st.header("Step 6 — Data Leakage Audit")
        scopes = dict(st.session_state.get("analysis_scopes", {}))
        if cs.get("last_train_date"):
            scopes["Trang 4 (tương quan/Granger)"] = cs["last_train_date"]
        elif cs.get("scope") == "Toàn bộ":
            scopes["Trang 4 (tương quan/Granger)"] = str(P["dates"][-1].date())
        audit = leakage_audit(P, cfg, df_all, date_col, prov, scopes,
                              st.session_state.get("test_eval_counts", {}), st.session_state.get("pq_scope"))
        aud_df = pd.DataFrame(audit)
        st.dataframe(aud_df, hide_index=True, **STRETCH)
        audit_ok = bool((aud_df["Kết quả"] == "PASS").all())
        st.session_state["leakage_audit"] = aud_df
        if audit_ok:
            st.success("Kiểm toán rò rỉ: PASS toàn bộ.")
        else:
            st.error("EXPERIMENT INVALID — có kiểm tra FAIL; nút huấn luyện bị khóa.")
        st.caption("PASS nghĩa là điều kiện an toàn được thỏa (thống nhất cho mọi dòng). Phân loại rò rỉ tham khảo Kaufman et al. (2012).")

        ready = audit_ok and step2_ok and conf_k and conf_f and bool(miss_reason.strip())
        if not miss_reason.strip():
            st.info("Cần ghi lý do chọn xử lý khuyết/biến đổi (Step 4) trước khi lưu.")
        if not (conf_k and conf_f):
            st.info("Cần tick hai ô [XÁC NHẬN] (K và tần suất).")
        signature = hashlib.md5(json.dumps([cfg, miss_reason], sort_keys=True, default=str).encode()).hexdigest()

        if st.button("Lưu cấu hình chuẩn bị", type="primary", disabled=not ready, **STRETCH):
            st.session_state.update({
                "endogenous_vars": P["ycols"], "exogenous_vars": P["xcols"], "scaler_type": scaler_kind,
                "lookback_p": int(p_val), "lookback_q": int(q_val), "horizon_h": int(H_val), "horizon_q": int(H_val),
                "diff_order": int(dY), "prep_reason": miss_reason, "prep_config": cfg, "prep_signature": signature,
                "split_config": {"train_ratio": train_r, "val_ratio": val_r,
                                 "test_ratio": round(1 - train_r - val_r, 4), "purge": int(purge)},
                "split_idx": {"n_train": seg["TRAIN"][1] - seg["TRAIN"][0], "n_val": seg["VAL"][1] - seg["VAL"][0],
                              "n_test": seg["TEST"][1] - seg["TEST"][0], "TRAIN": seg["TRAIN"], "VAL": seg["VAL"],
                              "TEST": seg["TEST"], "purge": int(purge)},
                "ts_df": P["final"], "walk_forward": wf,
                "scaler_params": None if P["scaler"] is None else {
                    k: getattr(P["scaler"], k).tolist() for k in ("mean_", "scale_", "data_min_", "data_max_") if hasattr(P["scaler"], k)},
            })
            st.success(f"Đã lưu: Y={len(P['ycols'])}, X={len(P['xcols'])}, p={p_val}, q={q_val}, H={H_val}, "
                       f"Train/Val/Test = {seg['TRAIN']}/{seg['VAL']}/{seg['TEST']}.")
        st.session_state["_p13_ready"] = (audit_ok, signature)

# ============================================================
# TAB 2 — ĐỘ NHẠY p,q (VAR / VARX), CHỈ TRÊN VAL
# ============================================================
with tab2:
    st.subheader("Độ nhạy lựa chọn p, q — VAR / VARX")
    st.markdown("p = lookback nội sinh, q = lookback ngoại sinh (không phải horizon). Dự báo trực tiếp h = 1..H bằng OLS "
                "trên TRAIN, chọn (p*, q*) theo RMSE trung bình trên **VAL**; TEST không được chạm.")
    st.info("DeepVAR / DeepVARX không có kết quả giả ở đây: siêu tham số của chúng được chọn bằng huấn luyện thật trên TRAIN "
            "và early stopping/chọn trên VAL ở Trang 15.")
    saved = st.session_state.get("prep_config")
    if not saved:
        st.warning("Hãy lưu cấu hình chuẩn bị ở tab 'Step 1–6' trước (cần kiểm toán PASS).")
    else:
        PS = run_prepare(df_all, date_col, saved)
        if "error" in PS:
            st.error(PS["error"])
        else:
            c1, c2 = st.columns(2)
            p_list = [int(x) for x in re.findall(r"\d+", c1.text_input("Danh sách p (lookback Y)", "1,3,5,7,10,14,30"))]
            q_list = [int(x) for x in re.findall(r"\d+", c2.text_input("Danh sách q (lookback X)", "1,3,7,14"))]
            models = st.multiselect("Mô hình", ["VAR", "VARX"], default=["VAR", "VARX"])
            n_combo = (len(p_list) if "VAR" in models else 0) + (len(p_list) * len(q_list) if "VARX" in models else 0)
            st.info(f"Số tổ hợp: **{n_combo}** (VAR chỉ lưới theo p; VARX lưới p × q).")
            if st.button("Chạy lưới p,q trên VAL", type="primary", **STRETCH):
                K0, Hs = len(PS["ycols"]), saved["H"]
                out, prog = [], st.progress(0)
                combos = [("VAR", p, 0) for p in p_list] * ("VAR" in models) + \
                         [("VARX", p, q) for p in p_list for q in q_list] * ("VARX" in models)
                for i, (m, p, q) in enumerate(combos):
                    r = eval_linear(PS["scaled"], K0, PS["seg"]["TRAIN"], PS["seg"]["VAL"], p, q, m == "VARX", Hs)
                    out.append({"Model": m, "p": p, "q": q if m == "VARX" else 0,
                                "RMSE (Val)": r["rmse_mean"] if r else np.nan, "RMSE h=H (Val)": r["rmse_H"] if r else np.nan,
                                "Hệ số c": r["c"] if r else np.nan, "Cửa sổ train": r["n_train"] if r else 0})
                    prog.progress((i + 1) / max(len(combos), 1))
                pq = pd.DataFrame(out)
                st.session_state["pq_results"] = pq
                st.session_state["pq_scope"] = "VAL"
                v = pq.dropna(subset=["RMSE (Val)"])
                if len(v):
                    best = v.loc[v.groupby("Model")["RMSE (Val)"].idxmin()]
                    st.session_state["best_pq"] = best.set_index("Model")[["p", "q", "RMSE (Val)"]].to_dict("index")
            if st.session_state.get("pq_results") is not None:
                pq = pd.DataFrame(st.session_state["pq_results"])
                st.dataframe(pq.sort_values(["Model", "RMSE (Val)"]).style.format(
                    {"RMSE (Val)": "{:.5f}", "RMSE h=H (Val)": "{:.5f}"}), hide_index=True, height=350, **STRETCH)
                for m, b in (st.session_state.get("best_pq") or {}).items():
                    a1, a2, a3 = st.columns(3)
                    a1.metric(m, f"p* = {int(b['p'])}" + (f", q* = {int(b['q'])}" if m == "VARX" else ""))
                    a2.metric("RMSE Validation", f"{b['RMSE (Val)']:.5f}")
                    a3.caption("Tổ hợp bị bỏ (NaN) khi số cửa sổ train không đủ so với số hệ số c.")

# ============================================================
# ĐIỀU HƯỚNG
# ============================================================
st.divider()
ok_now, sig_now = st.session_state.get("_p13_ready", (False, None))
saved_ok = ok_now and st.session_state.get("prep_signature") == sig_now
if not saved_ok:
    st.caption("Nút huấn luyện bị khóa cho đến khi kiểm toán rò rỉ PASS và cấu hình hiện tại đã được lưu.")
if st.button("Tiếp theo: Huấn luyện mô hình →", disabled=not saved_ok, **STRETCH):
    st.switch_page("pages/6_Huan_luyen.py")
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_utils import difference
from modules.dispatcher import PKG_FILES, build_packages
from modules.feature_selection import vif_table
from modules.panel_data import panel_diff_train, train_end_from_ratio
from modules.stat_tests import adf_kpss, integration_order
from modules.ui import fmt_p, require, reset_downstream, setup_page

setup_page("4 · Chốt biến & Dispatcher")
st.title("Trang 4 · Chốt biến & Bộ định tuyến dữ liệu")
st.caption("Gom bằng chứng từ Trang 1–3, tự động nhặt biến vào hai giỏ, rồi cắt "
           "**toàn bộ panel** thành 3 gói dữ liệu sẵn sàng huấn luyện.")
require(["df_clean_panel", "d", "evidence_lin", "evidence_nl"],
        "Cần hoàn tất **Trang 1**, **Trang 2** (chốt d, mở trang để lưu bằng chứng) và mở **Trang 3** (lưu bằng chứng XGBoost).")

panel = st.session_state["df_clean_panel"]
proxy_df = st.session_state["df_clean"]
pm = st.session_state["panel_meta"]
date_col, entity_col, target = pm["date_col"], pm["entity_col"], st.session_state["target"]
xcols = list(st.session_state["x_cols"])
train_end = st.session_state["train_end"]
train_n = st.session_state["train_n"]
d = st.session_state["d"]
d_eff = max(d, 1)
ev_lin = st.session_state["evidence_lin"].set_index("Biến")
ev_nl = st.session_state["evidence_nl"].set_index("Biến")
if not set(xcols) <= set(ev_lin.index) or not set(xcols) <= set(ev_nl.index):
    st.warning("Bằng chứng ở Trang 2/3 không khớp danh sách biến hiện tại. Hãy mở lại Trang 2 và Trang 3 để cập nhật.")
    st.stop()


@st.cache_data(show_spinner="Đang tổng hợp tính dừng…")
def stationarity_info(train_df: pd.DataFrame, d_: int) -> pd.DataFrame:
    rows = []
    for c in train_df.columns:
        try:
            di = integration_order(train_df[c], 2, 0.05)
            ok = adf_kpss(difference(train_df[[c]], d_)[c], 0.05)["stationary"]
        except ImportError:
            di, ok = np.nan, None
        rows.append({"Biến": c, "d_i": di, "ok": ok})
    return pd.DataFrame(rows).set_index("Biến")


@st.cache_data(show_spinner="Đang tính VIF trên toàn panel…")
def vif_info(panel_, date_col_, entity_col_, cols_, train_end_, d_) -> pd.Series:
    x = panel_diff_train(panel_, date_col_, entity_col_, cols_, train_end_, d_)
    return vif_table(x).set_index("Biến")["VIF"] if x.shape[1] > 1 else pd.Series(1.0, index=cols_)


stn = stationarity_info(proxy_df.iloc[:train_n][xcols], int(d))
vif = vif_info(panel, date_col, entity_col, xcols, train_end, d_eff)
if stn["ok"].isna().any():
    st.warning("Thiếu `statsmodels`: chưa kiểm tra được tính dừng theo biến (tạm coi là đạt).")

R = pd.DataFrame(index=xcols)
R["d_i"] = stn["d_i"]
R["stat_ok"] = stn["ok"].map(lambda v: True if v is None or v != v else bool(v))
R["vif"] = vif
R["gr_ok"] = ev_lin.loc[xcols, "Granger đạt"].astype(bool)
R["gr_p"] = ev_lin.loc[xcols, "p Granger (Bonferroni)"]
R["ccf_lag"] = ev_lin.loc[xcols, "Trễ CCF"]
R["ccf_ok"] = ev_lin.loc[xcols, "CCF đạt"].astype(bool)
R["cons"] = ev_lin.loc[xcols, "Số tỉnh Granger đạt"]
n_prov = int(ev_lin["Tổng số tỉnh"].iloc[0])
R["bds_p"] = ev_nl.loc[xcols, "p BDS (phần dư Y~Y,X)"]
R["bds_yes"] = R["bds_p"] < 0.05
R["xgb_rank"] = ev_nl.loc[xcols, "Hạng (đại diện)"].astype(int)
R["xgb_rank_all"] = ev_nl.loc[xcols, "Hạng (toàn panel)"].astype(int)

# =============================================================================== PHẦN 1: bằng chứng (thu gọn)
with st.expander("Hồ sơ minh chứng tổng hợp (Evidence Matrix)", expanded=False):
    M = pd.DataFrame({
        "Biến (X)": xcols,
        "Tính dừng": [("–" if r.d_i != r.d_i else f"d={int(r.d_i)} " + ("✅" if r.stat_ok else "⚠️ chưa dừng")) for r in R.itertuples()],
        "VIF": [f"{r.vif:.1f}" + (" ⚠️" if r.vif >= 10 else "") for r in R.itertuples()],
        "Granger (đại diện)": [("Đạt" if r.gr_ok else "Trượt") + f" (p={fmt_p(r.gr_p)})" for r in R.itertuples()],
        "Nhất quán": [f"{int(r.cons)}/{n_prov} tỉnh" for r in R.itertuples()],
        "Trễ CCF": [f"lag {int(r.ccf_lag)}" + (" ✓" if r.ccf_ok else "") if r.ccf_lag == r.ccf_lag else "–" for r in R.itertuples()],
        "Phi tuyến BDS": [("–" if r.bds_p != r.bds_p else ("Có" if r.bds_yes else "Không") + f" (p={fmt_p(r.bds_p)})") for r in R.itertuples()],
        "XGBoost": [f"Top {r.xgb_rank} (đại diện) · Top {r.xgb_rank_all} (toàn panel)" for r in R.itertuples()],
    })
    sty = (M.style
           .map(lambda v: "background-color:#d5f5e3" if "✅" in v else ("background-color:#fdebd0" if "⚠️" in v else ""), subset=["Tính dừng"])
           .map(lambda v: "background-color:#fdebd0" if "⚠️" in v else "background-color:#d5f5e3", subset=["VIF"])
           .map(lambda v: "background-color:#d5f5e3" if v.startswith("Đạt") else "background-color:#fadbd8", subset=["Granger (đại diện)"])
           .map(lambda v: "background-color:#e8daef" if v.startswith("Có") else "", subset=["Phi tuyến BDS"]))
    st.dataframe(sty, width="stretch", hide_index=True)

# =============================================================================== PHẦN 2: hai giỏ biến
st.subheader("Chốt biến — hai tập dữ liệu")
r1, r2, r3 = st.columns(3)
top_n = r1.number_input("Top-n XGBoost coi là quan trọng", 1, max(1, len(xcols)), min(3, len(xcols)))
min_cons = r2.slider("Nhất quán tối thiểu giữa các tỉnh (Granger)", 0, 100, 0, 10,
                     help="0% = chỉ cần đạt ở tỉnh đại diện. Tăng lên để loại biến chỉ đạt ở vài tỉnh.") / 100
auto_model = st.session_state.get("base_model", "VAR")
bm_opts = ["VAR", "VECM"] if d == 1 else ["VAR"]
base_model = r3.selectbox("Mô hình cơ sở (từ Johansen ở Trang 2)", bm_opts, index=bm_opts.index(auto_model) if auto_model in bm_opts else 0)

left, right = st.columns(2, gap="large")


def _fail(r) -> str:
    out = []
    if not r.gr_ok:
        out.append(f"Trượt Granger (p={fmt_p(r.gr_p)})")
    if r.vif >= 10:
        out.append(f"VIF={r.vif:.1f} ≥ 10")
    if not r.stat_ok:
        out.append(f"chưa dừng ở d={d}")
    if r.cons / max(n_prov, 1) < min_cons:
        out.append(f"chỉ đạt ở {int(r.cons)}/{n_prov} tỉnh")
    return "; ".join(out)


base_auto = {v: bool(R.loc[v, "gr_ok"] and R.loc[v, "vif"] < 10 and R.loc[v, "stat_ok"]
                     and R.loc[v, "cons"] / max(n_prov, 1) >= min_cons) for v in xcols}
with left:
    st.markdown("#### Tập Cơ Sở")
    st.caption(f"Luồng tuyến tính · {base_model}, VARX")
    st.markdown("**Quy tắc tự động**\n- Granger = Đạt (p < 0.05)\n- Không vướng VIF cao (< 10)\n"
                f"- Đồng nhất bậc d (dừng sau khi sai phân d = {d})\n- Nhất quán ≥ {min_cons:.0%} số tỉnh")
    base_sel = {}
    for v in sorted(xcols, key=lambda v: (not base_auto[v], R.loc[v, "gr_p"])):
        r = R.loc[v]
        ok = base_auto[v]
        base_sel[v] = st.checkbox(f"{'✅' if ok else '❌'} {v} — " + (f"Granger p={fmt_p(r.gr_p)}, {int(r.cons)}/{n_prov} tỉnh" if ok else _fail(r)),
                                  value=ok, key=f"base_{v}_{int(ok)}_{d}_{int(min_cons * 100)}")
    base_final = [v for v in xcols if base_sel[v]]
    if base_model == "VECM":
        st.warning("Gói A giữ **mức gốc** để mô hình tự tính số hạng hiệu chỉnh sai số (ECT) từ Johansen.")
    elif d > 0:
        st.warning(f"Gói A sẽ bị ép sai phân bậc d = {d} để thỏa mãn tính dừng.")

with right:
    st.markdown("#### Tập Mở Rộng")
    st.caption("Luồng phi tuyến · VARX-L, LSTM, DeepVAR, DeepVARX")
    st.markdown(f"**Quy tắc tự động**\n- Bao gồm toàn bộ Tập Cơ Sở\n- Cộng biến có tín hiệu phi tuyến (BDS) hoặc Top {top_n} XGBoost (kể cả đã trượt Granger)")
    reasons = {v: "Kế thừa Tập cơ sở" for v in base_final}
    for v in base_final:
        st.checkbox(f"✅ {v} — Kế thừa Tập cơ sở", value=True, disabled=True, key=f"ext_inh_{v}")
    extras = [v for v in xcols if v not in base_final]
    auto_ext = {v: bool(R.loc[v, "bds_yes"] or R.loc[v, "xgb_rank"] <= top_n) for v in extras}
    ext_extra = []
    for v in sorted(extras, key=lambda v: (not auto_ext[v], R.loc[v, "xgb_rank"])):
        r = R.loc[v]
        why = []
        if r.bds_yes:
            why.append("Tín hiệu BDS phi tuyến")
        if r.xgb_rank <= top_n:
            why.append(f"Top {r.xgb_rank} XGBoost" + (" dù trượt Granger" if not r.gr_ok else ""))
        reason = "; ".join(why) if why else "Không có tín hiệu phi tuyến / XGBoost thấp"
        on = st.checkbox(f"{'🌟' if auto_ext[v] else '▫️'} {v} — {reason}", value=auto_ext[v],
                         key=f"ext_{v}_{int(auto_ext[v])}_{top_n}_{len(base_final)}")
        if on:
            ext_extra.append(v)
            reasons[v] = reason
    ext_final = base_final + ext_extra
    st.info("Gói C giữ mức gốc (d = 0), Standard Scaler **theo từng tỉnh** (khớp trên TRAIN) + mã hóa tỉnh (embedding).")

c1, c2, c3 = st.columns(3)
c1.metric("Tập Cơ Sở", f"{len(base_final)} biến")
c2.metric("Tập Mở Rộng", f"{len(ext_final)} biến")
c3.metric("Dành riêng cho DeepVARX", f"+{len(ext_final) - len(base_final)} biến")

# =============================================================================== PHẦN 3: Dispatcher
st.divider()
st.subheader("Bộ định tuyến dữ liệu")
dates = np.sort(panel[date_col].unique())
cur_ratio = (dates <= np.datetime64(train_end)).sum() / len(dates)
opts = sorted({0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, round(cur_ratio, 2)})
def_idx = min(range(len(opts)), key=lambda i: abs(opts[i] - cur_ratio))
sel = st.selectbox("Mốc chia Train/Test (theo ngày, chung cho mọi tỉnh — kế thừa từ Trang 1)", opts, index=def_idx,
                   format_func=lambda r: f"{r:.0%} — TRAIN đến {pd.Timestamp(dates[max(int(len(dates) * r) - 1, 0)]):%d/%m/%Y}")
new_end = train_end if sel == opts[def_idx] else train_end_from_ratio(panel, date_col, sel)
if new_end < train_end:
    st.warning("Mốc TRAIN mới sớm hơn mốc đã dùng ở Trang 1–3: bằng chứng chọn biến đã nhìn thấy một phần TEST mới → rủi ro rò rỉ.")
st.markdown(
    f"- **Gói A** (Tập Cơ Sở, {'mức gốc' if base_model == 'VECM' else f'sai phân bậc {d}'}) → VAR/VECM, VARX  \n"
    f"- **Gói B** (Tập Mở Rộng, sai phân bậc {d}) → VARX-L  \n"
    "- **Gói C** (Tập Mở Rộng, mức gốc, Standard Scaler theo tỉnh + mã tỉnh) → LSTM, DeepVAR, DeepVARX")
st.markdown("<style>button[data-testid='stBaseButton-primary']{height:3.4rem;font-size:1.1rem;font-weight:700}</style>",
            unsafe_allow_html=True)
try:
    clicked = st.button("CẮT PANEL THÀNH 3 GÓI DỮ LIỆU HUẤN LUYỆN", type="primary", width="stretch")
except TypeError:
    clicked = st.button("CẮT PANEL THÀNH 3 GÓI DỮ LIỆU HUẤN LUYỆN", type="primary", use_container_width=True)

if clicked:
    if not ext_final:
        st.error("Chưa chọn biến nào. Hãy tick ít nhất một biến trước khi tạo gói.")
    else:
        if not base_final:
            st.warning("Tập Cơ Sở rỗng: VARX sẽ không có biến ngoại sinh (chỉ còn VAR).")
        with st.spinner("Đang cắt gói dữ liệu…"):
            pk, meta = build_packages(panel, date_col, entity_col, target, base_final, ext_final, int(d), new_end, base_model)
        reset_downstream("base_set")
        st.session_state.update(train_end=new_end, base_set=base_final, ext_set=ext_final, ext_sources=reasons,
                                base_model=base_model, packages=pk, pkg_meta=meta, export_paths=meta["paths"])
        n_e = panel[entity_col].nunique()
        st.success(f"Đã tạo 3 gói cho **{n_e}** tỉnh: A = {len(pk['A']):,} dòng ({len(base_final)} biến X) · "
                   f"B = {len(pk['B']):,} dòng ({len(ext_final)} biến X) · C = {len(pk['C']):,} dòng ({len(ext_final)} biến X). "
                   f"TRAIN đến {new_end:%d/%m/%Y}.")
        st.toast("Đã khởi tạo 3 gói dữ liệu huấn luyện")

pk, meta = st.session_state.get("packages"), st.session_state.get("pkg_meta")
if pk is not None:
    st.markdown(f"**Gói đã lưu** — Tập Cơ Sở: {', '.join(meta['base']) or '(rỗng)'} · Tập Mở Rộng: {', '.join(meta['ext'])}")
    names = {"A": "Gói A · Cơ Sở", "B": "Gói B · Mở Rộng (sai phân)", "C": "Gói C · Mở Rộng (scaled + mã tỉnh)"}
    for tab, k in zip(st.tabs(list(names.values())), names):
        with tab:
            path = Path(meta["paths"][k])
            st.caption(f"`data/processed/{PKG_FILES[k]}` · {len(pk[k]):,} dòng × {pk[k].shape[1]} cột")
            st.dataframe(pk[k].head(8), width="stretch")
            if path.exists():
                st.download_button("Tải CSV", path.read_bytes(), file_name=PKG_FILES[k], mime="text/csv", key=f"dl_{k}")
    st.page_link("pages/5_Dac_ta_Mo_hinh_va_Huan_luyen.py", label="Sang Trang 5 · Đặc tả mô hình & huấn luyện →", icon="🧬")

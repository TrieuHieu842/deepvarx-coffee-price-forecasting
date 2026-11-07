import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import modules.dispatcher
importlib.reload(modules.dispatcher)

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

# =============================================================================== PHẦN 2: Bộ 3 Biến Nội sinh (Y) & Danh sách Biến Ngoại sinh (X)
st.divider()
st.subheader("1. Bộ 3 biến nội sinh (Y — Đều là Giá Cà Phê)")
st.caption("Theo kế hoạch cải tiến mô hình, hệ thống sử dụng bộ 3 chuỗi giá cà phê tương hỗ làm các biến nội sinh:")

col_y1, col_y2, col_y3 = st.columns(3)
with col_y1:
    st.info("**Y₁: `coffee_price_vn`**\n\nGiá cà phê Robusta thu mua bình quân tại Việt Nam (VNĐ/kg) — **Biến mục tiêu trọng tâm cần dự báo**.")
with col_y2:
    st.info("**Y₂: `coffee_price_london_vnd`**\n\nGiá Robusta sàn ICE London quy đổi sang VNĐ/kg (`london_price_usd_ton * usd_vnd_rate / 1000`). Chuẩn quốc tế tham chiếu >95% hợp đồng.")
with col_y3:
    st.info("**Y₃: `coffee_price_arabica_vnd`**\n\nGiá Arabica sàn ICE New York quy đổi sang VNĐ/kg (`arabica_price_usd_lb * usd_vnd_rate / 0.45359237`). Tác động qua tỷ lệ phối trộn/thay thế.")

all_panel_cols = list(panel.columns)
planned_endog = ["coffee_price_vn", "coffee_price_london_vnd", "coffee_price_arabica_vnd"]
available_endog = [c for c in planned_endog if c in all_panel_cols]
if not available_endog:
    available_endog = [target] if target in all_panel_cols else [c for c in all_panel_cols if "price" in c.lower()][:3]

c_y_btn1, c_y_btn2 = st.columns([2, 2])
if c_y_btn1.button("🎯 Đặt lại Bộ 3 biến nội sinh đề xuất", help="Khôi phục mặc định: coffee_price_vn, coffee_price_london_vnd, coffee_price_arabica_vnd", use_container_width=True):
    st.session_state["chosen_endog"] = available_endog
    st.rerun()

if "chosen_endog" not in st.session_state or not st.session_state["chosen_endog"]:
    st.session_state["chosen_endog"] = available_endog

endog_options = list(dict.fromkeys(available_endog + [c for c in all_panel_cols if any(k in c.lower() for k in ["price", "gia", "vnd"]) and c != date_col and c != entity_col]))
chosen_endog = st.multiselect(
    "Danh sách biến nội sinh (Y) đưa vào hệ thống mô hình:",
    options=endog_options,
    default=[c for c in st.session_state["chosen_endog"] if c in endog_options],
    help="Các biến nội sinh sẽ được dự báo đồng thời trong hệ thống VAR / DeepVAR. Biến đầu tiên là biến mục tiêu trọng tâm (Y₁)."
)
st.session_state["chosen_endog"] = chosen_endog
if not chosen_endog:
    st.error("⚠️ Phải chọn ít nhất 1 biến nội sinh làm mục tiêu dự báo!")
else:
    st.success(f"✅ Đã chọn **{len(chosen_endog)}** biến nội sinh (Y): {', '.join(chosen_endog)}. Biến mục tiêu trọng tâm Y₁: **{chosen_endog[0]}**")

# =============================================================================== PHẦN 3: Lựa chọn Biến Ngoại sinh (X)
st.divider()
st.subheader("2. Lựa chọn danh sách biến ngoại sinh (X) — Các nhân tố Vĩ mô & Thị trường")
st.caption("Chọn các biến ngoại sinh bạn muốn đưa vào huấn luyện mô hình. Hệ thống sẽ cấp phát đồng nhất tập biến này "
           "cho tất cả các mô hình có biến ngoại sinh (VARX, VARX-L, DeepVARX, LSTM). "
           "**Lưu ý: Các biến nội sinh (chuỗi giá cà phê) đã được tách riêng ở Mục 1 và không nằm trong danh sách X này.**")

exclude_from_x = set(chosen_endog) | {"london_price_usd_ton", "arabica_price_usd_lb", "coffee_price_vnd_kg_raw", "coffee_price_vnd_kg",
                                     "coffee_price_vn", "coffee_price_london_vnd", "coffee_price_arabica_vnd"}
avail_x = [c for c in xcols if c not in exclude_from_x]

planned_exog = ["coffee_inventory", "diesel_price_vnd_liter", "usd_vnd_rate", "is_harvest_season_vn", "is_harvest_season_brazil"]
rec_exog = [c for c in planned_exog if c in avail_x]
if not rec_exog:
    rec_exog = [c for c in avail_x if (c in R.index and (R.loc[c, "gr_ok"] or R.loc[c, "ccf_ok"])) or "harvest" in c or "usd" in c][:5]

bx1, bx2, bx3, bx4 = st.columns(4)
if bx1.button("🎯 Chọn bộ 5 biến ngoại sinh đề xuất", help="Khôi phục mặc định: coffee_inventory, diesel_price_vnd_liter, usd_vnd_rate, is_harvest_season_vn, is_harvest_season_brazil", use_container_width=True):
    st.session_state["chosen_x_features"] = rec_exog
    st.rerun()

if bx2.button("🌟 Chọn tất cả biến ngoại sinh", help="Đưa toàn bộ biến ngoại sinh khả dụng vào mô hình", use_container_width=True):
    st.session_state["chosen_x_features"] = list(avail_x)
    st.rerun()

if bx3.button("📊 Chọn biến đạt thống kê", help="Chọn các biến ngoại sinh đạt Granger, CCF hoặc BDS", use_container_width=True):
    stat_vars = [c for c in avail_x if c in R.index and (R.loc[c, "gr_ok"] or R.loc[c, "ccf_ok"])]
    st.session_state["chosen_x_features"] = stat_vars if stat_vars else rec_exog
    st.rerun()

if bx4.button("🧹 Bỏ chọn tất cả biến X", help="Xóa chọn biến ngoại sinh để mô hình chỉ chạy với chuỗi nội sinh", use_container_width=True):
    st.session_state["chosen_x_features"] = []
    st.rerun()

if "chosen_x_features" not in st.session_state or any(c in st.session_state["chosen_x_features"] for c in exclude_from_x):
    st.session_state["chosen_x_features"] = [c for c in (st.session_state.get("chosen_x_features") or rec_exog) if c in avail_x]
    if not st.session_state["chosen_x_features"]:
        st.session_state["chosen_x_features"] = rec_exog

chosen_x = st.multiselect(
    "Danh sách biến ngoại sinh X đưa vào huấn luyện mô hình:",
    options=avail_x,
    default=[c for c in st.session_state["chosen_x_features"] if c in avail_x],
    help="Bạn có thể tự do thêm hoặc bớt bất kỳ biến ngoại sinh nào. Mọi mô hình dự báo sẽ nhận đúng tập biến này."
)
st.session_state["chosen_x_features"] = chosen_x

if chosen_x:
    st.success(f"Đã chọn **{len(chosen_x)}** biến ngoại sinh (X): {', '.join(chosen_x)}")
    with st.expander("Bảng đối soát thống kê của các biến ngoại sinh đã chọn", expanded=False):
        sub_M = M[M["Biến (X)"].isin(chosen_x)]
        st.dataframe(sub_M, width="stretch", hide_index=True)
else:
    st.warning("⚠️ Chưa chọn biến ngoại sinh nào. Các mô hình đa biến sẽ chạy chỉ với chuỗi giá nội sinh.")

# =============================================================================== PHẦN 4: Cấu hình mốc Train/Test & Mô hình cơ sở
st.divider()
st.subheader("3. Thiết lập mô hình cơ sở & Mốc chia Train / Test")
c_m1, c_m2 = st.columns(2)

auto_model = st.session_state.get("base_model", "VAR")
bm_opts = ["VAR", "VECM"] if d == 1 else ["VAR"]
base_model = c_m1.selectbox("Mô hình cơ sở tuyến tính (từ Johansen ở Trang 2)", bm_opts,
                            index=bm_opts.index(auto_model) if auto_model in bm_opts else 0)

dates = np.sort(panel[date_col].unique())
cur_ratio = (dates <= np.datetime64(train_end)).sum() / len(dates)
opts = sorted({0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, round(cur_ratio, 2)})
def_idx = min(range(len(opts)), key=lambda i: abs(opts[i] - cur_ratio))
sel = c_m2.selectbox("Mốc chia Train/Test (theo ngày chung cho toàn panel)", opts, index=def_idx,
                     format_func=lambda r: f"{r:.0%} — TRAIN đến {pd.Timestamp(dates[max(int(len(dates) * r) - 1, 0)]):%d/%m/%Y}")
new_end = train_end if sel == opts[def_idx] else train_end_from_ratio(panel, date_col, sel)
if new_end < train_end:
    st.warning("Mốc TRAIN mới sớm hơn mốc đã dùng ở Trang 1–3: bằng chứng chọn biến đã nhìn thấy một phần TEST mới → rủi ro rò rỉ.")

# =============================================================================== PHẦN 5: Nút xác nhận & Sinh dữ liệu
st.divider()
st.markdown("<style>button[data-testid='stBaseButton-primary']{height:3.4rem;font-size:1.1rem;font-weight:700}</style>",
            unsafe_allow_html=True)
clicked = st.button("🚀 XÁC NHẬN BỘ BIẾN & TẠO DỮ LIỆU HUẤN LUYỆN", type="primary", use_container_width=True)

if clicked:
    selected_x = list(chosen_x)
    selected_y = list(chosen_endog) if chosen_endog else ([target] if target else ["coffee_price_vn"])
    primary_target = selected_y[0]
    
    with st.spinner("Đang chuẩn bị dữ liệu huấn luyện cho toàn panel…"):
        try:
            pk, meta = modules.dispatcher.build_packages(panel, date_col, entity_col, primary_target, selected_x, selected_x, int(d), new_end, base_model, endog_cols=selected_y)
        except TypeError:
            importlib.reload(modules.dispatcher)
            pk, meta = modules.dispatcher.build_packages(panel, date_col, entity_col, primary_target, selected_x, selected_x, int(d), new_end, base_model, endog_cols=selected_y)
    reset_downstream("base_set")
    st.session_state.update(
        train_end=new_end, target=primary_target, endog_cols=selected_y, exog_cols=selected_x,
        base_set=selected_x, ext_set=selected_x,
        ext_sources={v: "Người dùng chọn trực tiếp" for v in selected_x},
        base_model=base_model, packages=pk, pkg_meta=meta, export_paths=meta["paths"]
    )
    n_e = panel[entity_col].nunique()
    st.success(f"🎉 Đã tạo thành công dữ liệu cho **{n_e}** tỉnh với **{len(selected_y)}** biến nội sinh (Y: {', '.join(selected_y)}) "
               f"và **{len(selected_x)}** biến ngoại sinh (X: {', '.join(selected_x)}). TRAIN đến {new_end:%d/%m/%Y}.")
    st.toast("Dữ liệu huấn luyện đã sẵn sàng!")

# =============================================================================== PHẦN 6: Xem trước dữ liệu đã lưu
pk, meta = st.session_state.get("packages"), st.session_state.get("pkg_meta")
if pk is not None:
    st.divider()
    st.subheader("4. Dữ liệu huấn luyện đã sẵn sàng")
    st.markdown(f"**Biến nội sinh (Y):** {', '.join(meta.get('endog_cols', [target]))} *(Mục tiêu dự báo trọng tâm Y₁: **{meta.get('target', target)}**)*")
    st.markdown(f"**Biến ngoại sinh (X):** {', '.join(meta.get('exog_cols', meta.get('base', []))) if meta.get('exog_cols', meta.get('base')) else '(Không dùng biến ngoại sinh)'}")
    st.caption(f"Bảng dữ liệu: {len(pk['C']):,} dòng × {pk['C'].shape[1]} cột (đã chuẩn hóa và sẵn sàng cho các mô hình)")
    st.dataframe(pk["C"].head(8), width="stretch")
    st.page_link("pages/5_Dac_ta_Mo_hinh_va_Huan_luyen.py", label="Chuyển sang Trang 5 · Đặc tả mô hình & Huấn luyện →", icon="🧬")

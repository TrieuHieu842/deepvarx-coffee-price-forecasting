import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.data_utils import INTERP_METHODS, ROOT, list_data_files, numeric_columns
from modules.feature_selection import high_corr_pairs, suggest_multicollinearity_drops, vif_table
from modules.panel_data import (build_passport, build_wide_passport, detect_columns, entity_frame,
                                find_default_file, fingerprint, is_endog_derived_col, make_demo_panel,
                                normalize_panel, normalize_wide, panel_diff_train_z, panel_gap_report,
                                panel_interpolate, panel_zero_diff, province_display_labels, read_table,
                                train_end_from_ratio, validate_panel, validate_wide, wide_diff_train_z,
                                wide_gap_report, wide_interpolate, wide_zero_diff, proxy_table)
from modules.ui import reset_downstream, setup_page, step_box

setup_page("1 · Tải dữ liệu & Làm sạch")
st.title("Trang 1 · Tải dữ liệu & Làm sạch toàn cục")
st.caption("Tự động nhận diện cấu trúc **Panel dọc** (có cột `province`) hoặc **Bảng rộng đa tỉnh** (như `data_daily_clean_v2.csv`, "
           "nơi các cột giá tỉnh Việt Nam là biến nội sinh Y và các cột ngoại sinh X giữ nguyên).")

tab1, tab2, tab3 = st.tabs(["Tải & Tổng quan dữ liệu", "Khoảng trống & Nội suy", "Biến tĩnh, Đa cộng tuyến"])


# =============================================================================== KHỞI TẠO PANEL / WIDE
def init_panel(tbl, meta, date_col, entity_cols, target, dayfirst, dup_policy) -> bool:
    """Kiểm tra → chuẩn hoá Panel (KHÔNG gộp) → passport → ghi session_state."""
    issues = validate_panel(tbl, date_col, entity_cols, target)
    for lvl, msg in issues:
        (st.error if lvl == "error" else st.warning)(msg)
    if any(lvl == "error" for lvl, _ in issues):
        return False
    master, info, entity_col = normalize_panel(tbl, date_col, entity_cols, dayfirst, dup_policy)
    pp = build_passport(master, meta, info, date_col, entity_cols, entity_col, target)
    reset_downstream("raw")
    st.session_state.update(
        data_mode="panel",
        df_master=master,
        ds_tinh=sorted(master[entity_cols[0]].astype(str).unique().tolist()),
        total_rows=len(master),
        rows_per_province=pp["rows_per_entity"],
        data_passport=pp,
        target=target,
        endog_cols=[target],
        endog_labels={},
        panel_meta={"mode": "panel", "date_col": date_col, "entity_cols": entity_cols, "entity_col": entity_col,
                    "target": target, "endog_cols": [target], "endog_labels": {}, "derived_endog": [],
                    "num_cols": pp["num_cols"]},
    )
    return True


def init_wide(tbl, meta, date_col, endog_cols, target, dayfirst, dup_policy) -> bool:
    """Kiểm tra → chuẩn hoá Bảng rộng đa tỉnh (các tỉnh VN là cột nội sinh, ngoại sinh giữ nguyên) → passport."""
    nums_all = numeric_columns(tbl, exclude=[date_col] + list(endog_cols))
    derived = [c for c in nums_all if is_endog_derived_col(c, endog_cols)]
    issues = validate_wide(tbl, date_col, endog_cols, target, derived)
    for lvl, msg in issues:
        (st.error if lvl == "error" else st.warning)(msg)
    if any(lvl == "error" for lvl, _ in issues):
        return False
    master, info = normalize_wide(tbl, date_col, dayfirst, dup_policy)
    ordered_endog = [target] + [c for c in endog_cols if c != target]
    labels = province_display_labels(ordered_endog)
    pp = build_wide_passport(master, meta, info, date_col, ordered_endog, target, derived, labels)
    reset_downstream("raw")
    st.session_state.update(
        data_mode="wide",
        df_master=master,
        ds_tinh=[labels[c] for c in ordered_endog],
        total_rows=len(master),
        rows_per_province=pp["rows_per_entity"],
        data_passport=pp,
        target=target,
        endog_cols=ordered_endog,
        endog_labels=labels,
        panel_meta={"mode": "wide", "date_col": date_col, "entity_cols": [], "entity_col": None,
                    "target": target, "endog_cols": ordered_endog, "endog_labels": labels,
                    "derived_endog": derived, "num_cols": pp["exog_cols"], "all_num_cols": pp["num_cols"]},
    )
    return True


# =============================================================================== TAB 1
with tab1:
    st.markdown("#### 1 · Tải dữ liệu")
    c_up, c_btn = st.columns([3, 2])
    up = c_up.file_uploader("Tải file CSV/Excel từ máy", type=["csv", "xlsx", "xls"])
    data_files = list_data_files()
    with c_btn:
        if data_files:
            picked_file = st.selectbox(
                "Chọn file có sẵn trong thư mục `data/`",
                data_files,
                format_func=lambda p: str(p.relative_to(ROOT)),
            )
            use_picked = st.button("Nạp file đã chọn từ data/", use_container_width=True)
        else:
            picked_file, use_picked = None, False
        b1, b2 = st.columns(2)
        use_default = b1.button("Dữ liệu mặc định", use_container_width=True)
        use_demo = b2.button("Demo (9 tỉnh)", use_container_width=True)
        dflt = find_default_file()
        if dflt:
            st.caption(f"Mặc định: `{dflt.name}`")

    payload = None
    if use_picked and picked_file is not None:
        payload = (picked_file.name, picked_file.read_bytes())
    elif use_default:
        if dflt:
            payload = (dflt.name, dflt.read_bytes())
        else:
            st.warning("Không tìm thấy file mặc định.")
    elif use_demo:
        payload = ("demo_panel.csv", make_demo_panel().to_csv(index=False).encode("utf-8"))
    elif up is not None:
        data_u = up.getvalue()
        fp_u = fingerprint(data_u)
        if fp_u != st.session_state.get("_last_upload_fp"):
            st.session_state["_last_upload_fp"] = fp_u
            payload = (up.name, data_u)

    if payload is not None:
        tbl, meta = read_table(payload[1], payload[0])
        same = (st.session_state.get("data_passport") or {}).get("sha256") == meta["sha256"]
        if same and st.session_state.get("df_master") is not None:
            st.info("File không thay đổi (trùng mã SHA-256) — giữ nguyên dữ liệu đã nạp, không tải lại.")
        else:
            st.session_state.update(_table=tbl, _table_meta=meta, _auto_init=True)

    tbl, meta = st.session_state.get("_table"), st.session_state.get("_table_meta")
    if tbl is not None:
        det = detect_columns(tbl)
        tag = meta["sha256"][:8]
        has_master = st.session_state.get("df_master") is not None
        with st.expander("Nhận diện cấu trúc & cột dữ liệu", expanded=not has_master and not det["confident"]):
            cols_all = list(tbl.columns)
            mode_pick = st.radio(
                "Cấu trúc dữ liệu",
                ["wide", "panel"],
                index=0 if det["mode"] == "wide" else 1,
                format_func=lambda m: (
                    "Bảng rộng đa tỉnh — mỗi tỉnh VN là 1 cột giá nội sinh (Y), ngoại sinh (X) giữ nguyên"
                    if m == "wide" else
                    "Panel dọc — có cột không gian (`province`) theo từng dòng"
                ),
                horizontal=True,
                key=f"mode_{tag}",
            )
            dayfirst_help = False
            if mode_pick == "wide":
                k1, k2, k3 = st.columns([1, 2, 1])
                date_col = k1.selectbox(
                    "Cột thời gian",
                    cols_all,
                    index=cols_all.index(det["date"]) if det["date"] in cols_all else 0,
                    key=f"date_{tag}",
                )
                nums = numeric_columns(tbl, exclude=[date_col])
                def_endog = [c for c in det["endog"] if c in nums]
                endog_pick = k2.multiselect(
                    "Các cột nội sinh Y (giá từng tỉnh Việt Nam)",
                    nums,
                    default=def_endog,
                    key=f"endog_{tag}",
                )
                t_opts = endog_pick if endog_pick else nums
                def_t = det["target"] if det["target"] in t_opts else (t_opts[0] if t_opts else None)
                target_pick = k3.selectbox(
                    "Biến trọng tâm Y₁",
                    t_opts,
                    index=t_opts.index(def_t) if def_t in t_opts else 0,
                    key=f"target_w_{tag}",
                ) if t_opts else None
                entity_cols = []
                o1, o2 = st.columns(2)
                dayfirst = o1.checkbox("Ngày dạng dd/mm/yyyy (day-first)", value=dayfirst_help, key=f"df_w_{tag}")
                dup_label = o2.radio(
                    "Khi trùng ngày",
                    ["Lấy trung bình các dòng trùng", "Giữ dòng đầu", "Giữ nguyên (chỉ cảnh báo)"],
                    key=f"dup_w_{tag}",
                )
            else:
                k1, k2, k3 = st.columns(3)
                date_col = k1.selectbox(
                    "Cột thời gian",
                    cols_all,
                    index=cols_all.index(det["date"]) if det["date"] in cols_all else 0,
                    key=f"date_p_{tag}",
                )
                entity_cols = k2.multiselect(
                    "Cột không gian (tỉnh/khu vực; có thể thêm loại cà phê)",
                    cols_all,
                    default=[c for c in det["entity"] if c in cols_all],
                    key=f"ent_{tag}",
                )
                nums = numeric_columns(tbl, exclude=[date_col] + entity_cols)
                def_t = "coffee_price_vn" if "coffee_price_vn" in nums else (det["target"] if det["target"] in nums else (nums[0] if nums else None))
                target_pick = k3.selectbox(
                    "Cột giá mục tiêu Y₁ (Trọng tâm)",
                    nums,
                    index=nums.index(def_t) if def_t in nums else 0,
                    key=f"target_p_{tag}",
                ) if nums else None
                endog_pick = [target_pick] if target_pick else []
                o1, o2 = st.columns(2)
                dayfirst = o1.checkbox("Ngày dạng dd/mm/yyyy (day-first)", value=dayfirst_help, key=f"df_p_{tag}")
                dup_label = o2.radio(
                    "Khi trùng khoá (thực thể + ngày)",
                    ["Lấy trung bình các dòng trùng", "Giữ dòng đầu", "Giữ nguyên (chỉ cảnh báo)"],
                    key=f"dup_p_{tag}",
                )
            dup_policy = {"Lấy trung bình các dòng trùng": "mean", "Giữ dòng đầu": "first",
                          "Giữ nguyên (chỉ cảnh báo)": "keep"}[dup_label]
            apply = st.button("Áp dụng & khởi tạo dữ liệu", key=f"apply_{tag}")

        auto = st.session_state.pop("_auto_init", False) and det["confident"]
        if apply or auto:
            if mode_pick == "wide":
                if init_wide(tbl, meta, date_col, endog_pick, target_pick, dayfirst, dup_policy):
                    st.success("Đã khởi tạo hệ sinh thái dữ liệu bảng rộng đa tỉnh (các tỉnh VN là biến nội sinh).")
            else:
                if init_panel(tbl, meta, date_col, entity_cols, target_pick, dayfirst, dup_policy):
                    st.success("Đã khởi tạo hệ sinh thái dữ liệu bảng (Panel Data).")

    master = st.session_state.get("df_master")
    if master is not None:
        pp = st.session_state["data_passport"]
        is_wide = pp.get("mode") == "wide"
        st.markdown("#### 2 · Thẻ tóm tắt")
        if is_wide:
            m = st.columns(3)
            m[0].metric("Tổng số dòng (ngày)", f"{pp['rows']:,}")
            m[1].metric("Biến nội sinh Y (tỉnh VN)", pp["n_entities"])
            m[2].metric("Biến ngoại sinh X", len(pp["exog_cols"]))
        else:
            rp = list(pp["rows_per_entity"].values())
            m = st.columns(3)
            m[0].metric("Tổng số dòng", f"{pp['rows']:,}")
            m[1].metric("Tổng số tỉnh/thực thể", pp["n_entities"])
            m[2].metric("Dữ liệu mỗi tỉnh (TB)", f"{pp['rows_per_entity_mean']:,.0f}",
                        delta=None if pp["balanced"] else f"{min(rp):,} – {max(rp):,}", delta_color="off")
        st.markdown("#### Khoảng thời gian")
        st.metric("Từ ngày", f"{pp['min_date']:%d/%m/%Y} – {pp['max_date']:%d/%m/%Y}")

        st.markdown("#### 3 · Xem trước dữ liệu")
        p1, p2 = st.columns(2)
        p1.caption("10 dòng đầu")
        p1.dataframe(master.head(10), width="stretch")
        p2.caption("10 dòng cuối")
        p2.dataframe(master.tail(10), width="stretch")

        st.markdown("#### 4 · Xác nhận cấu trúc")
        if is_wide:
            lbls = pp["endog_labels"]
            endog_desc = ", ".join(f"**{lbls.get(c, c)}** (`{c}`)" for c in pp["endog_cols"])
            st.success(
                f"Dữ liệu được nhận diện là dạng **Bảng rộng đa tỉnh (Wide Multivariate)** — không phải Panel dọc. "
                f"Hệ thống giữ nguyên **{pp['rows']:,}** ngày quan sát; gom **{len(pp['endog_cols'])}** cột giá tỉnh Việt Nam "
                f"làm biến nội sinh Y ({endog_desc}) và giữ nguyên **{len(pp['exog_cols'])}** cột ngoại sinh X."
            )
            if pp["derived_endog"]:
                st.info(
                    "Các cột lợi suất phái sinh từ giá nội sinh được tự động tách khỏi danh sách ngoại sinh X để tránh rò rỉ: "
                    + ", ".join(f"`{c}`" for c in pp["derived_endog"])
                )
        else:
            st.success(f"Dữ liệu được nhận diện là dạng **Panel Data** (đa không gian). Đã ghi nhận "
                       f"**{pp['n_entities']}** địa phương, toàn bộ **{pp['rows']:,}** dòng được bảo toàn (không gộp).")
        if pp["bad_dates"]:
            st.warning(f"{pp['bad_dates']:,} dòng có ngày không đọc được đã bị loại. Nếu ngày dạng dd/mm/yyyy, hãy tick "
                       "“day-first” rồi áp dụng lại.")
        if pp["duplicate_keys"]:
            key_lbl = "ngày" if is_wide else "thực thể + ngày"
            st.warning(f"Phát hiện {pp['duplicate_keys']:,} dòng trùng khoá ({key_lbl}); chính sách: {pp['dup_policy']}.")
        if not is_wide and not pp["balanced"]:
            st.warning("Panel **không cân bằng**: các tỉnh có số dòng/khoảng thời gian khác nhau (xem bảng chất lượng). "
                       "Khi huấn luyện toàn cục cần dùng chung một mốc ngày chia Train/Test.")

        with st.expander("Chất lượng theo tỉnh · Vai trò/Phạm vi biến · Dấu vân tay"):
            if is_wide:
                st.markdown(f"**Biến nội sinh (Giá các tỉnh Việt Nam)** — tần suất suy ra `{pp['freq']}`")
                st.dataframe(pp["entity_table"], width="stretch", hide_index=True)
                st.markdown("**Danh sách toàn bộ biến số & vai trò**")
                st.dataframe(pp["column_table"], width="stretch", hide_index=True)
            else:
                st.markdown("**Theo tỉnh** (số dòng kỳ vọng suy ra từ tần suất " + f"`{pp['freq']}`" + ")")
                st.dataframe(pp["entity_table"], width="stretch", hide_index=True)
                st.markdown("**Phạm vi biến** — *Toàn quốc*: giống nhau giữa các tỉnh trong cùng ngày (chỉ có T quan sát độc lập, "
                            "không phải N×T); *Theo tỉnh*: khác nhau giữa các tỉnh.")
                st.dataframe(pp["column_table"], width="stretch", hide_index=True)
            st.caption(f"SHA-256: `{pp['sha256']}` · encoding {pp['encoding']} · phân cách `{pp['delimiter']}` · "
                       f"{pp['bytes'] / 1e6:.2f} MB")

        st.divider()
        train_ratio = st.slider("Tỷ lệ TRAIN (theo ngày — chung cho mọi tỉnh)", 0.5, 0.95,
                                float(st.session_state.get("train_ratio", 0.70)), 0.05)
        st.session_state["train_ratio"] = train_ratio
        te_ = train_end_from_ratio(master, pp["date_col"], train_ratio)
        st.caption(f"TRAIN đến hết ngày **{te_:%d/%m/%Y}**; sau mốc này là TEST cho **mọi** tỉnh (tránh rò rỉ).")

if st.session_state.get("df_master") is None:
    with tab2:
        st.info("Hãy nạp dữ liệu ở Tab 1 trước.")
    with tab3:
        st.info("Hãy nạp dữ liệu ở Tab 1 trước.")
    st.stop()

master = st.session_state["df_master"]
meta_p = st.session_state["panel_meta"]
is_wide = meta_p.get("mode") == "wide"
date_col, entity_col, target = meta_p["date_col"], meta_p["entity_col"], meta_p["target"]
endog_cols = meta_p.get("endog_cols") or [target]
endog_labels = meta_p.get("endog_labels") or {}
num_cols = meta_p["num_cols"]
if is_wide:
    x_all = [c for c in num_cols if c not in endog_cols]
    check_cols = endog_cols + x_all
else:
    x_all = [c for c in num_cols if c != target]
    check_cols = num_cols
train_end = train_end_from_ratio(master, date_col, st.session_state.get("train_ratio", 0.7))


@st.cache_data(show_spinner="Đang nội suy từng tỉnh…")
def cached_interp(master_, date_col_, entity_col_, cols_, train_end_, method_):
    return panel_interpolate(master_, date_col_, entity_col_, cols_, train_end_, method_)


@st.cache_data(show_spinner="Đang nội suy chuỗi dữ liệu rộng…")
def cached_interp_wide(master_, date_col_, cols_, train_end_, method_):
    return wide_interpolate(master_, date_col_, cols_, train_end_, method_)


# =============================================================================== TAB 2
with tab2:
    if is_wide:
        step_box("Khoảng trống được đo trên từng cột (nội sinh &amp; ngoại sinh). Nội suy <b>chống look-ahead</b>: "
                 "đoạn TRAIN dùng phương pháp đã chọn (chỉ nội suy bên trong), đoạn TEST chỉ forward-fill.")
        rep = wide_gap_report(master, date_col, check_cols, train_end)
        gap_col = "Max gap (dòng)"
        protected = set(endog_cols)
    else:
        step_box("Khoảng trống được đo <b>trong từng tỉnh</b> (không qua ranh giới hai tỉnh). Nội suy <b>chống look-ahead</b>: "
                 "đoạn TRAIN dùng phương pháp đã chọn (chỉ nội suy bên trong), đoạn TEST chỉ forward-fill. Cùng một mốc ngày cho mọi tỉnh.")
        rep = panel_gap_report(master, entity_col, check_cols, date_col, train_end)
        gap_col = "Max gap (dòng, trong 1 tỉnh)"
        protected = {target}

    c1, c2, c3 = st.columns(3)
    gap_thr = c1.number_input("Ngưỡng max gap (số dòng liên tiếp)", 1, 2000, 30)
    miss_thr = c2.slider("Ngưỡng % khuyết tối đa (toàn bộ)", 5, 90, 40)
    method_label = c3.selectbox("Phương pháp nội suy", list(INTERP_METHODS))
    method = INTERP_METHODS[method_label]
    rep["Đề xuất loại"] = np.where((~rep["Biến"].isin(protected)) & ((rep[gap_col] > gap_thr)
                                                                      | (rep["% khuyết (toàn bộ)"] > miss_thr)), "🗑️ Loại", "")
    st.dataframe(rep.sort_values(gap_col, ascending=False), width="stretch", hide_index=True)
    drop_gap = st.multiselect("Biến X bị loại ở bước khoảng trống", x_all,
                              default=rep.loc[rep["Đề xuất loại"] != "", "Biến"].tolist())
    after_cols = [c for c in check_cols if c not in drop_gap]
    if is_wide:
        interp = cached_interp_wide(master, date_col, after_cols, train_end, method)
    else:
        interp = cached_interp(master, date_col, entity_col, after_cols, train_end, method)

    lost = len(master) - len(interp)
    if lost:
        st.warning(f"{lost:,} dòng đầu chuỗi bị loại vì chưa có giá trị hợp lệ sau nội suy.")
    gappy = [c for c in after_cols if master[c].isna().any()]
    if gappy:
        if is_wide:
            pv = st.selectbox("Xem trước nội suy cho biến", gappy)
            a_, b_ = master.sort_values(date_col), interp.sort_values(date_col)
        else:
            v1, v2 = st.columns(2)
            pv = v1.selectbox("Xem trước nội suy cho biến", gappy)
            pe = v2.selectbox("Tỉnh", st.session_state["ds_tinh"])
            a_, b_ = master[master[entity_col] == pe], interp[interp[entity_col] == pe]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a_[date_col], y=a_[pv], name="Gốc (có khuyết)", mode="lines+markers", marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=b_[date_col], y=b_[pv], name=f"Sau nội suy ({method_label})", line=dict(dash="dot")))
        fig.add_vline(x=train_end, line_dash="dash", annotation_text="TRAIN | TEST")
        fig.update_layout(height=320, margin=dict(t=30, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    if is_wide:
        n_x_kept = len([c for c in after_cols if c not in endog_cols])
        st.success(f"Sau bước này: {len(endog_cols)} biến nội sinh Y · {n_x_kept} biến X · {len(interp):,} ngày.")
    else:
        st.success(f"Sau bước này: {len(after_cols) - 1} biến X · {len(interp):,} dòng · {interp[entity_col].nunique()} tỉnh.")

# =============================================================================== TAB 3
with tab3:
    if is_wide:
        step_box("Loại biến X có tỷ lệ sai phân bằng 0 cao (tần suất giả) trên đoạn TRAIN; rồi kiểm tra đa cộng tuyến "
                 "bằng ρ và VIF trên <b>sai phân bậc 1 đoạn TRAIN</b>.")
        x_after = [c for c in after_cols if c not in endog_cols]
        zd_series = wide_zero_diff(interp, date_col, x_after, train_end) if x_after else pd.Series(dtype=float)
    else:
        step_box("Loại biến X có tỷ lệ sai phân bằng 0 cao (tần suất giả) — tính <b>trong từng tỉnh</b>; rồi lọc đa cộng tuyến "
                 "bằng ρ và VIF trên <b>sai phân bậc 1 xếp chồng của đoạn TRAIN toàn panel</b> (sai phân trong từng tỉnh, không băng qua ranh giới tỉnh).")
        x_after = [c for c in after_cols if c != target]
        zd_series = panel_zero_diff(interp, date_col, entity_col, x_after, train_end) if x_after else pd.Series(dtype=float)

    zr = pd.DataFrame({"Biến": x_after, "Tỷ lệ Δ=0 (TRAIN)": zd_series.to_numpy()})
    s1, s2 = st.columns([1, 2])
    static_thr = s1.slider("Ngưỡng tỷ lệ Δ=0", 0.1, 1.0, 0.95, 0.05,
                           help="Chỉ loại các biến hầu như bất biến (ví dụ >95% ngày không đổi). Đặt cao để giữ biến mùa vụ/tồn kho.")
    zr["Đề xuất loại"] = np.where(zr["Tỷ lệ Δ=0 (TRAIN)"] >= static_thr, "🗑️ Loại", "")
    with s2:
        st.dataframe(zr.style.format({"Tỷ lệ Δ=0 (TRAIN)": "{:.1%}"}), width="stretch", hide_index=True)

    keep_all_x = st.checkbox("Giữ lại toàn bộ biến nghiên cứu (Không tự động ép loại biến)", value=True,
                             help="Bỏ chọn nếu muốn hệ thống tự động loại bớt biến theo ngưỡng")

    drop_static = st.multiselect("Biến X bị loại (tần suất giả / đứng yên)", x_after,
                                 default=[] if keep_all_x else zr.loc[zr["Đề xuất loại"] != "", "Biến"].tolist())
    cand = [c for c in x_after if c not in drop_static]
    removal_multi = pd.DataFrame(columns=["Biến", "Lý do", "Đại diện giữ lại"])
    drop_multi, final_x = [], cand
    if len(cand) >= 2:
        if is_wide:
            dtr = wide_diff_train_z(interp, date_col, [target] + cand, train_end)
        else:
            dtr = panel_diff_train_z(interp, date_col, entity_col, [target] + cand, train_end)
        k1, k2, k3 = st.columns(3)
        corr_method = k1.radio("Tương quan", ["pearson", "spearman"], horizontal=True)
        rho_thr = k2.slider("Ngưỡng |ρ|", 0.5, 0.99, 0.95, 0.01)
        vif_thr = k3.slider("Ngưỡng VIF", 3.0, 50.0, 20.0, 1.0)
        corr = dtr[cand].corr(method=corr_method)
        hm = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu",
                                  reversescale=True, text=corr.round(2).values, texttemplate="%{text}"))
        hm.update_layout(height=420, margin=dict(t=30, b=10), title=f"Tương quan của ΔX (TRAIN, {len(dtr):,} quan sát)")
        h1, h2 = st.columns([3, 2])
        h1.plotly_chart(hm, width="stretch")
        with h2:
            st.markdown("**Cặp biến |ρ| vượt ngưỡng**")
            pairs = high_corr_pairs(dtr[cand], rho_thr, corr_method)
            st.dataframe(pairs if len(pairs) else pd.DataFrame({"Kết quả": ["Không có cặp nào"]}), width="stretch",
                         hide_index=True)
            st.markdown("**VIF hiện tại**")
            st.dataframe(vif_table(dtr[cand]), width="stretch", hide_index=True)
        removal_multi = suggest_multicollinearity_drops(dtr, target, cand, rho_thr, vif_thr, corr_method)
        drop_multi = st.multiselect("Biến X bị loại (đa cộng tuyến — giữ 1 đại diện/cụm)", cand,
                                    default=[] if keep_all_x else removal_multi["Biến"].tolist())
        final_x = [c for c in cand if c not in drop_multi]
    else:
        st.info("Còn ít hơn 2 biến X — bỏ qua lọc đa cộng tuyến.")

    st.divider()
    st.subheader("Đầu ra → st.session_state")
    log = [(v, "Khoảng trống", f"max gap > {gap_thr} hoặc % khuyết > {miss_thr}%") for v in drop_gap]
    log += [(v, "Tần suất giả", f"Tỷ lệ Δ=0 ≥ {static_thr:.0%}") for v in drop_static]
    for v in drop_multi:
        why = removal_multi.loc[removal_multi["Biến"] == v, "Lý do"]
        log.append((v, "Đa cộng tuyến", why.iloc[0] if len(why) else "Loại thủ công"))
    log_df = pd.DataFrame(log, columns=["Biến", "Giai đoạn", "Lý do"])
    o1, o2 = st.columns([1, 2])
    with o1:
        st.metric("Biến X giữ lại", len(final_x), delta=f"-{len(x_all) - len(final_x)} so với ban đầu", delta_color="off")
        st.write(", ".join(final_x) if final_x else "—")
    with o2:
        st.dataframe(log_df if len(log_df) else pd.DataFrame({"Nhật ký loại biến": ["Chưa loại biến nào"]}),
                     width="stretch", hide_index=True)

    def _default_model2_exog(cols_avail, exclude_cols):
        ret_order = [
            "ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd",
            "temp_mean", "precip_nasa", "humidity_air",
            "temp_mean_ch", "precip_nasa_ch", "humidity_air_ch",
        ]
        lvl_order = [
            "london_price_usd_ton", "arabica_price_usd_lb", "brent_price_usd_barrel", "dxy_index", "usd_vnd_rate",
            "temp_mean", "precip_nasa", "humidity_air",
            "temp_mean_ch", "precip_nasa_ch", "humidity_air_ch",
        ]
        excl = set(exclude_cols)
        out = [c for c in ret_order if c in cols_avail and c not in excl]
        if len(out) < 5:
            out = [c for c in lvl_order if c in cols_avail and c not in excl]
        if not out:
            out = [c for c in cols_avail if c not in excl and not c.startswith("ecm_") and not c.endswith(("_lag1", "_lag2"))][:8]
        return out

    if is_wide:
        primary_y = st.selectbox(
            "Biến nội sinh trọng tâm Y₁ (dùng làm chuẩn đối chiếu cho kiểm định ở Trang 2–3)",
            endog_cols,
            index=endog_cols.index(target) if target in endog_cols else 0,
            format_func=lambda c: f"{endog_labels.get(c, c)} ({c})",
        )
        if st.button("Chốt dữ liệu sạch đa tỉnh", type="primary"):
            ordered_endog = [primary_y] + [c for c in endog_cols if c != primary_y]
            clean = interp[[date_col] + ordered_endog + final_x].copy()
            wide_df = clean.set_index(date_col).sort_index()
            reset_downstream("raw")

            default_exog = _default_model2_exog(clean.columns, ordered_endog)

            pm_new = dict(meta_p, target=primary_y, endog_cols=ordered_endog)
            st.session_state.update(
                data_mode="wide",
                df_clean_panel=clean,
                df_clean=wide_df,
                x_cols=final_x,
                train_end=train_end,
                proxy_province=endog_labels.get(primary_y, primary_y),
                target=primary_y,
                panel_meta=pm_new,
                train_n=int((wide_df.index <= train_end).sum()),
                removal_log=log_df,
                endog_cols=ordered_endog,
                chosen_endog=ordered_endog,
                exog_cols=default_exog,
                chosen_x_features=default_exog,
            )
            st.success(f"Đã lưu dữ liệu sạch đa tỉnh ({len(clean):,} ngày · {len(ordered_endog)} biến nội sinh Y · "
                       f"{len(final_x)} biến ngoại sinh X). Chuyển sang **Trang 2**.")
    else:
        min_cov = st.slider("Độ phủ dữ liệu tối thiểu của tỉnh đại diện", 0.5, 1.0, 0.8, 0.05)
        pt = proxy_table(interp, date_col, entity_col, target, train_end, min_cov)
        st.dataframe(pt.style.format({"Độ phủ": "{:.0%}", "ρ (Δ) với TB các tỉnh còn lại": "{:.3f}"}), width="stretch")
        proxy = st.selectbox("Tỉnh đại diện (dùng cho kiểm định ở Trang 2–3)", list(pt.index), index=0,
                             help="Mặc định: tỉnh đủ độ phủ có biến động Δ giá bám sát nhất với các tỉnh còn lại. Có thể đổi thủ công.")
        if st.button("Chốt dữ liệu sạch & trích tỉnh đại diện", type="primary", disabled=len(final_x) == 0):
            clean = interp[[date_col, entity_col, target] + final_x].copy()
            proxy_df = entity_frame(clean, date_col, entity_col, proxy, [target] + final_x)
            reset_downstream("raw")

            default_endog = [c for c in ["price_vnd_kg", "coffee_price_vn"] if c in clean.columns]
            if not default_endog:
                default_endog = [target]
            default_exog = _default_model2_exog(clean.columns, default_endog)

            st.session_state.update(
                data_mode="panel",
                df_clean_panel=clean,
                df_clean=proxy_df,
                x_cols=final_x,
                train_end=train_end,
                proxy_province=proxy,
                train_n=int((proxy_df.index <= train_end).sum()),
                removal_log=log_df,
                endog_cols=default_endog,
                chosen_endog=default_endog,
                exog_cols=default_exog,
                chosen_x_features=default_exog,
            )
            st.success(f"Đã lưu df_clean_panel ({len(clean):,} dòng, {clean[entity_col].nunique()} tỉnh) và chuỗi đại diện "
                       f"**{proxy}** ({len(proxy_df):,} dòng). Chuyển sang **Trang 2**.")

    if st.session_state.get("df_clean_panel") is not None:
        if is_wide:
            st.caption(f"Đang lưu (Wide): {len(st.session_state.get('endog_cols') or [])} biến Y · "
                       f"{len(st.session_state['x_cols'])} biến X · {len(st.session_state['df_clean_panel']):,} ngày.")
        else:
            st.caption(f"Đang lưu: {len(st.session_state['x_cols'])} biến X · panel {len(st.session_state['df_clean_panel']):,} dòng · "
                       f"tỉnh đại diện {st.session_state['proxy_province']}.")

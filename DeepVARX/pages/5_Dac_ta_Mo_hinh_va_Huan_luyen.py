import time

import numpy as np
import pandas as pd
import streamlit as st

from modules.data_utils import difference, metrics
from modules.model_specs import ORDER, PROFILES, allocation
from modules.models.baseline_varx import run_local_pkg, select_order
from modules.ui import require, setup_page

setup_page("5 · Đặc tả mô hình & Huấn luyện")
st.title("Trang 5 · Đặc tả mô hình & Huấn luyện")
st.caption("Mô hình **Local** (VAR, VARX, VARX-L) được huấn luyện riêng cho từng tỉnh; mô hình **Global** (LSTM, DeepVAR, DeepVARX) "
           "huấn luyện **một lần** trên toàn panel rồi tách dự báo theo tỉnh. Mọi dự báo quy về VNĐ/kg.")
require(["packages", "pkg_meta", "base_set", "ext_set"], "Cần hoàn tất Trang 1–4 (bấm **CẮT PANEL THÀNH 3 GÓI** ở Trang 4).")

pk, meta = st.session_state["packages"], st.session_state["pkg_meta"]
target, date_col, entity_col = meta["target"], meta["date_col"], meta["entity_col"]
ents, d, train_end = meta["entities"], meta["d"], meta["train_end"]
base_set, ext_set = meta["base"], meta["ext"]
jr = max(int(st.session_state.get("johansen_rank") or 1), 1)

try:
    import torch  # noqa: F401
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

name = st.selectbox("Chọn mô hình", ORDER, format_func=lambda m: PROFILES[m]["title"])
prof, alloc = PROFILES[name], allocation(name, base_set, ext_set)
is_linear = prof["family"] == "linear"

# =============================================================================== KHỐI 1
st.subheader("Tổng quan mô hình")
with st.container(border=True):
    st.markdown(f"### {prof['title']}")
    for eq in prof["latex"]:
        st.latex(eq)
    a, b = st.columns(2)
    with a:
        st.markdown("**Công dụng**")
        st.write(prof["use"])
        st.markdown("**Ưu điểm**")
        st.markdown("\n".join(f"- {x}" for x in prof["pros"]))
    with b:
        st.markdown("**Nhược điểm**")
        st.markdown("\n".join(f"- {x}" for x in prof["cons"]))
        st.markdown("**Dữ liệu cấp phát**")
        st.write(alloc["label"])
        st.caption(("Local — huấn luyện riêng từng tỉnh" if alloc["scope"] == "local" else "Global — huấn luyện 1 lần trên toàn panel")
                   + " · Biến: " + ", ".join([target + " (Y)"] + alloc["feats"]))
    st.caption(f"Cài đặt: {prof['impl']}")

# =============================================================================== KHỐI 2
st.subheader("Bảng điều khiển siêu tham số")
hp = {}
levels_in = alloc["pkg"] == "A" and meta["A_mode"] == "levels"
if is_linear:
    kp, kq = f"p_{name}", f"q_{name}"
    st.session_state.setdefault(kp, 2)
    st.session_state.setdefault(kq, 1)
    has_x = bool(alloc["feats"])
    pkg_df = pk[alloc["pkg"]]
    groups = dict(tuple(pkg_df.groupby(entity_col, observed=True)))

    def order_for(g, crit_, mp, mq):
        cols = [target] + alloc["feats"]
        g = g.sort_values(date_col)
        dd = difference(g[cols], d).to_numpy(float) if levels_in else g[cols].to_numpy(float)
        off = d if levels_in else 0
        rows_end = int((g["split"].to_numpy()[off:] == "train").sum())
        return select_order(dd[:, :1], dd[:, 1:] if has_x else None, rows_end, int(mp), int(mq), crit_)

    c1, c2, c3 = st.columns([1, 1, 2])
    crit = c1.radio("Tiêu chí gợi ý", ["AIC", "BIC"], horizontal=True, key=f"crit_{name}")
    max_p = c2.number_input("p tối đa (tìm kiếm)", 1, 30, 8, key=f"maxp_{name}")
    max_q = c3.number_input("q tối đa (tìm kiếm)", 1, 30, 5, key=f"maxq_{name}", disabled=not has_x)
    if st.button("Gợi ý tự động (p, q) từ AIC/BIC (trên tỉnh đại diện)"):
        proxy = st.session_state.get("proxy_province") or ents[0]
        grid, best = order_for(groups[proxy if proxy in groups else ents[0]], crit, max_p, max_q)
        st.session_state[kp], st.session_state[kq] = best
        st.session_state[f"grid_{name}"] = (grid, crit)
    if st.session_state.get(f"grid_{name}"):
        grid, c_used = st.session_state[f"grid_{name}"]
        st.caption(f"Top gợi ý theo {c_used} (đã điền vào ô bên dưới):")
        st.dataframe(grid.head(5).style.format({c_used: "{:.1f}"}), width="stretch", hide_index=True)
    q1, q2, q3 = st.columns(3)
    hp["p"] = int(q1.number_input("Bậc trễ p (của Y)", 1, 30, key=kp))
    hp["q"] = int(q2.number_input("Bậc trễ q (của X)", 1, 30, key=kq, disabled=not has_x)) if has_x else 0
    hp["alpha"] = float(q3.number_input("Hệ số phạt Lasso λ", 0.0001, 5.0, 0.05, format="%.4f", key=f"lam_{name}")) \
        if name == "VARX-L" else None
    hp["per_prov"] = st.checkbox("Tự chọn (p, q) riêng cho từng tỉnh theo AIC/BIC (bỏ qua ô p, q ở trên)", value=False,
                                 key=f"pp_{name}")
    hp["ect"] = 0
    if name == "VARX" and meta["A_mode"] == "levels" and has_x:
        use_ect = st.checkbox(f"Dùng hiệu chỉnh sai số (VECM-X): thêm ECT với hạng r = {jr} từ Johansen", value=True,
                              key="use_ect")
        hp["ect"] = jr if use_ect else 0
else:
    c1, c2, c3, c4 = st.columns(4)
    hp["epochs"] = int(c1.number_input("Số Epochs", 5, 500, 40, key=f"ep_{name}"))
    hp["batch"] = int(c2.selectbox("Batch Size", [64, 128, 256, 512], index=2, key=f"bs_{name}"))
    hp["lr"] = float(c3.select_slider("Learning Rate", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-3,
                                      format_func=lambda x: f"{x:g}", key=f"lr_{name}"))
    hp["hidden"] = int(c4.selectbox("Hidden Size", [16, 32, 64, 128], index=2, key=f"hs_{name}"))
    e1, e2, e3, e4, e5 = st.columns(5)
    hp["seq_len"] = int(e1.number_input("Độ dài chuỗi", 3, 120, 30, key=f"sl_{name}"))
    hp["layers"] = int(e2.number_input("Số lớp LSTM", 1, 4, 1, key=f"ly_{name}"))
    hp["emb"] = int(e3.number_input("Chiều embedding tỉnh", 0, 32, 4, key=f"em_{name}",
                                    help="0 = không dùng mã hóa tỉnh"))
    hp["early"] = bool(e4.checkbox("Bật Early Stopping", value=True, key=f"es_{name}"))
    hp["patience"] = int(e5.number_input("Patience", 2, 100, 8, key=f"pa_{name}", disabled=not hp["early"]))
    if not TORCH_OK:
        st.error("Chưa cài PyTorch — chạy `pip install torch` để huấn luyện mô hình học sâu.")
    st.caption(f"Mô hình Global học trên toàn panel ({len(ents)} tỉnh) nên một epoch lâu hơn nhiều so với một tỉnh; nên thử ít epoch trước.")

# =============================================================================== KHỐI 3
st.subheader("Cửa sổ huấn luyện")
blocked = (name == "VARX" and not base_set) or (not is_linear and not TORCH_OK)
if name == "VARX" and not base_set:
    st.warning("Tập Cơ Sở rỗng nên VARX không có biến ngoại sinh (hãy dùng VAR hoặc chỉnh lại ở Trang 4).")
run = st.button(f"Khởi chạy Huấn luyện [{name}]", type="primary", disabled=blocked)
bar = st.progress(0.0)
term = st.empty()
store = st.session_state.get("results") or {}
prev = store.get(name)

if run:
    lines = []

    def log(msg: str):
        lines.append(msg)
        term.code("\n".join(lines[-25:]), language="text")

    t0 = time.perf_counter()
    log(f"$ train --model {name} --scope {alloc['scope']} --data \"{alloc['label']}\"")
    out = {"scope": alloc["scope"], "feats": alloc["feats"], "hp": hp, "history": None, "attn": None}
    if is_linear:
        preds, models = [], {}
        for i, e in enumerate(ents, 1):
            g = groups[e]
            p_, q_ = (hp["p"], hp["q"])
            if hp["per_prov"]:
                _, best = order_for(g, crit, max_p, max_q)
                p_, q_ = best[0], (best[1] if has_x else 0)
            res, mdl = run_local_pkg(g, date_col, target, alloc["feats"], d, p_, q_, hp["alpha"], hp["ect"], 2, levels_in)
            res.insert(1, "province", e)
            preds.append(res)
            models[e] = mdl
            m = metrics(res["y_true"], res["y_pred"])
            log(f"[{i}/{len(ents)}] {e}: p={p_} q={q_} · RMSE={m['RMSE']:,.1f} · MAPE={m['MAPE (%)']:.2f}%")
            bar.progress(i / len(ents))
        pred = pd.concat(preds, ignore_index=True)
        out["model"] = models
    else:
        from modules.models.deepvarx import DeepVARX, prepare_panel_windows, train_model
        from modules.models.lstm_attention import LSTMForecaster

        data = prepare_panel_windows(pk["C"], date_col, entity_col, target, alloc["feats"], hp["seq_len"], train_end,
                                     meta["scalers"])
        n_in = len(data["cols"])
        Cls = DeepVARX if name == "DeepVARX" else LSTMForecaster
        model = Cls(n_in, hp["hidden"], hp["layers"], 0.1, len(ents), hp["emb"])
        log(f"[model] {name} · {n_in} biến vào · hidden={hp['hidden']} · embedding tỉnh={hp['emb']} · "
            f"windows train/val/test={len(data['Xtr']):,}/{len(data['Xva']):,}/{len(data['Xte']):,}")

        def cb(ep, total, trl, val):
            bar.progress(min(1.0, ep / total))
            log(f"Epoch {ep:03d}/{total:03d} - loss: {trl:.5f} - val_loss: {val:.5f}")

        tr = train_model(model, data, hp["epochs"], hp["lr"], hp["batch"],
                         hp["patience"] if hp["early"] else hp["epochs"] + 1, 42, name == "DeepVARX", cb,
                         restore_best=hp["early"])
        if hp["early"]:
            log(f"[early-stopping] epoch tốt nhất = {tr['best_epoch']}")
        pred = pd.DataFrame({"date": data["dates_te"], "province": data["prov_te"], "y_true": data["true_te"],
                             "y_pred": tr["pred_scaled"] * data["sd_te"] + data["mu_te"], "y_prev": data["prev_te"]})
        out.update(model=model, history=tr["history"])
        if "attn_test" in tr:
            out["attn"] = {"cols": data["cols"], "test": tr["attn_test"], "dates": data["dates_te"], "prov": data["prov_te"],
                           "epoch": tr.get("attn_epoch"), "lag": tr["attn_lag"], "time": tr["attn_time"],
                           "best_epoch": tr["best_epoch"]}
    by = pd.DataFrame([{"province": e, **metrics(g["y_true"], g["y_pred"])} for e, g in pred.groupby("province")])
    avg = {k: float(by[k].mean()) for k in ("RMSE", "MAE", "MAPE (%)")}
    out.update(pred=pred, by_prov=by, metrics=avg, seconds=time.perf_counter() - t0)
    log(f"[done] TB {len(by)} tỉnh: RMSE={avg['RMSE']:,.1f} · MAE={avg['MAE']:,.1f} · MAPE={avg['MAPE (%)']:.2f}% (VNĐ/kg) · "
        f"{out['seconds']:.1f}s")
    out["log"] = "\n".join(lines)
    store[name] = out
    st.session_state["results"] = store
    st.success(f"Đã huấn luyện {name} ({alloc['scope']}) và lưu vào session_state.")
elif prev is not None:
    bar.progress(1.0)
    term.code(prev["log"], language="text")
    m = prev["metrics"]
    st.caption(f"Kết quả đã lưu (trung bình {len(prev['by_prov'])} tỉnh): RMSE {m['RMSE']:,.1f} · MAE {m['MAE']:,.1f} · MAPE {m['MAPE (%)']:.2f}%")

# =============================================================================== TRẠNG THÁI
st.divider()
trained = st.session_state.get("results") or {}
rows = [{"Mô hình": m, "Phạm vi": allocation(m, base_set, ext_set)["scope"],
         "Trạng thái": "Đã huấn luyện" if m in trained else "⬜ Chưa",
         "RMSE TB (VNĐ/kg)": trained[m]["metrics"]["RMSE"] if m in trained else np.nan} for m in ORDER]
st.dataframe(pd.DataFrame(rows).style.format({"RMSE TB (VNĐ/kg)": "{:,.1f}"}, na_rep="–"), width="stretch", hide_index=True)
if trained:
    st.page_link("pages/6_Bao_cao_Thuc_nghiem.py", label="Sang Trang 6 · Báo cáo thực nghiệm →", icon="📑")

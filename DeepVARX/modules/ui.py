"""Helper giao diện Streamlit dùng chung cho mọi trang."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "style.css"

STEPS = [
    ("df_clean_panel", "Trang 1 · Dữ liệu sạch (toàn panel)"),
    ("d", "Trang 2 · Bậc d & kiểm định tuyến tính"),
    ("evidence_nl", "Trang 3 · Bằng chứng phi tuyến (XGBoost)"),
    ("packages", "Trang 4 · Gói dữ liệu A/B/C"),
    ("results", "Trang 5 · Mô hình đã huấn luyện"),
]
_EV = ["evidence_lin", "evidence_nl", "base_set", "ext_set", "ext_sources", "packages", "pkg_meta", "export_paths",
       "results", "base_model", "johansen_rank"]
DOWNSTREAM = {
    "raw": ["df_clean_panel", "df_clean", "x_cols", "train_n", "train_end", "proxy_province", "removal_log", "d",
            "vecm_hint"] + _EV,
    "df_clean": ["d", "vecm_hint"] + _EV,
    "base_set": ["ext_set", "ext_sources", "packages", "pkg_meta", "export_paths", "results"],
}


def reset_downstream(step: str) -> None:
    for k in DOWNSTREAM.get(step, []):
        st.session_state.pop(k, None)


def setup_page(title: str, icon: str = "☕") -> None:
    st.set_page_config(page_title=f"DeepVARX · {title}", page_icon=icon, layout="wide")
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    sidebar_status()


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown("### Tiến độ")
        for key, label in STEPS:
            done = st.session_state.get(key) is not None
            st.markdown(f"{'✅' if done else '⬜'} {label}")
        if st.session_state.get("df_master") is not None:
            st.divider()
            st.caption(f"Y: **{st.session_state.get('target')}** · {len(st.session_state.get('ds_tinh') or [])} tỉnh")
            if st.session_state.get("proxy_province"):
                st.caption(f"Tỉnh đại diện: **{st.session_state['proxy_province']}**")
            if st.session_state.get("base_set") is not None:
                st.caption(f"Tập Cơ Sở: {len(st.session_state['base_set'])} biến")
            if st.session_state.get("ext_set") is not None:
                st.caption(f"Tập Mở Rộng: {len(st.session_state['ext_set'])} biến")


def require(keys: list[str], hint: str) -> None:
    missing = [k for k in keys if st.session_state.get(k) is None]
    if missing:
        st.warning(hint)
        st.stop()


def step_box(text: str) -> None:
    st.markdown(f"<div class='dv-step'>{text}</div>", unsafe_allow_html=True)


def fmt_p(x: float) -> str:
    if x != x:
        return "–"
    return "<0.0001" if x < 1e-4 else f"{x:.4f}"

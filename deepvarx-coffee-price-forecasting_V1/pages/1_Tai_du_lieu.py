import streamlit as st
import pandas as pd
import numpy as np
import os

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "coffee_price_cleaned.csv")

st.set_page_config(page_title="Dữ liệu & Tiền xử lý", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()

st.caption(APP_HEADER)
st.title("Trang 1 — Dữ liệu & Tiền xử lý")
st.markdown("Nạp dữ liệu, kiểm tra chất lượng và thực hiện tiền xử lý tự động trước khi đưa vào phân tích.")

# ── LOAD DATA ──────────────────────────────────────────────────────────────
st.subheader("1. Nạp dữ liệu")

load_mode = st.radio(
    "Nguồn dữ liệu",
    ["Dùng file mặc định (coffee_price_cleaned.csv)", "Upload file khác"],
    horizontal=True,
)

df_raw = None

if load_mode.startswith("Dùng file"):
    if os.path.exists(DEFAULT_PATH):
        try:
            df_raw = pd.read_csv(DEFAULT_PATH, sep="\t")
            st.success(f"Đã tải: `{os.path.basename(DEFAULT_PATH)}`")
        except Exception as e:
            st.error(f"Lỗi khi đọc file mặc định: {e}")
    else:
        st.error(f"Không tìm thấy file tại: `{DEFAULT_PATH}`")
else:
    uploaded = st.file_uploader("Chọn file CSV hoặc Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                # Thử tab trước, fallback sang comma
                raw_bytes = uploaded.read()
                uploaded.seek(0)
                first_line = raw_bytes.decode("utf-8", errors="ignore").split("\n")[0]
                sep = "\t"if first_line.count("\t") > first_line.count(",") else ","
                df_raw = pd.read_csv(uploaded, sep=sep)
            else:
                df_raw = pd.read_excel(uploaded)
            st.success(f"Đã tải: `{uploaded.name}`")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

if df_raw is not None:
    st.session_state["raw_data"] = df_raw

df_raw = st.session_state.get("raw_data")

if df_raw is None:
    st.info("Chưa có dữ liệu. Vui lòng chọn nguồn dữ liệu phía trên.")
    st.stop()

# ── PARSE DATE ─────────────────────────────────────────────────────────────
df_raw = df_raw.copy()
date_col = "date"if "date"in df_raw.columns else None
if date_col:
    try:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], format="%m/%d/%Y", errors="coerce")
        if df_raw[date_col].isna().sum() > 0:
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], infer_datetime_format=True, errors="coerce")
    except Exception:
        pass

# ── THÔNG TIN DỮ LIỆU TRƯỚC XỬ LÝ ─────────────────────────────────────────
st.divider()
st.subheader("2. Thông tin dữ liệu (trước xử lý)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số dòng", f"{df_raw.shape[0]:,}")
c2.metric("Số biến", df_raw.shape[1])

if date_col and pd.api.types.is_datetime64_any_dtype(df_raw[date_col]):
    t_min = df_raw[date_col].min().strftime("%d/%m/%Y")
    t_max = df_raw[date_col].max().strftime("%d/%m/%Y")
    c3.metric("Từ ngày", t_min)
    c4.metric("Đến ngày", t_max)

n_provinces = df_raw["province"].nunique() if "province"in df_raw.columns else "-"
st.caption(f"Số tỉnh/thành: **{n_provinces}** | Tab-separated CSV")

# Xem nhanh
with st.expander("Xem trước dữ liệu"):
    st.dataframe(df_raw.head(30), use_container_width=True)

# ── PHÂN TÍCH MISSING ──────────────────────────────────────────────────────
st.divider()
st.subheader("3. Phân tích Missing Values")

miss_count = df_raw.isna().sum()
miss_pct = (miss_count / len(df_raw) * 100).round(3)
miss_df = pd.DataFrame({
    "Biến": miss_count.index,
    "Số missing": miss_count.values,
    "% missing": miss_pct.values,
    "Kiểu dữ liệu": df_raw.dtypes.astype(str).values,
})
miss_df = miss_df.sort_values("Số missing", ascending=False).reset_index(drop=True)

total_miss = int(miss_count.sum())
n_vars_miss = int((miss_count > 0).sum())

c1, c2 = st.columns(2)
c1.metric("Tổng số ô missing", f"{total_miss:,}")
c2.metric("Số biến có missing", n_vars_miss)

if total_miss > 0:
    st.warning(f"{n_vars_miss} biến có giá trị thiếu.")
    st.dataframe(
        miss_df[miss_df["Số missing"] > 0].style.format({"% missing": "{:.3f}%"}),
        use_container_width=True, height=220,
    )
else:
    st.success("Không có giá trị thiếu.")
    st.dataframe(miss_df.head(10), use_container_width=True, height=220)

# ── PHÂN TÍCH OUTLIER (IQR) ─────────────────────────────────────────────────
st.divider()
st.subheader("4. Phân tích Outlier (phương pháp IQR × 1.5)")
st.caption("Outlier được xác định khi giá trị < Q1 − 1.5×IQR hoặc > Q3 + 1.5×IQR.")

numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()

outlier_rows = []
for col in numeric_cols:
    s = df_raw[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    outlier_rows.append({
        "Biến": col,
        "Q1": round(float(q1), 4),
        "Q3": round(float(q3), 4),
        "IQR": round(float(iqr), 4),
        "Ngưỡng dưới": round(float(lo), 4),
        "Ngưỡng trên": round(float(hi), 4),
        "Số outlier": n_out,
        "% outlier": round(n_out / len(df_raw) * 100, 3),
    })

outlier_df = pd.DataFrame(outlier_rows).sort_values("Số outlier", ascending=False).reset_index(drop=True)
total_out = int(outlier_df["Số outlier"].sum())
n_vars_out = int((outlier_df["Số outlier"] > 0).sum())

c1, c2 = st.columns(2)
c1.metric("Tổng outlier", f"{total_out:,}")
c2.metric("Số biến có outlier", n_vars_out)

st.dataframe(
    outlier_df.style.format({
        "% outlier": "{:.3f}%",
        "Q1": "{:.4f}", "Q3": "{:.4f}", "IQR": "{:.4f}",
        "Ngưỡng dưới": "{:.4f}", "Ngưỡng trên": "{:.4f}",
    }),
    use_container_width=True, height=280,
)

# ── TỰ ĐỘNG TIỀN XỬ LÝ ─────────────────────────────────────────────────────
st.divider()
st.subheader("5. Tự động tiền xử lý")

st.markdown("""
**Phương pháp xử lý sẽ được áp dụng:**
| Bước | Đối tượng | Phương pháp | Lý do |
|------|-----------|-------------|-------|
| 1 | Missing values | **Forward fill (ffill)** rồi **Backward fill (bfill)** | Dữ liệu chuỗi thời gian liên tục theo ngày, giá trị liền kề là xấp xỉ tốt nhất |
| 2 | Outlier (số) | **Winsorizing IQR × 1.5** — clip về ngưỡng dưới/trên | Giữ nguyên cấu trúc dữ liệu, không xóa dòng, tránh mất thông tin thời gian |
| 3 | Cột thời gian | Parse `date`sang `datetime`(định dạng `%m/%d/%Y`) | Đảm bảo sắp xếp đúng thứ tự thời gian |
""")

if st.button("Bắt đầu tiền xử lý tự động", type="primary", use_container_width=True):
    with st.spinner("Đang xử lý..."):
        df_proc = df_raw.copy()

        # 1. Parse date
        if date_col and date_col in df_proc.columns:
            df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors="coerce")
            df_proc = df_proc.sort_values([date_col]).reset_index(drop=True)

        # 2. Missing — ffill + bfill theo từng tỉnh
        if "province"in df_proc.columns:
            df_proc = df_proc.groupby("province", group_keys=False).apply(
                lambda g: g.sort_values(date_col).ffill().bfill() if date_col else g.ffill().bfill()
            )
        else:
            df_proc = df_proc.ffill().bfill()

        miss_after = int(df_proc.isna().sum().sum())

        # 3. Outlier — winsorize (IQR × 1.5) cho các cột số
        clipped = 0
        for col in numeric_cols:
            if col in df_proc.columns:
                s = df_proc[col]
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_before = int(((s < lo) | (s > hi)).sum())
                df_proc[col] = s.clip(lower=lo, upper=hi)
                clipped += n_before

        st.session_state["processed_data"] = df_proc
        st.session_state["preprocess_done"] = True
        st.session_state["date_col"] = date_col or "date"
        st.session_state["group_col"] = "province"if "province"in df_proc.columns else None

    st.success("Tiền xử lý hoàn tất!")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missing đã xử lý", f"{total_miss:,} → {miss_after:,}")
    c2.metric("Outlier đã clip", f"{clipped:,}")
    c3.metric("Dữ liệu sau xử lý", f"{df_proc.shape[0]:,} dòng")
    c4.metric("Số biến", df_proc.shape[1])

    st.caption("Phương pháp: Missing = ffill→bfill theo tỉnh | Outlier = Winsorizing IQR×1.5")

# Hiển thị nếu đã xử lý trước đó
proc = st.session_state.get("processed_data")
if proc is not None and not st.session_state.get("preprocess_done"):
    st.info(f"Dữ liệu đã được xử lý trước đó: {proc.shape[0]:,} dòng × {proc.shape[1]} cột.")

st.divider()
if st.button("Tiếp theo: Phân tích tương quan →", use_container_width=True):
    st.switch_page("pages/2_Phan_tich_tuong_quan.py")

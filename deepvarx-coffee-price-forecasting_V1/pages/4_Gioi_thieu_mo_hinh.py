import streamlit as st

APP_HEADER = "☕ DeepVARX — Hệ thống dự báo giá cà phê xuất khẩu Việt Nam"

st.set_page_config(page_title="Giới thiệu mô hình", page_icon="☕", layout="wide")

with st.sidebar:
    st.markdown("## ☕ DeepVARX")
    st.caption("Dự báo giá cà phê xuất khẩu Việt Nam")
    st.divider()

st.caption(APP_HEADER)
st.title("Trang 4 — Giới thiệu mô hình")
st.markdown("Tổng quan lý thuyết và kiến trúc của **4 mô hình** được thực nghiệm trong đề tài.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["VAR", "VARX", "DeepVAR", "DeepVARX"])

# ── VAR ─────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("VAR — Vector AutoRegression")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        **Mô tả**: VAR mô hình hoá hệ thống nhiều chuỗi thời gian nội sinh, trong đó mỗi biến
        được giải thích bởi các giá trị trễ của **chính nó** và **tất cả biến khác** trong hệ.

        **Điều kiện áp dụng**:
        - Tất cả chuỗi phải **dừng** (I(0)) hoặc đã sai phân
        - Không có biến ngoại sinh

        **Tham số**:
        - **p**: độ trễ (lag order) — xác định qua AIC/BIC/HQC
        """)
        st.latex(r"""
        \mathbf{y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{y}_{t-1} + \mathbf{A}_2 \mathbf{y}_{t-2}
        + \cdots + \mathbf{A}_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t
        """)
        st.markdown("""
        Trong đó:
        - $\\mathbf{y}_t \\in \\mathbb{R}^K$: vector $K$ biến nội sinh tại thời điểm $t$
        - $\\mathbf{A}_i \\in \\mathbb{R}^{K \\times K}$: ma trận hệ số tại lag $i$
        - $\\boldsymbol{\\varepsilon}_t$: vector nhiễu trắng
        """)
    with c2:
        st.code("""
Kiến trúc VAR(p):
┌─────────────────────────┐
│ Input: y_{t-1},...,y_{t-p}│
│ (K biến × p lag) │
└─────────┬───────────────┘
          │ Linear mapping
          
┌─────────────────────────┐
│ A_1 y_{t-1} + ... │
│ + A_p y_{t-p} + c │
└─────────┬───────────────┘
          
┌─────────────────────────┐
│ Output: ŷ_t (K biến) │
└─────────────────────────┘
        """, language="text")
    st.info("**Ưu điểm**: Đơn giản, diễn giải được, phù hợp dữ liệu nhỏ. \n**Nhược điểm**: Giả định tuyến tính, không xử lý ngoại sinh, nhạy cảm với số lượng biến.")

# ── VARX ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("VARX — VAR with Exogenous Variables")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        **Mô tả**: Mở rộng VAR bằng cách thêm **biến ngoại sinh** (exogenous) —
        các biến tác động đến hệ thống nhưng không bị ảnh hưởng ngược lại.

        **Tham số**:
        - **p**: lag nội sinh
        - **q**: lag ngoại sinh
        """)
        st.latex(r"""
        \mathbf{y}_t = \mathbf{c}
        + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{y}_{t-i}
        + \sum_{j=0}^{q} \mathbf{B}_j \mathbf{x}_{t-j}
        + \boldsymbol{\varepsilon}_t
        """)
        st.markdown("""
        Trong đó:
        - $\\mathbf{x}_{t-j} \\in \\mathbb{R}^M$: vector $M$ biến ngoại sinh tại lag $j$
        - $\\mathbf{B}_j \\in \\mathbb{R}^{K \\times M}$: ma trận hệ số ngoại sinh
        """)
    with c2:
        st.code("""
Kiến trúc VARX(p,q):
┌──────────────┐ ┌──────────────┐
│ Endogenous │ │ Exogenous │
│ y_{t-1..p} │ │ x_{t-0..q} │
└──────┬───────┘ └──────┬───────┘
       │ A_i │ B_j
       └────────┬─────────┘
                
        ┌───────────────┐
        │ Linear combine│
        └───────┬───────┘
                
        ┌───────────────┐
        │ Output: ŷ_t │
        └───────────────┘
        """, language="text")
    st.info("**Ưu điểm**: Tận dụng thông tin ngoại sinh (giá thế giới, tỷ giá...). \n**Nhược điểm**: Vẫn tuyến tính, không nắm bắt phi tuyến.")

# ── DeepVAR ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("DeepVAR — Deep Learning VAR")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        **Mô tả**: Thay thế ánh xạ tuyến tính trong VAR bằng **mạng nơ-ron sâu** (LSTM / Transformer).
        Cho phép nắm bắt **quan hệ phi tuyến** và **phụ thuộc dài hạn** trong dữ liệu.

        **Tham số**:
        - **p**: lookback window (độ dài chuỗi đầu vào)
        - **q**: forecast horizon (số bước dự báo)
        - **hidden_size**: kích thước hidden state
        - **num_layers**: số lớp LSTM
        """)
        st.latex(r"""
        \hat{\mathbf{y}}_{t:t+q} = f_\theta\!\left(\mathbf{y}_{t-p:t}\right)
        """)
        st.markdown("""
        Trong đó $f_\\theta$ là mạng LSTM với tham số $\\theta$ được huấn luyện bằng **MSE loss**:
        """)
        st.latex(r"""
        \mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \|\mathbf{y}_t - \hat{\mathbf{y}}_t\|^2
        """)
    with c2:
        st.code("""
Kiến trúc DeepVAR:
┌──────────────────────────┐
│ Input Window │
│ y_{t-p}, ..., y_{t-1} │
│ Shape: (p, K) │
└──────────┬───────────────┘
           │
    ┌─────────────┐
    │ LSTM Layer 1 │ hidden_size
    └──────┬────────┘
    ┌─────────────┐
    │ LSTM Layer 2 │
    └──────┬────────┘
           │
    ┌─────────────┐
    │ FC Linear │
    └──────┬────────┘
           
┌──────────────────────────┐
│ Output: ŷ_{t:t+q} │
│ Shape: (q, K) │
└──────────────────────────┘
        """, language="text")
    st.info("**Ưu điểm**: Nắm bắt phi tuyến, phụ thuộc dài hạn. \n**Nhược điểm**: Cần nhiều dữ liệu, khó diễn giải, hyperparameter phức tạp.")

# ── DeepVARX ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("DeepVARX — Mô hình đề xuất")
    st.markdown("**DeepVARX** tích hợp biến ngoại sinh vào kiến trúc DeepVAR — đây là mô hình đề xuất chính của đề tài.")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
        **Mô tả**: Kết hợp chuỗi nội sinh (lookback p) và chuỗi ngoại sinh (lookback q)
        vào cùng 1 kiến trúc LSTM hai nhánh, sau đó fusion để dự báo.

        **Tham số**:
        - **p**: lookback nội sinh
        - **q**: lookback ngoại sinh
        - **hidden_size**: kích thước hidden state
        - **num_layers**: số lớp LSTM
        - **dropout**: tỷ lệ dropout
        - **horizon**: số bước dự báo
        """)
        st.latex(r"""
        \hat{\mathbf{y}}_{t:t+h}
        = f_\theta\!\left(
            \mathbf{y}_{t-p:t},\;
            \mathbf{x}_{t-q:t}
          \right)
        """)
        st.markdown("""
        **Hàm mất mát** — MSE kết hợp với regularization:
        """)
        st.latex(r"""
        \mathcal{L} = \underbrace{\frac{1}{T}\sum_t\|\mathbf{y}_t - \hat{\mathbf{y}}_t\|^2}_{\text{MSE}}
                    + \lambda \|\theta\|^2
        """)
        st.markdown("""
        **Huấn luyện**: Adam optimizer · Early stopping · Learning rate scheduler
        """)
    with c2:
        st.code("""
Kiến trúc DeepVARX:

 Endogenous Exogenous
 y_{t-p:t} x_{t-q:t}
 Shape:(p,K) Shape:(q,M)
     │ │
┌─────────┐ ┌──────────┐
│LSTM Endo │ │LSTM Exog │
│(hidden_e) │ │(hidden_x) │
└────┬──────┘ └──────┬────┘
     │ │
     └──────────┬────────────┘
                │ Concatenate + FC
          ┌───────────────┐
          │ Fusion Layer │
          └─────┬──────────┘
                │ Dropout
          ┌───────────────┐
          │ FC Output │
          └─────┬──────────┘
                
    ŷ_{t:t+h}, Shape:(h,K)
        """, language="text")
    st.success("**DeepVARX** là mô hình đề xuất — kết hợp sức mạnh của LSTM và thông tin ngoại sinh để dự báo giá cà phê xuất khẩu.")

st.divider()
if st.button("Tiếp theo: Chuẩn bị & Thực nghiệm →", use_container_width=True):
    st.switch_page("pages/5_Chuan_bi_thuc_nghiem.py")

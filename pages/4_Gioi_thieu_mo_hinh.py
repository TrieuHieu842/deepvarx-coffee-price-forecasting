import streamlit as st

st.set_page_config(page_title="Giới thiệu mô hình", page_icon="📖", layout="wide")
st.title("Giới thiệu mô hình")

tab_var, tab_varx, tab_lstm, tab_deepvarx = st.tabs(["VAR", "VARX", "LSTM", "DeepVARX"])

with tab_var:
    st.header("Vector Autoregression (VAR)")
    st.markdown("Mô hình hóa mối quan hệ động giữa nhiều chuỗi thời gian nội sinh, "
                "mỗi biến được hồi quy theo giá trị trễ của chính nó và của các biến còn lại.")
    st.latex(r"Y_t = C + \sum_{i=1}^{p}\Phi_i Y_{t-i} + \varepsilon_t")
    st.markdown("""
    - $Y_t$: vector các biến nội sinh tại thời điểm $t$ (ví dụ: giá Robusta, Arabica)
    - $\\Phi_i$: ma trận hệ số tại độ trễ $i$
    - $p$: bậc trễ của mô hình (xác định qua AIC/BIC hoặc PACF)
    - **Hạn chế**: số tham số tăng theo $O(k^2 p)$ với $k$ là số biến — dễ quá khớp khi $k, p$ lớn
    """)

with tab_varx:
    st.header("Vector Autoregression with Exogenous variables (VARX)")
    st.markdown("Mở rộng VAR bằng cách thêm các biến ngoại sinh (không được mô hình hóa động lực "
                "ngược lại), ví dụ: lượng mưa, tỷ giá, giá dầu.")
    st.latex(r"Y_t = C + \sum_{i=1}^{p}\Phi_i Y_{t-i} + \sum_{j=0}^{q}B_j X_{t-j} + \varepsilon_t")
    st.markdown("""
    - $X_t$: vector biến ngoại sinh
    - $B_j$: ma trận hệ số tác động của biến ngoại sinh tại độ trễ $j$
    - **VARX-L** (Nicholson, Matteson & Bien, 2017): chính quy hóa VARX bằng Group Lasso theo cấu
      trúc độ trễ, giải quyết bùng nổ tham số khi có đa cộng tuyến cao giữa các biến ngoại sinh
    """)

with tab_lstm:
    st.header("Long Short-Term Memory (LSTM)")
    st.markdown("Mạng nơ-ron hồi quy có cấu trúc cổng, giải quyết vấn đề vanishing/exploding "
                "gradient của RNN truyền thống, cho phép nắm bắt phụ thuộc phi tuyến dài hạn.")
    st.latex(r"""
    \begin{aligned}
    f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}\\
    i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}\\
    o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) \quad \text{(output gate)}\\
    \tilde{C}_t &= \tanh(W_C[h_{t-1}, x_t] + b_C)\\
    C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t\\
    h_t &= o_t \odot \tanh(C_t)
    \end{aligned}
    """)
    st.markdown("**Hạn chế**: hộp đen, khó diễn giải; cần nhiều dữ liệu để tránh quá khớp.")

with tab_deepvarx:
    st.header("DeepVARX — Mô hình lai đề xuất")
    st.markdown("Kết hợp nhánh tuyến tính VARX-L (minh bạch, chính quy hóa) và nhánh phi tuyến LSTM "
                "(học phần dư, kích hoạt có điều kiện qua kiểm định BDS), tích hợp qua trust gate "
                "suy giảm theo horizon dự báo.")

    st.subheader("1. Nhánh tuyến tính (VARX-L)")
    st.latex(r"""
    \hat{\Phi}, \hat{B} = \arg\min_{\Phi,B}\ \frac{1}{2T}\sum_{t=1}^{T}
    \left\lVert Y_t - C - \sum_{i=1}^{p}\Phi_i Y_{t-i} - \sum_{j=0}^{q}B_j X_{t-j}\right\rVert_2^2
    + \lambda_1\big[\mathcal{P}_Y(\Phi)+\mathcal{P}_X(B)\big] + \lambda_2 \mathcal{P}_{SS}(\Phi)
    """)
    st.latex(r"\mathcal{P}_Y(\Phi)=\sqrt{k}\sum_{i=1}^{p}\lVert\Phi_i\rVert_F,\quad "
             r"\mathcal{P}_{SS}(\Phi)=\max(0,\ \rho(\mathbf{\Phi})-1+\epsilon)")

    st.subheader("2. Nhánh phi tuyến (LSTM có điều kiện)")
    st.latex(r"e_t = Y_t - \hat{Y}_t^{VARX}, \qquad \hat{e}_t = \text{LSTM}(e_{t-1},\dots,e_{t-k}; X_t)")
    st.caption("LSTM chỉ được kích hoạt nếu kiểm định BDS bác bỏ giả thuyết phần dư i.i.d. "
               "(Brock, Dechert, Scheinkman & LeBaron, 1996).")

    st.subheader("3. Trust gate suy giảm theo horizon")
    st.latex(r"g_h = \sigma(\alpha - \beta h), \qquad "
             r"\hat{Y}_{t+h} = \hat{Y}_{t+h}^{VARX} + g_h \odot \hat{e}_{t+h}")

    st.subheader("4. Quy trình huấn luyện 2 giai đoạn")
    st.markdown("""
    1. **Giai đoạn 1**: ước lượng $\\Phi, B$ bằng Proximal Gradient (ISTA) đến hội tụ
    2. **Kiểm định BDS** trên phần dư để quyết định có kích hoạt Giai đoạn 2 không
    3. **Giai đoạn 2**: huấn luyện LSTM + trust gate trên phần dư, cố định (hoặc fine-tune nhẹ) nhánh tuyến tính
    """)

"""Hồ sơ học thuật (Academic Profile) và quy tắc cấp phát dữ liệu cho 6 mô hình."""
from __future__ import annotations

ORDER = ["VAR", "VARX", "VARX-L", "LSTM", "DeepVAR", "DeepVARX"]

PROFILES = {
    "VAR": {
        "title": "VAR (Vector Autoregression)", "family": "linear",
        "latex": [r"Y_t = c + \Phi_1 Y_{t-1} + \dots + \Phi_p Y_{t-p} + \epsilon_t"],
        "use": "Mô hình cơ sở (Baseline) đánh giá động lực nội sinh của giá.",
        "pros": ["Đơn giản, dễ diễn giải", "Lý thuyết thống kê hoàn thiện"],
        "cons": ["Phân rã nhanh khi số biến tăng (Curse of Dimensionality)", "Chỉ nắm bắt quan hệ tuyến tính",
                 "Không dùng biến ngoại sinh"],
        "impl": "OLS trên chuỗi giá đã sai phân bậc d, chạy RIÊNG TỪNG TỈNH (Local); dự báo 1 bước rồi quy đổi về VNĐ/kg.",
    },
    "VARX": {
        "title": "VARX (VAR with Exogenous Variables)", "family": "linear",
        "latex": [r"Y_t = c + \sum_{i=1}^{p} \Phi_i Y_{t-i} + \sum_{j=0}^{q} B_j X_{t-j} + \epsilon_t"],
        "use": "Đánh giá tác động tuyến tính của biến ngoại sinh đã được Granger xác nhận.",
        "pros": ["Khai thác thông tin vĩ mô, thời tiết (X)"],
        "cons": ["Nhạy cảm với đa cộng tuyến", "Dễ overfit nếu chọn sai biến X"],
        "impl": "Chạy riêng từng tỉnh. Dùng j = 1..q (không dùng X_t) để dự báo 1 bước không cần X tương lai. Nếu Johansen cho r > 0 (VECM), thêm số hạng hiệu chỉnh sai số ECT = β′Z_t (β từ Johansen, ước lượng trên TRAIN của từng tỉnh).",
    },
    "VARX-L": {
        "title": "VARX-L (VARX với Lasso Regularization)", "family": "linear",
        "latex": [r"\min_{c,\Phi,B}\; \sum_t \lVert Y_t - \hat Y_t \rVert^2 \;+\; \lambda \sum \lvert \Phi \rvert \;+\; \lambda \sum \lvert B \rvert"],
        "use": "Tự động chọn lọc đặc trưng (Feature Selection) trong không gian nhiều biến.",
        "pros": ["Ép trọng số các biến nhiễu về 0", "Giải quyết tốt đa cộng tuyến"],
        "cons": ["Phải chọn hệ số phạt λ", "Hệ số bị co (thiên lệch) nên khó đọc như OLS"],
        "impl": "Lasso (sklearn) trên đặc trưng đã chuẩn hoá, chạy RIÊNG TỪNG TỈNH (Local); λ = hệ số phạt bên dưới.",
    },
    "LSTM": {
        "title": "LSTM (Long Short-Term Memory)", "family": "deep",
        "latex": [r"f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)",
                  r"i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i),\quad o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)"],
        "use": "Cơ chế cổng (Forget, Input, Output) kiểm soát luồng thông tin theo thời gian.",
        "pros": ["Nắm bắt phụ thuộc dài hạn", "Không cần dữ liệu phải dừng hoá"],
        "cons": ["Hộp đen (Black-box), khó giải thích biến nào đóng góp nhiều nhất"],
        "impl": "Global: một mạng huấn luyện 1 lần trên toàn panel, nhận chuỗi (Y + X) trượt theo thời gian + embedding tỉnh. Không có Attention.",
    },
    "DeepVAR": {
        "title": "DeepVAR", "family": "deep",
        "latex": [r"Y_t = f_\theta\!\left(Y_{t-1}, \dots, Y_{t-p}\right) + \epsilon_t"],
        "use": "Tham số hoá trực tiếp phương trình VAR bằng mạng nơ-ron (LSTM/RNN), loại bỏ ràng buộc tuyến tính. "
               "So sánh trực tiếp với VAR để thấy tác dụng của phi tuyến trên cùng chuỗi Y.",
        "pros": ["Bắt được động lực phi tuyến của chính Y"],
        "cons": ["Không dùng biến ngoại sinh", "Hộp đen"],
        "impl": "Global: LSTM chỉ nhận chuỗi giá Y (Standard Scaler theo tỉnh, d = 0) + embedding tỉnh.",
    },
    "DeepVARX": {
        "title": "DeepVARX (Mô hình đề xuất)", "family": "deep",
        "latex": [r"Y_t = f_\theta\!\left(Y_{t-1:t-p},\, X_{t-1:t-p}\right) + \epsilon_t",
                  r"\alpha_{t,v}=\mathrm{softmax}_v\!\left(W_a x_t + b_a\right),\qquad c=\sum_{\tau}\beta_\tau h_\tau"],
        "use": "Tích hợp đa biến động (Y và X) qua LSTM sâu, kết hợp Cơ chế Chú ý (Variable + Temporal Attention) "
               "để có trọng số thay đổi theo thời gian.",
        "pros": ["Xử lý dữ liệu thô phức tạp, bắt được phi tuyến",
                 "Attention giúp giải thích biến nào được “chú ý” khi nào (giảm tính hộp đen)"],
        "cons": ["Cần nhiều dữ liệu và siêu tham số", "Trọng số Attention là gợi ý diễn giải, không phải bằng chứng nhân quả"],
        "impl": "Global (1 lần trên toàn panel) + embedding tỉnh. α: gate theo biến (mỗi bước thời gian); β: attention theo thời gian trên trạng thái ẩn LSTM.",
    },
}


def allocation(name: str, base: list[str], ext: list[str]) -> dict:
    """Dữ liệu cấp phát: gói (A/B/C), tập biến X và phạm vi huấn luyện (Local theo tỉnh / Global toàn panel)."""
    table = {
        "VAR": ("Gói A — Tập Cơ Sở, chỉ Y", "A", [], "local"),
        "VARX": ("Gói A — Tập Cơ Sở, Y + X", "A", list(base), "local"),
        "VARX-L": ("Gói B — Tập Mở Rộng sai phân, Y + X", "B", list(ext), "local"),
        "LSTM": ("Gói C — Tập Mở Rộng, Y + X (scaled, mã tỉnh)", "C", list(ext), "global"),
        "DeepVAR": ("Gói C — chỉ Y (scaled, mã tỉnh)", "C", [], "global"),
        "DeepVARX": ("Gói C — Tập Mở Rộng, Y + X (scaled, mã tỉnh)", "C", list(ext), "global"),
    }
    label, pkg, feats, scope = table[name]
    return {"label": label, "pkg": pkg, "feats": feats, "scope": scope}

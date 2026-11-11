# Hệ thống mô hình Vector Autoregression (VAR) dự báo giá cà phê Robusta Việt Nam

Thư mục này đóng gói độc lập toàn bộ mã nguồn, dữ liệu đầu vào, bảng kết quả và biểu đồ trực quan cho các phiên bản mô hình **VAR (Vector Autoregression)** đối chứng với **Naive (Random Walk)** trên chuỗi dừng tỷ suất sinh lợi `log-return`.

## 1. Cấu trúc thư mục
- `var_coffee_models.py`: Mô-đun xử lý chuỗi dừng (`log-return`), kiến trúc OLS đa biến `CoffeeVAR`, lựa chọn bậc trễ tối ưu AIC/BIC, kiểm định `Diebold-Mariano (HAC Newey-West L=5 + HLN)`.
- `run_var_experiments.py`: Chương trình thực nghiệm chính (đánh giá các bậc trễ $p \in \{1, 2, 3, 5, 7\}$, xuất toàn bộ CSV và 2 biểu đồ trực quan).
- `data_daily_panel_clean_v2.csv`: Tập dữ liệu panel ngày (2013–2026) cho 3 tỉnh Đắk Lắk, Lâm Đồng, Gia Lai.
- `var_model_comparison_results.csv`: Bảng kết quả tổng hợp (`Naive`, `VAR(1)`, `VAR(2)`, `VAR(3)`, `VAR(5)`, `VAR(7)`).
- `var_province_results.csv`: Sai số và kiểm định DM chi tiết theo từng tỉnh.
- `figures_var/`:
  - `fig1_var_forecast.png`: Biểu đồ đường dự báo VAR(5) vs Naive vs Thực tế trên 3 tỉnh.
  - `fig2_var_lag_comparison.png`: Biểu đồ so sánh RMSE và Directional Accuracy giữa các bậc trễ $p$.

## 2. Cách chạy thực nghiệm
```powershell
cd DeepVARX\modules\models\var
python run_var_experiments.py
```

## 3. Tóm tắt kết quả trên tập Test (542 ngày: `2024-10-03` -> `2026-10-30`)
| Mô hình | Bậc trễ $p$ | Số biến | RMSE (VNĐ/kg) | MAE (VNĐ/kg) | MAPE (%) | $R^2_{\text{price}}$ | $R^2_{\text{ret}}$ | DA (%) | DM-stat vs Naive | DM $p_{\text{greater}}$ vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive (Random Walk)** | 1 | 3 | **1,922.89** | **1,201.63** | **1.119%** | 0.9830 | -0.0013 | N/A | Benchmark | Benchmark |
| **VAR(1)** | 1 | 3 | 1,932.63 | 1,205.18 | 1.122% | 0.9828 | -0.0125 | **52.24%** | -1.165 | 0.8776 |
| **VAR(2)** | 2 | 3 | 1,931.98 | 1,208.84 | 1.125% | 0.9828 | -0.0108 | 49.97% | -0.826 | 0.7954 |
| **VAR(3)** | 3 | 3 | 1,927.51 | 1,209.73 | 1.127% | 0.9829 | -0.0073 | 48.89% | -0.384 | 0.6493 |
| **VAR(5)** | 5 | 3 | 1,925.23 | 1,209.91 | 1.127% | 0.9829 | -0.0055 | 49.16% | -0.209 | 0.5827 |
| **VAR(7)** | 7 | 3 | 1,931.34 | 1,218.20 | 1.134% | 0.9828 | -0.0118 | 48.69% | -0.677 | 0.7505 |

## 4. Phát hiện cốt lõi từ thực nghiệm VAR
1. **VAR thuần túy không thể thắng Random Walk:** Không có phiên bản VAR nào có $p < 0.05$ trong kiểm định DM. Tất cả các bậc trễ đều có RMSE $\approx 1,925 - 1,932$ VNĐ, nhỉnh hơn một chút so với Naive ($1,922.89$ VNĐ). Tỷ lệ đoán đúng hướng dao động quanh 48% - 52% (ngang ngửa tung đồng xu ngẫu nhiên).
2. **Nguyên nhân toán học:** Chuỗi giá cà phê sau khi lấy tỷ suất sinh lợi là chuỗi tựa Martingale; thông tin tự hồi quy từ chính nó không đủ tín hiệu để giải thích biến động tương lai.
3. **Ý nghĩa học thuật:** Kết quả này khẳng định sự cần thiết bắt buộc phải bổ sung các biến ngoại sinh thị trường vĩ mô (chuyển sang mô hình **VARX**) và phi tuyến tính (**DeepVAR / LSTM**).


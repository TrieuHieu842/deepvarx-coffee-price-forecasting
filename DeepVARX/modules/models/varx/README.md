# Hệ thống mô hình Vector Autoregression with Exogenous Variables (VARX & VARX-L)

Thư mục này đóng gói độc lập toàn bộ mã nguồn, dữ liệu đầu vào, bảng kết quả và biểu đồ trực quan cho các phiên bản mô hình **VARX (Vector Autoregression with Exogenous Variables)** và biến thể có hình phạt co cụm **VARX-L (Ridge / Lasso)** đối chứng với **Naive (Random Walk)** và **VAR Baseline**.

## 1. Cấu trúc thư mục
- `varx_coffee_models.py`: Mô-đun xử lý chuỗi dừng (`log-return`), kiến trúc `CoffeeVARX` (hỗ trợ OLS, Ridge, Lasso), tính toán Permutation Feature Importance và kiểm định `Diebold-Mariano (HAC Newey-West L=5 + HLN)`.
- `run_varx_experiments.py`: Chương trình thực nghiệm chính (đánh giá `Naive`, `VAR Baseline`, `VARX 1`, `VARX 2`, `VARX-L 2 (Ridge)`, `VARX-L 2 (Lasso)`, `VARX 3`, xuất toàn bộ CSV và 3 biểu đồ trực quan).
- `data_daily_panel_clean_v2.csv`: Tập dữ liệu panel ngày (2013–2026) cho 3 tỉnh Đắk Lắk, Lâm Đồng, Gia Lai.
- `varx_model_comparison_results.csv`: Bảng kết quả tổng hợp tất cả các phiên bản.
- `varx_province_results.csv`: Sai số và kiểm định DM chi tiết theo từng tỉnh (`dak_lak`, `lam_dong`, `gia_lai`).
- `varx_feature_importance.csv`: Bảng xếp hạng tầm quan trọng của 8 biến ngoại sinh trong `VARX 2`.
- `figures_varx/`:
  - `fig1_varx_forecast.png`: Biểu đồ đường dự báo VARX 2 vs VARX-L vs Naive vs Thực tế trên 3 tỉnh.
  - `fig2_varx_comparison.png`: Biểu đồ so sánh RMSE và Directional Accuracy giữa các phiên bản VARX.
  - `fig3_varx_feature_importance.png`: Tỷ trọng đóng góp Permutation Feature Importance của các biến ngoại sinh.

## 2. Cách chạy thực nghiệm
```powershell
cd DeepVARX\modules\models\varx
python run_varx_experiments.py
```

## 3. Tóm tắt kết quả trên tập Test (542 ngày: `2024-10-03` -> `2026-10-30`)
| Mô hình | Mô tả | Số biến | Penalty | RMSE (VNĐ/kg) | MAE (VNĐ/kg) | MAPE (%) | $R^2_{\text{ret}}$ | DA (%) | DM-stat vs Naive | DM $p$ vs Naive |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive (Random Walk)** | Chuẩn đối chứng $P_{t-1}$ | 3 | none | 1,922.89 | 1,201.63 | 1.119% | -0.0013 | 0.00% | Benchmark | Benchmark |
| **VAR Baseline** | Thuần nội sinh ($p=5$, không có X) | 3 | none | 1,925.23 | 1,209.91 | 1.127% | -0.0055 | 49.16% | -0.209 | 0.5827 (Thua) |
| **VARX 1** | Giá + Sàn London & Tỷ giá | 5 | none | 1,694.13 | 1,078.79 | 0.998% | 0.2194 | 76.62% | +2.408 | 0.0082 (Thắng) |
| **VARX 2** | Giá + Kinh tế + Thời tiết (11 biến OLS) | 11 | none | 1,685.73 | 1,075.36 | 0.995% | 0.2281 | **76.69%** | +2.507 | 0.0062 (Thắng) |
| **VARX-L 2 (Ridge)** 🏆 | 11 biến cốt lõi có phạt Ridge ($\alpha=0.1$) | 11 | ridge | **1,660.28** | **1,047.25** | **0.970%** | **0.2546** | 75.15% | **+3.171** | **0.0008 (Vượt trội)** |
| **VARX-L 2 (Lasso)** | 11 biến cốt lõi có phạt Lasso ($\alpha=10^{-4}$) | 11 | lasso | 1,853.73 | 1,152.00 | 1.072% | 0.0703 | 67.45% | +5.047 | 0.0000 |
| **VARX 3** | Toàn bộ 20 biến (Full Exogenous OLS) | 20 | none | 1,692.17 | 1,078.79 | 0.998% | 0.2226 | 76.89% | +2.443 | 0.0074 |

## 4. Phát hiện cốt lõi từ thực nghiệm VARX
1. **Bổ sung biến ngoại sinh giúp mô hình tuyến tính lật ngược tình thế:**
   - Trong khi `VAR Baseline` thua Naive (RMSE 1,925 vs 1,922), việc bổ sung các biến ngoại sinh giúp `VARX 2` hạ mạnh RMSE xuống **1,685.73 VNĐ** và tỷ lệ đúng hướng nhảy vọt lên **76.69%** ($p = 0.0062$).
2. **Hiệu quả vượt bậc của hình phạt Ridge (`VARX-L 2`):**
   - Với hình phạt co cụm Ridge ($\alpha = 0.1$), **RMSE đạt mức thấp nhất 1,660.28 VNĐ/kg** (giảm 262 VNĐ so với Naive, giảm 25 VNĐ so với OLS thuần). Kiểm định DM đạt $p = 0.0008$, chứng minh tính ổn định cao của ma trận hiệp phương sai khi có phạt L2.
3. **Biến ngoại sinh chi phối mạnh nhất:**
   - Kết quả Permutation Feature Importance chỉ ra **`ret_london` (sàn ICE London) nắm giữ tới 97.85% tỷ trọng đóng góp** vào việc hạ RMSE của mô hình VARX.
4. **Hiện tượng bão hòa số chiều tại `VARX 3`:**
   - Nạp toàn bộ 20 biến khiến RMSE tăng từ 1,685 lên 1,692 VNĐ (tương tự như ở LSTM và DeepVAR), khẳng định bộ **11 biến cốt lõi** là tối ưu nhất.


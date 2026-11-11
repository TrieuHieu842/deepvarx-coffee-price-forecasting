# Hệ thống mô hình DeepVAR xác suất đa biến (Low-Rank Gaussian) dự báo giá cà phê Robusta Việt Nam

Thư mục `D:\TLCN\deepvar` đóng gói độc lập toàn bộ mã nguồn, dữ liệu, kết quả thực nghiệm (điểm + xác suất) và biểu đồ trực quan cho 3 phiên bản **DeepVAR** đối chứng với **Naive xác suất**, **DeepVAR-Level Raw**, và biến thể **Diagonal DeepVAR**.

## 1. Kiến trúc & Cơ sở lý thuyết
Mô hình [`CoffeeDeepVAR`](deepvar_coffee_models.py) kết hợp:
1. **Salinas et al. (NeurIPS 2019 — DeepVAR / Low-Rank Gaussian Process)**: Tham số hóa phân phối chuẩn đa biến đồng thời $\mathcal{N}(\mu_t, \Sigma_t)$ với hiệp phương sai hạng thấp + đường chéo $\Sigma_t = \operatorname{diag}(d_t) + V_t V_t^\top$ ($K=3$ tỉnh: Đắk Lắk, Lâm Đồng, Gia Lai; `rank = 2`), nắm bắt tương quan không gian liên tỉnh (`~0.88–0.96`).
2. **Kiến trúc Dual-Encoder + $\beta$-NLL (Seitzer et al., ICLR 2022)**: Tách nhánh Mean (`LSTM_mu`: 1 lớp, 64 units) và nhánh Covariance (`LSTM_cov`: 1 lớp, 32 units), tối ưu $\beta\text{-NLL}$ ($\beta = 0.5$) và Early Stopping theo **Validation CRPS** để tránh xung đột gradient và hiện tượng co sập phương sai sớm.
3. **Chuỗi dừng & Neo trên Naive (Beck et al., 2025; Kadir et al., 2026)**: Dự báo tỷ suất sinh lợi $r_t = \ln(P_t/P_{t-1})$ (`lag >= 1`), hiệu chỉnh quy mô phương sai $\tau$ trên Validation, và sinh 1000 mẫu Monte Carlo để tái tạo giá $\hat{P}_t = P_{t-1}\exp(r_t)$ kèm khoảng tin cậy 80% / 95%.

## 2. Cấu trúc thư mục
- `deepvar_coffee_models.py`: Mô-đun xử lý dữ liệu, kiến trúc `CoffeeDeepVAR`, đối chứng `evaluate_raw_level_deepvar`, tính CRPS/CRPS-Sum/NLL/PICP, kiểm định Diebold–Mariano (MSE & CRPS) và Permutation Importance.
- `run_deepvar_experiments.py`: Chương trình thực nghiệm chính (5 seeds: `42, 123, 456, 789, 2024`).
- `data_daily_panel_clean_v2.csv`: Tập dữ liệu panel ngày (2013–2026).
- `benchmark_model_comparison_results.csv`: Bảng đối chứng VAR / VARX / VARX-L / DeepVARX.
- `deepvar_model_comparison_results.csv`: Bảng tổng hợp hiệu suất điểm & xác suất.
- `deepvar_province_results.csv`: Kết quả chi tiết theo từng tỉnh (`dak_lak`, `lam_dong`, `gia_lai`).
- `deepvar_correlation_comparison.csv`: Tương quan liên tỉnh dự báo bởi $\Sigma_t$ vs thực nghiệm Test.
- `deepvar_feature_importance.csv`: Permutation Importance trên cả Mean RMSE và Joint CRPS.
- `figures_deepvar/`:
  - `fig1_deepvar_probabilistic_intervals.png`
  - `fig2_deepvar_versions_and_calibration.png`
  - `fig3_covariance_heatmap_and_importance.png`

## 3. Chạy lại thực nghiệm
```powershell
cd D:\TLCN\deepvar
python run_deepvar_experiments.py
```

## 4. Tóm tắt kết quả trên tập Test (542 ngày: `2024-10-03` → `2026-10-30`, 5 seeds)
| Mô hình | Số biến | Rank | RMSE (VND) | MAE (VND) | DA (%) | NLL_cal | CRPS (ret) | CRPS-Sum | Cov95 (%) | DM_p vs Naive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Naive (RW Prob)** | 1 | 3 | 1922.89 | 1201.63 | 0.00% | -9.096 | 0.00865 | 0.02540 | 90.2% | Benchmark |
| **DeepVAR-Level (Raw)** | 3 | 2 | 15138.10 | 13396.47 | - | - | - | - | - | 1.0000 |
| **DeepVAR 1 (Giá QK)** | 3 | 2 | 1908.82 ± 1.7 | 1196.91 ± 1.5 | 51.64% | -9.297 | 0.00857 | 0.02520 | 92.0% | 0.0925 |
| **DeepVAR 2 (TT+KT)** | 11 | 2 | **1603.82 ± 11.4** | **1031.35 ± 9.9** | 75.89% | **-9.885** | **0.00712** | **0.02084** | 89.2% | **0.0000** |
| **DeepVAR 2 (Diag)** | 11 | 0 | 1615.63 ± 14.8 | 1044.35 ± 11.4 | **76.25%** | -8.721 | 0.00737 | 0.02102 | 97.2% | 0.0000 |
| **DeepVAR 3 (Full NS)** | 20 | 2 | 1662.56 ± 11.8 | 1052.87 ± 13.2 | 74.74% | -9.388 | 0.00743 | 0.02179 | 87.6% | 0.0000 |

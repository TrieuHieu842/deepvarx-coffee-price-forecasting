# BÁO CÁO TỔNG HỢP KIỂM ĐỊNH THỐNG KÊ DIEBOLD–MARIANO

## 1. Thông số thực nghiệm
- **Các mô hình so sánh:** DeepVAR 1, DeepVAR 2, DeepVAR 3, DeepVARX 1, DeepVARX 2, DeepVARX 3, LSTM 1, LSTM 2, LSTM 3, Naive (Random Walk), VAR(1), VAR(2), VAR(5), VARX 1, VARX 2, VARX 3, VARX-L 2 (Ridge)
- **Các tỉnh / khu vực:** dak_lak, gia_lai, lam_dong
- **Bước dự báo (Horizons):** [1]
- **Hàm mất mát:** mse, mae, crps
- **Mức ý nghĩa thống kê (Alpha):** 0.05
- **Tổng số cặp kiểm định:** 861
- **Tổng số quan sát chi tiết:** 466,662

## 2. Bảng kết quả kiểm định tổng hợp (Summary Results)

| province | horizon | model_a | model_b | loss_function | n_observations | hln_stat | hln_p_value | holm_p_value | is_statistically_significant | better_model_by_loss | better_model_statistical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +3.087 | 0.0021 | 0.2717 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +2.978 | 0.0030 | 0.3702 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +1.136 | 0.2564 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +2.578 | 0.0102 | 0.9295 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.300 | 0.0010 | 0.1384 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | +1.344 | 0.1795 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +2.912 | 0.0037 | 0.4386 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +2.656 | 0.0081 | 0.8311 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | -1.426 | 0.1543 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(1) | mse | 542 | +1.185 | 0.2364 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(2) | mse | 542 | +0.290 | 0.7723 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(5) | mse | 542 | +1.196 | 0.2322 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +2.052 | 0.0406 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +2.141 | 0.0327 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.108 | 0.0355 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +2.847 | 0.0046 | 0.5017 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | -2.609 | 0.0093 | 0.8828 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -2.650 | 0.0083 | 0.8367 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | -2.408 | 0.0164 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | -0.731 | 0.4654 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | -3.055 | 0.0024 | 0.3002 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +1.229 | 0.2198 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | -1.844 | 0.0657 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | -3.092 | 0.0021 | 0.2698 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(1) | mse | 542 | -3.054 | 0.0024 | 0.3002 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(2) | mse | 542 | -3.145 | 0.0018 | 0.2277 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(5) | mse | 542 | -3.047 | 0.0024 | 0.3026 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 1 | mse | 542 | -0.534 | 0.5937 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 2 | mse | 542 | -0.452 | 0.6516 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 3 | mse | 542 | -0.536 | 0.5922 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | -0.332 | 0.7397 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -2.484 | 0.0133 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -0.887 | 0.3756 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +1.171 | 0.2422 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -2.937 | 0.0035 | 0.4114 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | +1.896 | 0.0584 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | +0.137 | 0.8914 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -2.983 | 0.0030 | 0.3664 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -2.940 | 0.0034 | 0.4102 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -3.022 | 0.0026 | 0.3256 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -2.907 | 0.0038 | 0.4386 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -0.061 | 0.9515 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 2 | mse | 542 | +0.045 | 0.9641 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -0.034 | 0.9730 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | +0.520 | 0.6034 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +2.193 | 0.0287 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +2.883 | 0.0041 | 0.4592 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | -0.902 | 0.3676 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +2.639 | 0.0086 | 0.8524 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.314 | 0.0211 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | -1.161 | 0.2460 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(1) | mse | 542 | -1.086 | 0.2782 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -1.068 | 0.2858 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(5) | mse | 542 | -0.923 | 0.3565 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +1.962 | 0.0503 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +2.052 | 0.0406 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +2.011 | 0.0448 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +2.640 | 0.0085 | 0.8524 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +1.890 | 0.0593 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -2.524 | 0.0119 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | +2.612 | 0.0092 | 0.8828 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +1.130 | 0.2591 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -2.585 | 0.0100 | 0.9192 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -2.544 | 0.0112 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -2.598 | 0.0096 | 0.8946 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -2.511 | 0.0123 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 1 | mse | 542 | +0.219 | 0.8267 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +0.365 | 0.7154 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 3 | mse | 542 | +0.272 | 0.7858 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.354 | 0.1762 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.263 | 0.0012 | 0.1537 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | +1.324 | 0.1859 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -0.819 | 0.4130 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.302 | 0.0010 | 0.1384 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.269 | 0.0011 | 0.1515 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -3.350 | 0.0009 | 0.1176 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -3.276 | 0.0011 | 0.1489 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -0.359 | 0.7197 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -0.258 | 0.7964 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -0.348 | 0.7283 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | +0.032 | 0.9745 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 2 | mse | 542 | +2.887 | 0.0040 | 0.4592 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 3 | mse | 542 | +2.618 | 0.0091 | 0.8828 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | -1.419 | 0.1564 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(1) | mse | 542 | -0.748 | 0.4546 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(2) | mse | 542 | -0.890 | 0.3736 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(5) | mse | 542 | +0.530 | 0.5960 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.011 | 0.0448 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 2 | mse | 542 | +2.101 | 0.0361 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.067 | 0.0392 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +2.809 | 0.0051 | 0.5456 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | LSTM 3 | mse | 542 | -2.715 | 0.0068 | 0.7177 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -2.914 | 0.0037 | 0.4386 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(1) | mse | 542 | -2.888 | 0.0040 | 0.4592 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(2) | mse | 542 | -2.966 | 0.0032 | 0.3812 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(5) | mse | 542 | -2.909 | 0.0038 | 0.4386 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 1 | mse | 542 | -1.039 | 0.2992 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 2 | mse | 542 | -0.978 | 0.3285 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 3 | mse | 542 | -1.073 | 0.2837 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | -1.015 | 0.3108 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.660 | 0.0080 | 0.8276 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(1) | mse | 542 | -2.626 | 0.0089 | 0.8702 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(2) | mse | 542 | -2.694 | 0.0073 | 0.7575 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(5) | mse | 542 | -2.617 | 0.0091 | 0.8828 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 1 | mse | 542 | -0.117 | 0.9065 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 2 | mse | 542 | +0.008 | 0.9939 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 3 | mse | 542 | -0.089 | 0.9291 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.588 | 0.5570 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | +1.476 | 0.1404 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | +0.540 | 0.5894 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | +1.257 | 0.2093 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +2.056 | 0.0402 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +2.144 | 0.0325 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +2.111 | 0.0352 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +2.848 | 0.0046 | 0.5017 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(2) | mse | 542 | -0.234 | 0.8147 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(5) | mse | 542 | +1.060 | 0.2894 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 1 | mse | 542 | +2.035 | 0.0423 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 2 | mse | 542 | +2.124 | 0.0341 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 3 | mse | 542 | +2.090 | 0.0371 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +2.824 | 0.0049 | 0.5263 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VAR(5) | mse | 542 | +1.288 | 0.1984 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 1 | mse | 542 | +2.052 | 0.0406 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 2 | mse | 542 | +2.144 | 0.0325 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 3 | mse | 542 | +2.110 | 0.0353 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +2.864 | 0.0043 | 0.4823 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 1 | mse | 542 | +2.032 | 0.0427 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 2 | mse | 542 | +2.125 | 0.0340 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 3 | mse | 542 | +2.089 | 0.0372 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +2.834 | 0.0048 | 0.5154 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 2 | mse | 542 | +1.391 | 0.1648 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 3 | mse | 542 | +0.353 | 0.7243 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +0.679 | 0.4974 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX 3 | mse | 542 | -2.205 | 0.0279 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | +0.518 | 0.6046 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +0.694 | 0.4877 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +2.883 | 0.0041 | 0.5103 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +2.738 | 0.0064 | 0.7477 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | -0.312 | 0.7554 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +2.082 | 0.0378 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +3.177 | 0.0016 | 0.2092 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | -0.228 | 0.8199 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +2.651 | 0.0083 | 0.8489 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +2.090 | 0.0371 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +1.446 | 0.1487 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(1) | mae | 542 | +0.218 | 0.8277 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(2) | mae | 542 | -1.224 | 0.2214 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(5) | mae | 542 | -0.652 | 0.5144 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +1.723 | 0.0855 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +1.778 | 0.0760 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +1.694 | 0.0908 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +2.671 | 0.0078 | 0.8276 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | -0.728 | 0.4669 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -2.864 | 0.0043 | 0.5324 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | -2.520 | 0.0120 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | -0.024 | 0.9811 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -2.956 | 0.0033 | 0.4169 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | -1.242 | 0.2148 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | -3.348 | 0.0009 | 0.1185 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | -2.853 | 0.0045 | 0.5394 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(1) | mae | 542 | -2.865 | 0.0043 | 0.5324 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -2.930 | 0.0035 | 0.4446 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -2.884 | 0.0041 | 0.5103 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 1 | mae | 542 | -1.865 | 0.0627 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 2 | mae | 542 | -1.871 | 0.0619 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 3 | mae | 542 | -2.032 | 0.0427 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | -0.636 | 0.5249 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -2.720 | 0.0067 | 0.7560 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -2.117 | 0.0347 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | +0.381 | 0.7034 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -2.813 | 0.0051 | 0.6058 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | -0.570 | 0.5686 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -2.938 | 0.0034 | 0.4381 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -2.709 | 0.0070 | 0.7662 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -2.723 | 0.0067 | 0.7560 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -2.785 | 0.0055 | 0.6541 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -2.735 | 0.0064 | 0.7477 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -1.514 | 0.1305 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 2 | mae | 542 | -1.504 | 0.1333 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -1.668 | 0.0958 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | -0.295 | 0.7680 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.083 | 0.0377 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.154 | 0.0017 | 0.2215 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | +0.105 | 0.9163 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +2.652 | 0.0082 | 0.8489 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.099 | 0.0363 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +0.479 | 0.6325 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(1) | mae | 542 | +0.371 | 0.7108 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(2) | mae | 542 | -0.420 | 0.6747 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -0.207 | 0.8358 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +1.748 | 0.0810 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +1.802 | 0.0722 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +1.719 | 0.0863 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.675 | 0.0077 | 0.8276 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +2.225 | 0.0265 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -2.133 | 0.0334 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | +1.287 | 0.1985 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | -0.138 | 0.8901 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.057 | 0.0402 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.067 | 0.0392 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.136 | 0.0331 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.095 | 0.0366 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 1 | mae | 542 | -0.483 | 0.6294 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 2 | mae | 542 | -0.412 | 0.6805 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -0.593 | 0.5532 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +1.513 | 0.1308 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -3.254 | 0.0012 | 0.1633 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | -0.918 | 0.3592 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -2.437 | 0.0151 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.145 | 0.0018 | 0.2261 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.156 | 0.0017 | 0.2215 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.218 | 0.0014 | 0.1834 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.163 | 0.0016 | 0.2177 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -1.710 | 0.0878 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -1.686 | 0.0924 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -1.831 | 0.0676 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | -0.591 | 0.5547 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 2 | mae | 542 | +2.725 | 0.0066 | 0.7560 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 3 | mae | 542 | +2.150 | 0.0320 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +0.413 | 0.6800 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(1) | mae | 542 | +0.292 | 0.7705 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(2) | mae | 542 | -0.590 | 0.5557 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(5) | mae | 542 | -0.289 | 0.7727 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 1 | mae | 542 | +1.765 | 0.0781 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 2 | mae | 542 | +1.822 | 0.0691 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 3 | mae | 542 | +1.737 | 0.0829 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +2.736 | 0.0064 | 0.7477 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | LSTM 3 | mae | 542 | -1.861 | 0.0633 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -2.620 | 0.0090 | 0.8947 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(1) | mae | 542 | -2.634 | 0.0087 | 0.8685 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(2) | mae | 542 | -2.706 | 0.0070 | 0.7664 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VAR(5) | mae | 542 | -2.663 | 0.0080 | 0.8378 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 1 | mae | 542 | -1.389 | 0.1653 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 2 | mae | 542 | -1.358 | 0.1749 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 3 | mae | 542 | -1.531 | 0.1264 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | +0.169 | 0.8658 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -2.063 | 0.0396 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.076 | 0.0384 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.150 | 0.0320 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.109 | 0.0354 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 1 | mae | 542 | -0.393 | 0.6945 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 2 | mae | 542 | -0.325 | 0.7452 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 3 | mae | 542 | -0.506 | 0.6129 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +1.597 | 0.1109 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | -0.289 | 0.7723 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -1.513 | 0.1310 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -0.881 | 0.3788 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +1.699 | 0.0898 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +1.754 | 0.0800 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +1.671 | 0.0953 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +2.642 | 0.0085 | 0.8556 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(2) | mae | 542 | -1.665 | 0.0965 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(5) | mae | 542 | -0.824 | 0.4105 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 1 | mae | 542 | +1.710 | 0.0879 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 2 | mae | 542 | +1.765 | 0.0782 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 3 | mae | 542 | +1.681 | 0.0933 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +2.655 | 0.0082 | 0.8489 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VAR(5) | mae | 542 | +0.486 | 0.6271 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 1 | mae | 542 | +1.779 | 0.0759 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 2 | mae | 542 | +1.834 | 0.0673 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 3 | mae | 542 | +1.751 | 0.0805 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +2.722 | 0.0067 | 0.7560 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 1 | mae | 542 | +1.750 | 0.0807 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 2 | mae | 542 | +1.804 | 0.0718 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 3 | mae | 542 | +1.721 | 0.0858 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +2.677 | 0.0077 | 0.8276 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 2 | mae | 542 | +0.562 | 0.5740 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 3 | mae | 542 | -0.484 | 0.6285 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.476 | 0.0136 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX 3 | mae | 542 | -1.498 | 0.1348 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | +2.604 | 0.0095 | 0.9284 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +2.856 | 0.0045 | 0.5389 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | +3.457 | 0.0006 | 0.0077 | True | DeepVAR 2 | DeepVAR 2 |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +3.145 | 0.0018 | 0.0193 | True | DeepVAR 3 | DeepVAR 3 |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +0.912 | 0.3624 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +2.740 | 0.0064 | 0.0572 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +4.013 | < 0.0001 | 0.0010 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | -2.032 | 0.0427 | 0.2561 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | -3.295 | 0.0010 | 0.0126 | True | DeepVAR 2 | DeepVAR 2 |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | -2.143 | 0.0326 | 0.2279 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | +0.236 | 0.8139 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -2.974 | 0.0031 | 0.0307 | True | DeepVAR 3 | DeepVAR 3 |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | -0.785 | 0.4326 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +1.437 | 0.1512 | 0.6050 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +2.600 | 0.0096 | 0.0765 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +3.844 | 0.0001 | 0.0019 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +2.001 | 0.0459 | 0.2561 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +3.753 | 0.0002 | 0.0254 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +3.397 | 0.0007 | 0.0819 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +0.131 | 0.8955 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +2.847 | 0.0046 | 0.3331 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.333 | 0.0009 | 0.0945 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | +0.297 | 0.7668 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +3.397 | 0.0007 | 0.0819 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +2.991 | 0.0029 | 0.2619 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | +0.054 | 0.9567 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(1) | mse | 542 | -1.109 | 0.2679 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(2) | mse | 542 | -1.000 | 0.3179 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(5) | mse | 542 | -0.727 | 0.4673 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +2.819 | 0.0050 | 0.3412 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +2.879 | 0.0041 | 0.3261 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.788 | 0.0055 | 0.3456 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +3.408 | 0.0007 | 0.0801 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | -3.239 | 0.0013 | 0.1275 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -3.755 | 0.0002 | 0.0254 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | -3.508 | 0.0005 | 0.0602 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | -2.493 | 0.0130 | 0.7772 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | -3.782 | 0.0002 | 0.0232 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +0.554 | 0.5798 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | -3.009 | 0.0027 | 0.2576 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | -3.722 | 0.0002 | 0.0284 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(1) | mse | 542 | -3.760 | 0.0002 | 0.0251 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(2) | mse | 542 | -3.877 | 0.0001 | 0.0160 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(5) | mse | 542 | -3.980 | < 0.0001 | 0.0107 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VARX 1 | mse | 542 | -1.736 | 0.0831 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 2 | mse | 542 | -1.745 | 0.0815 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 3 | mse | 542 | -1.924 | 0.0548 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | -1.672 | 0.0952 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -3.382 | 0.0008 | 0.0838 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -1.021 | 0.3077 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +0.688 | 0.4920 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -3.421 | 0.0007 | 0.0798 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | +2.146 | 0.0323 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | +0.092 | 0.9270 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -3.372 | 0.0008 | 0.0856 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -3.417 | 0.0007 | 0.0798 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -3.516 | 0.0005 | 0.0594 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -3.591 | 0.0004 | 0.0460 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -0.445 | 0.6562 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 2 | mse | 542 | -0.343 | 0.7316 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -0.554 | 0.5798 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | +0.148 | 0.8826 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +2.833 | 0.0048 | 0.3394 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +3.314 | 0.0010 | 0.1001 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | +0.236 | 0.8139 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +3.398 | 0.0007 | 0.0819 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.986 | 0.0030 | 0.2628 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | -0.106 | 0.9154 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(1) | mse | 542 | -1.546 | 0.1226 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -1.478 | 0.1400 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(5) | mse | 542 | -1.107 | 0.2686 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +2.818 | 0.0050 | 0.3412 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +2.877 | 0.0042 | 0.3261 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +2.787 | 0.0055 | 0.3456 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +3.399 | 0.0007 | 0.0819 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +1.613 | 0.1074 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -2.853 | 0.0045 | 0.3331 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | +3.052 | 0.0024 | 0.2311 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +1.212 | 0.2261 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -2.827 | 0.0049 | 0.3412 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -2.881 | 0.0041 | 0.3261 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -2.954 | 0.0033 | 0.2749 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -3.007 | 0.0028 | 0.2576 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 1 | mse | 542 | +0.033 | 0.9738 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +0.216 | 0.8288 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 3 | mse | 542 | -0.057 | 0.9544 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.023 | 0.3066 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.340 | 0.0009 | 0.0930 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | +2.106 | 0.0357 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -0.562 | 0.5742 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.307 | 0.0010 | 0.1017 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.343 | 0.0009 | 0.0930 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -3.440 | 0.0006 | 0.0759 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -3.517 | 0.0005 | 0.0594 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -0.757 | 0.4496 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -0.677 | 0.4987 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -0.894 | 0.3715 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -0.289 | 0.7725 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 2 | mse | 542 | +3.419 | 0.0007 | 0.0798 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 3 | mse | 542 | +3.007 | 0.0028 | 0.2576 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | -0.270 | 0.7875 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(1) | mse | 542 | -1.363 | 0.1734 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(2) | mse | 542 | -1.284 | 0.1995 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(5) | mse | 542 | -1.017 | 0.3097 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.824 | 0.0049 | 0.3412 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 2 | mse | 542 | +2.887 | 0.0040 | 0.3238 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.796 | 0.0054 | 0.3443 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +3.424 | 0.0007 | 0.0798 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | LSTM 3 | mse | 542 | -2.995 | 0.0029 | 0.2614 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -3.368 | 0.0008 | 0.0859 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(1) | mse | 542 | -3.415 | 0.0007 | 0.0798 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(2) | mse | 542 | -3.518 | 0.0005 | 0.0594 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(5) | mse | 542 | -3.609 | 0.0003 | 0.0433 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VARX 1 | mse | 542 | -1.780 | 0.0757 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 2 | mse | 542 | -1.793 | 0.0736 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 3 | mse | 542 | -1.952 | 0.0514 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | -1.564 | 0.1183 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.967 | 0.0031 | 0.2748 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(1) | mse | 542 | -3.025 | 0.0026 | 0.2478 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(2) | mse | 542 | -3.112 | 0.0020 | 0.1920 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(5) | mse | 542 | -3.181 | 0.0016 | 0.1535 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 1 | mse | 542 | -0.542 | 0.5882 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 2 | mse | 542 | -0.443 | 0.6580 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 3 | mse | 542 | -0.680 | 0.4966 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.082 | 0.9346 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | -1.103 | 0.2706 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | -0.956 | 0.3393 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | -0.685 | 0.4935 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +2.800 | 0.0053 | 0.3443 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +2.859 | 0.0044 | 0.3310 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +2.769 | 0.0058 | 0.3550 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +3.383 | 0.0008 | 0.0838 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(2) | mse | 542 | -0.044 | 0.9646 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(5) | mse | 542 | +0.189 | 0.8498 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 1 | mse | 542 | +2.853 | 0.0045 | 0.3331 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 2 | mse | 542 | +2.910 | 0.0038 | 0.3088 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 3 | mse | 542 | +2.824 | 0.0049 | 0.3412 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +3.415 | 0.0007 | 0.0798 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VAR(5) | mse | 542 | +0.386 | 0.6995 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 1 | mse | 542 | +2.904 | 0.0038 | 0.3104 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 2 | mse | 542 | +2.968 | 0.0031 | 0.2748 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 3 | mse | 542 | +2.880 | 0.0041 | 0.3261 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +3.496 | 0.0005 | 0.0624 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 1 | mse | 542 | +2.960 | 0.0032 | 0.2748 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 2 | mse | 542 | +3.028 | 0.0026 | 0.2478 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 3 | mse | 542 | +2.937 | 0.0035 | 0.2867 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +3.574 | 0.0004 | 0.0487 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| gia_lai | 1 | VARX 1 | VARX 2 | mse | 542 | +1.005 | 0.3154 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 3 | mse | 542 | -0.465 | 0.6420 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +1.126 | 0.2606 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX 3 | mse | 542 | -2.969 | 0.0031 | 0.2748 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.012 | 0.3119 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +1.453 | 0.1467 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +3.823 | 0.0001 | 0.0194 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +3.341 | 0.0009 | 0.1045 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | +0.575 | 0.5657 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +2.488 | 0.0132 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +3.174 | 0.0016 | 0.1718 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | +0.241 | 0.8097 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +3.415 | 0.0007 | 0.0844 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +2.656 | 0.0081 | 0.7659 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +2.069 | 0.0390 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(1) | mae | 542 | -0.094 | 0.9249 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(2) | mae | 542 | -0.877 | 0.3810 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(5) | mae | 542 | -1.521 | 0.1289 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +2.476 | 0.0136 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +2.520 | 0.0120 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +2.418 | 0.0159 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +3.162 | 0.0017 | 0.1769 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | -2.893 | 0.0040 | 0.3930 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -3.778 | 0.0002 | 0.0228 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | -3.680 | 0.0003 | 0.0328 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | -2.458 | 0.0143 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -3.879 | 0.0001 | 0.0158 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | -1.811 | 0.0707 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | -4.073 | < 0.0001 | 0.0073 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | -3.778 | 0.0002 | 0.0228 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(1) | mae | 542 | -3.783 | 0.0002 | 0.0225 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -3.828 | 0.0001 | 0.0192 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -3.904 | 0.0001 | 0.0144 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | VARX 1 | mae | 542 | -2.348 | 0.0192 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 2 | mae | 542 | -2.407 | 0.0164 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 3 | mae | 542 | -2.542 | 0.0113 | 0.9817 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | -2.020 | 0.0439 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -3.295 | 0.0010 | 0.1205 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -2.165 | 0.0309 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | -0.867 | 0.3864 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -3.386 | 0.0008 | 0.0927 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | +0.325 | 0.7454 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -2.432 | 0.0153 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -3.295 | 0.0010 | 0.1205 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -3.303 | 0.0010 | 0.1183 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -3.350 | 0.0009 | 0.1022 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -3.421 | 0.0007 | 0.0838 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -1.299 | 0.1944 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 2 | mae | 542 | -1.307 | 0.1919 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -1.477 | 0.1401 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | -0.715 | 0.4749 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.448 | 0.0147 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.122 | 0.0019 | 0.1951 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | -0.130 | 0.8966 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +3.372 | 0.0008 | 0.0962 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.617 | 0.0091 | 0.8397 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +0.258 | 0.7968 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(1) | mae | 542 | -0.545 | 0.5861 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(2) | mae | 542 | -1.333 | 0.1831 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -2.138 | 0.0330 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +2.443 | 0.0149 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +2.486 | 0.0132 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +2.384 | 0.0175 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +3.116 | 0.0019 | 0.1968 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +1.240 | 0.2154 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -2.508 | 0.0124 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | +2.403 | 0.0166 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | +0.444 | 0.6570 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.443 | 0.0149 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.456 | 0.0144 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.507 | 0.0125 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.577 | 0.0102 | 0.9096 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 1 | mae | 542 | +0.085 | 0.9324 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 2 | mae | 542 | +0.172 | 0.8638 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -0.067 | 0.9463 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +1.356 | 0.1757 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -3.198 | 0.0015 | 0.1626 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | +1.167 | 0.2439 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -0.931 | 0.3522 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.125 | 0.0019 | 0.1951 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.131 | 0.0018 | 0.1929 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.179 | 0.0016 | 0.1701 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.258 | 0.0012 | 0.1347 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -0.816 | 0.4150 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -0.778 | 0.4372 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -0.951 | 0.3422 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | +0.033 | 0.9735 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 2 | mae | 542 | +3.466 | 0.0006 | 0.0719 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 3 | mae | 542 | +2.684 | 0.0075 | 0.7195 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +0.277 | 0.7819 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(1) | mae | 542 | -0.272 | 0.7857 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(2) | mae | 542 | -0.832 | 0.4055 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(5) | mae | 542 | -1.383 | 0.1674 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 1 | mae | 542 | +2.489 | 0.0131 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 2 | mae | 542 | +2.535 | 0.0115 | 0.9857 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 3 | mae | 542 | +2.431 | 0.0154 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +3.189 | 0.0015 | 0.1663 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | LSTM 3 | mae | 542 | -2.748 | 0.0062 | 0.6017 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -3.368 | 0.0008 | 0.0966 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(1) | mae | 542 | -3.373 | 0.0008 | 0.0962 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(2) | mae | 542 | -3.420 | 0.0007 | 0.0838 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VAR(5) | mae | 542 | -3.502 | 0.0005 | 0.0634 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 1 | mae | 542 | -1.641 | 0.1013 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 2 | mae | 542 | -1.656 | 0.0982 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 3 | mae | 542 | -1.823 | 0.0689 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | -0.942 | 0.3468 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -2.611 | 0.0093 | 0.8402 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.625 | 0.0089 | 0.8292 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.677 | 0.0077 | 0.7269 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.753 | 0.0061 | 0.5986 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 1 | mae | 542 | -0.218 | 0.8273 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 2 | mae | 542 | -0.160 | 0.8731 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 3 | mae | 542 | -0.368 | 0.7133 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +0.873 | 0.3830 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | -0.695 | 0.4873 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -1.384 | 0.1671 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -1.885 | 0.0599 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +2.430 | 0.0154 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +2.474 | 0.0136 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +2.373 | 0.0180 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +3.112 | 0.0020 | 0.1968 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(2) | mae | 542 | -1.420 | 0.1563 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(5) | mae | 542 | -2.052 | 0.0406 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 1 | mae | 542 | +2.447 | 0.0147 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 2 | mae | 542 | +2.490 | 0.0131 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 3 | mae | 542 | +2.389 | 0.0173 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +3.114 | 0.0019 | 0.1968 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VAR(5) | mae | 542 | -1.461 | 0.1446 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 1 | mae | 542 | +2.493 | 0.0130 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 2 | mae | 542 | +2.537 | 0.0115 | 0.9857 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 3 | mae | 542 | +2.436 | 0.0152 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +3.162 | 0.0017 | 0.1769 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 1 | mae | 542 | +2.569 | 0.0105 | 0.9216 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 2 | mae | 542 | +2.613 | 0.0092 | 0.8402 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 3 | mae | 542 | +2.511 | 0.0123 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +3.238 | 0.0013 | 0.1433 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 2 | mae | 542 | +0.459 | 0.6465 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 3 | mae | 542 | -0.691 | 0.4900 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +1.485 | 0.1381 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX 3 | mae | 542 | -1.439 | 0.1508 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | +1.558 | 0.1198 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +1.823 | 0.0689 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | +4.463 | < 0.0001 | 0.0001 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +3.591 | 0.0004 | 0.0032 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +0.145 | 0.8851 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +2.933 | 0.0035 | 0.0210 | True | DeepVARX 2 | DeepVARX 2 |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +3.713 | 0.0002 | 0.0025 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | -3.797 | 0.0002 | 0.0021 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | -4.596 | < 0.0001 | < 0.0001 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | -3.064 | 0.0023 | 0.0184 | True | DeepVAR 2 | DeepVAR 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | -2.429 | 0.0155 | 0.0773 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -3.658 | 0.0003 | 0.0028 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | -0.831 | 0.4061 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +0.397 | 0.6915 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +2.984 | 0.0030 | 0.0208 | True | DeepVARX 2 | DeepVARX 2 |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +3.792 | 0.0002 | 0.0021 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +1.171 | 0.2422 | 0.9689 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +3.121 | 0.0019 | 0.2032 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +3.140 | 0.0018 | 0.1960 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | -0.375 | 0.7081 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +2.987 | 0.0029 | 0.2802 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.356 | 0.0008 | 0.1098 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | +0.171 | 0.8642 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +2.990 | 0.0029 | 0.2802 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +2.722 | 0.0067 | 0.5688 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | +0.558 | 0.5768 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(1) | mse | 542 | -1.401 | 0.1617 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(2) | mse | 542 | -1.047 | 0.2955 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(5) | mse | 542 | -0.633 | 0.5272 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +2.127 | 0.0339 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +2.264 | 0.0240 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.194 | 0.0286 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +3.178 | 0.0016 | 0.1820 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | -1.895 | 0.0586 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -3.156 | 0.0017 | 0.1906 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | -2.075 | 0.0384 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | -1.047 | 0.2956 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | -3.165 | 0.0016 | 0.1864 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +0.806 | 0.4207 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | -2.981 | 0.0030 | 0.2826 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | -3.131 | 0.0018 | 0.1989 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(1) | mse | 542 | -3.224 | 0.0013 | 0.1621 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(2) | mse | 542 | -3.261 | 0.0012 | 0.1474 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(5) | mse | 542 | -3.268 | 0.0012 | 0.1474 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 1 | mse | 542 | -1.845 | 0.0656 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 2 | mse | 542 | -1.703 | 0.0891 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 3 | mse | 542 | -1.799 | 0.0727 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | -0.920 | 0.3580 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -3.175 | 0.0016 | 0.1824 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -1.110 | 0.2675 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +0.352 | 0.7250 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -3.181 | 0.0015 | 0.1815 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | +1.457 | 0.1456 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | -1.333 | 0.1832 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -3.151 | 0.0017 | 0.1922 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -3.245 | 0.0012 | 0.1522 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -3.268 | 0.0012 | 0.1474 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -3.261 | 0.0012 | 0.1474 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -1.429 | 0.1535 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 2 | mse | 542 | -1.256 | 0.2096 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -1.363 | 0.1733 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | -0.274 | 0.7843 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +3.014 | 0.0027 | 0.2668 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +3.384 | 0.0008 | 0.1012 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | +0.576 | 0.5646 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +3.025 | 0.0026 | 0.2630 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.755 | 0.0061 | 0.5319 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | +0.440 | 0.6600 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(1) | mse | 542 | -2.178 | 0.0298 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -1.373 | 0.1704 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(5) | mse | 542 | -0.694 | 0.4880 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +2.145 | 0.0324 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +2.282 | 0.0228 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +2.214 | 0.0272 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +3.188 | 0.0015 | 0.1803 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +1.242 | 0.2146 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -3.020 | 0.0026 | 0.2644 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | +1.835 | 0.0670 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | -0.195 | 0.8452 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -2.997 | 0.0029 | 0.2771 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -3.089 | 0.0021 | 0.2189 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -3.104 | 0.0020 | 0.2129 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -3.090 | 0.0021 | 0.2189 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 1 | mse | 542 | -1.289 | 0.1979 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 2 | mse | 542 | -1.070 | 0.2852 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 3 | mse | 542 | -1.197 | 0.2320 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +0.260 | 0.7947 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.399 | 0.0007 | 0.0965 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | +1.161 | 0.2463 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -1.337 | 0.1816 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.370 | 0.0008 | 0.1054 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.441 | 0.0006 | 0.0838 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -3.469 | 0.0006 | 0.0767 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -3.466 | 0.0006 | 0.0770 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -1.582 | 0.1143 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -1.408 | 0.1596 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -1.512 | 0.1312 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -0.465 | 0.6420 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 2 | mse | 542 | +3.033 | 0.0025 | 0.2589 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 3 | mse | 542 | +2.756 | 0.0060 | 0.5319 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | -0.053 | 0.9574 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(1) | mse | 542 | -1.653 | 0.0988 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(2) | mse | 542 | -1.300 | 0.1943 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(5) | mse | 542 | -0.815 | 0.4156 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.131 | 0.0335 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 2 | mse | 542 | +2.271 | 0.0235 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.202 | 0.0281 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +3.201 | 0.0014 | 0.1738 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | LSTM 3 | mse | 542 | -2.563 | 0.0106 | 0.8626 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -3.000 | 0.0028 | 0.2771 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(1) | mse | 542 | -3.096 | 0.0021 | 0.2167 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(2) | mse | 542 | -3.133 | 0.0018 | 0.1989 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(5) | mse | 542 | -3.143 | 0.0018 | 0.1958 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 1 | mse | 542 | -1.927 | 0.0546 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 2 | mse | 542 | -1.794 | 0.0734 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 3 | mse | 542 | -1.879 | 0.0607 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | -1.057 | 0.2911 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.730 | 0.0065 | 0.5632 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(1) | mse | 542 | -2.848 | 0.0046 | 0.4105 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(2) | mse | 542 | -2.864 | 0.0043 | 0.4036 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(5) | mse | 542 | -2.856 | 0.0044 | 0.4049 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 1 | mse | 542 | -1.165 | 0.2444 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 2 | mse | 542 | -0.957 | 0.3389 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 3 | mse | 542 | -1.080 | 0.2807 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.349 | 0.7270 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | -1.406 | 0.1604 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | -1.064 | 0.2879 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | -0.656 | 0.5120 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +2.127 | 0.0339 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +2.265 | 0.0239 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +2.195 | 0.0286 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +3.184 | 0.0015 | 0.1815 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(2) | mse | 542 | +0.526 | 0.5992 | 1.0000 | False | VAR(2) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(5) | mse | 542 | +0.916 | 0.3599 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 1 | mse | 542 | +2.273 | 0.0234 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 2 | mse | 542 | +2.405 | 0.0165 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 3 | mse | 542 | +2.340 | 0.0197 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +3.271 | 0.0011 | 0.1473 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VAR(5) | mse | 542 | +1.152 | 0.2498 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 1 | mse | 542 | +2.251 | 0.0248 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 2 | mse | 542 | +2.386 | 0.0174 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 3 | mse | 542 | +2.320 | 0.0207 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +3.264 | 0.0012 | 0.1474 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 1 | mse | 542 | +2.221 | 0.0268 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 2 | mse | 542 | +2.359 | 0.0187 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 3 | mse | 542 | +2.292 | 0.0223 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +3.250 | 0.0012 | 0.1509 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 2 | mse | 542 | +2.634 | 0.0087 | 0.7287 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 3 | mse | 542 | +1.271 | 0.2043 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +2.862 | 0.0044 | 0.4036 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX 3 | mse | 542 | -2.593 | 0.0098 | 0.8018 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | +2.618 | 0.0091 | 0.7540 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +2.818 | 0.0050 | 0.4457 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +2.832 | 0.0048 | 0.4995 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +2.933 | 0.0035 | 0.4162 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | +0.253 | 0.8005 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +2.682 | 0.0076 | 0.7022 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +3.181 | 0.0016 | 0.2021 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | -0.361 | 0.7182 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +2.787 | 0.0055 | 0.5500 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +2.208 | 0.0277 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | -0.167 | 0.8676 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(1) | mae | 542 | -1.086 | 0.2779 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(2) | mae | 542 | -1.053 | 0.2929 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(5) | mae | 542 | -1.077 | 0.2820 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +1.933 | 0.0538 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +2.057 | 0.0402 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +2.019 | 0.0440 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +2.914 | 0.0037 | 0.4269 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | +0.306 | 0.7597 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -2.820 | 0.0050 | 0.5079 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | -0.554 | 0.5795 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | +0.113 | 0.9104 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -2.889 | 0.0040 | 0.4466 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | -0.526 | 0.5993 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | -2.932 | 0.0035 | 0.4162 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | -2.836 | 0.0047 | 0.4974 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(1) | mae | 542 | -2.900 | 0.0039 | 0.4431 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -2.878 | 0.0042 | 0.4571 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -2.893 | 0.0040 | 0.4457 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 1 | mae | 542 | -1.536 | 0.1251 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 2 | mae | 542 | -1.334 | 0.1826 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 3 | mae | 542 | -1.375 | 0.1697 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | -0.078 | 0.9381 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -2.922 | 0.0036 | 0.4240 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -0.767 | 0.4431 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | -0.062 | 0.9507 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -2.995 | 0.0029 | 0.3649 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | -0.573 | 0.5669 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -3.241 | 0.0013 | 0.1715 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -2.937 | 0.0035 | 0.4142 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -3.005 | 0.0028 | 0.3559 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -2.978 | 0.0030 | 0.3786 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -2.984 | 0.0030 | 0.3743 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -1.489 | 0.1370 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 2 | mae | 542 | -1.311 | 0.1906 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -1.355 | 0.1760 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | -0.208 | 0.8356 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.669 | 0.0078 | 0.7210 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.166 | 0.0016 | 0.2106 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | -0.533 | 0.5943 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +2.773 | 0.0057 | 0.5689 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.195 | 0.0286 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | -0.278 | 0.7813 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(1) | mae | 542 | -1.371 | 0.1708 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(2) | mae | 542 | -1.287 | 0.1988 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -1.363 | 0.1734 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +1.916 | 0.0559 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +2.040 | 0.0418 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +2.003 | 0.0457 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.895 | 0.0039 | 0.4457 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +0.626 | 0.5319 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -2.739 | 0.0064 | 0.6120 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | +0.092 | 0.9268 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | -1.867 | 0.0624 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.685 | 0.0075 | 0.7022 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.749 | 0.0062 | 0.6052 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.729 | 0.0065 | 0.6222 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.742 | 0.0063 | 0.6120 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 1 | mae | 542 | -1.297 | 0.1953 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 2 | mae | 542 | -1.071 | 0.2846 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -1.123 | 0.2617 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +0.371 | 0.7109 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -3.242 | 0.0013 | 0.1715 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | -0.471 | 0.6376 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -2.211 | 0.0274 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.187 | 0.0015 | 0.1992 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.241 | 0.0013 | 0.1715 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.209 | 0.0014 | 0.1861 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.223 | 0.0013 | 0.1791 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -1.491 | 0.1364 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -1.294 | 0.1963 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -1.327 | 0.1849 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | -0.169 | 0.8655 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 2 | mae | 542 | +2.849 | 0.0046 | 0.4920 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 3 | mae | 542 | +2.263 | 0.0241 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +0.306 | 0.7595 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(1) | mae | 542 | -0.716 | 0.4743 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(2) | mae | 542 | -0.795 | 0.4269 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(5) | mae | 542 | -0.874 | 0.3826 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 1 | mae | 542 | +1.974 | 0.0489 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 2 | mae | 542 | +2.100 | 0.0362 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 3 | mae | 542 | +2.062 | 0.0397 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +2.973 | 0.0031 | 0.3786 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | LSTM 3 | mae | 542 | -1.992 | 0.0469 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -2.792 | 0.0054 | 0.5473 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(1) | mae | 542 | -2.852 | 0.0045 | 0.4913 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(2) | mae | 542 | -2.828 | 0.0049 | 0.5006 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VAR(5) | mae | 542 | -2.844 | 0.0046 | 0.4949 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 1 | mae | 542 | -1.371 | 0.1709 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 2 | mae | 542 | -1.144 | 0.2532 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 3 | mae | 542 | -1.188 | 0.2352 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | +0.238 | 0.8119 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -2.211 | 0.0275 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.284 | 0.0227 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.273 | 0.0234 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.284 | 0.0227 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 1 | mae | 542 | -0.277 | 0.7820 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 2 | mae | 542 | -0.022 | 0.9826 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 3 | mae | 542 | -0.076 | 0.9394 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +1.829 | 0.0679 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | -1.062 | 0.2889 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -1.034 | 0.3018 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -1.062 | 0.2887 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +1.936 | 0.0534 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +2.060 | 0.0399 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +2.022 | 0.0437 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +2.917 | 0.0037 | 0.4267 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(2) | mae | 542 | -0.393 | 0.6944 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(5) | mae | 542 | -0.341 | 0.7336 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 1 | mae | 542 | +2.006 | 0.0453 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 2 | mae | 542 | +2.128 | 0.0338 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 3 | mae | 542 | +2.091 | 0.0370 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +2.976 | 0.0030 | 0.3786 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VAR(5) | mae | 542 | -0.105 | 0.9165 | 1.0000 | False | VAR(2) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 1 | mae | 542 | +2.002 | 0.0458 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 2 | mae | 542 | +2.122 | 0.0343 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 3 | mae | 542 | +2.085 | 0.0375 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +2.954 | 0.0033 | 0.3965 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 1 | mae | 542 | +2.017 | 0.0441 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 2 | mae | 542 | +2.139 | 0.0329 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 3 | mae | 542 | +2.102 | 0.0360 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +2.970 | 0.0031 | 0.3795 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 2 | mae | 542 | +1.733 | 0.0837 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 3 | mae | 542 | +1.072 | 0.2842 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.843 | 0.0046 | 0.4949 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX 3 | mae | 542 | -0.458 | 0.6471 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | +2.646 | 0.0084 | 0.7551 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +2.650 | 0.0083 | 0.7530 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | +3.387 | 0.0008 | 0.0091 | True | DeepVAR 2 | DeepVAR 2 |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +3.235 | 0.0013 | 0.0116 | True | DeepVAR 3 | DeepVAR 3 |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +0.140 | 0.8885 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +3.203 | 0.0014 | 0.0116 | True | DeepVARX 2 | DeepVARX 2 |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +3.828 | 0.0001 | 0.0020 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | -1.257 | 0.2092 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | -3.487 | 0.0005 | 0.0069 | True | DeepVAR 2 | DeepVAR 2 |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | -0.764 | 0.4454 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | +0.154 | 0.8776 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -3.320 | 0.0010 | 0.0106 | True | DeepVAR 3 | DeepVAR 3 |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | +0.142 | 0.8875 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +0.940 | 0.3478 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +3.282 | 0.0011 | 0.0110 | True | DeepVARX 2 | DeepVARX 2 |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +3.939 | < 0.0001 | 0.0014 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +0.796 | 0.4263 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |

## 3. Tổng hợp các cặp có ưu thế thống kê vượt trội (Statistically Significant Differences)

Có **43** phép kiểm định xác nhận sự khác biệt có ý nghĩa thống kê:

- **dak_lak** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=+3.457, p-value Holm=0.0077).
- **dak_lak** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.145, p-value Holm=0.0193).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.013, p-value Holm=0.0010).
- **dak_lak** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-3.295, p-value Holm=0.0126).
- **dak_lak** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=-2.974, p-value Holm=0.0307).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.844, p-value Holm=0.0019).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=+3.753, p-value Holm=0.0254).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.755, p-value Holm=0.0254).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.782, p-value Holm=0.0232).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.722, p-value Holm=0.0284).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.760, p-value Holm=0.0251).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.877, p-value Holm=0.0160).
- **gia_lai** (h=1, MSE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.980, p-value Holm=0.0107).
- **gia_lai** (h=1, MSE): **DeepVAR 3** vượt trội hơn (HLN stat=-3.591, p-value Holm=0.0460).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-3.609, p-value Holm=0.0433).
- **gia_lai** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.574, p-value Holm=0.0487).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=+3.823, p-value Holm=0.0194).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.778, p-value Holm=0.0228).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.680, p-value Holm=0.0328).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.879, p-value Holm=0.0158).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-4.073, p-value Holm=0.0073).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.778, p-value Holm=0.0228).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.783, p-value Holm=0.0225).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.828, p-value Holm=0.0192).
- **gia_lai** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.904, p-value Holm=0.0144).
- **gia_lai** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=+4.463, p-value Holm=0.0001).
- **gia_lai** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.591, p-value Holm=0.0032).
- **gia_lai** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+2.933, p-value Holm=0.0210).
- **gia_lai** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.713, p-value Holm=0.0025).
- **gia_lai** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-3.797, p-value Holm=0.0021).
- **gia_lai** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-4.596, p-value Holm=0.0001).
- **gia_lai** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-3.064, p-value Holm=0.0184).
- **gia_lai** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=-3.658, p-value Holm=0.0028).
- **gia_lai** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+2.984, p-value Holm=0.0208).
- **gia_lai** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.792, p-value Holm=0.0021).
- **lam_dong** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=+3.387, p-value Holm=0.0091).
- **lam_dong** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.235, p-value Holm=0.0116).
- **lam_dong** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+3.203, p-value Holm=0.0116).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.828, p-value Holm=0.0020).
- **lam_dong** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-3.487, p-value Holm=0.0069).
- **lam_dong** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=-3.320, p-value Holm=0.0106).
- **lam_dong** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+3.282, p-value Holm=0.0110).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.939, p-value Holm=0.0014).

---
*Báo cáo được sinh tự động bởi Statistical Evaluation Module (Diebold-Mariano Suite).*
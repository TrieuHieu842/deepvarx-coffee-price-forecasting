# BÁO CÁO TỔNG HỢP KIỂM ĐỊNH THỐNG KÊ DIEBOLD–MARIANO

## 1. Thông số thực nghiệm
- **Các mô hình so sánh:** DeepVAR 1, DeepVAR 2, DeepVAR 3, DeepVARX 1, DeepVARX 2, DeepVARX 3, LSTM 1, LSTM 2, LSTM 3, Naive (Random Walk), VAR(1), VAR(2), VAR(5), VARX 1, VARX 2, VARX 3, VARX-L 2 (Ridge)
- **Các tỉnh / khu vực:** dak_lak, dak_nong, gia_lai, lam_dong
- **Bước dự báo (Horizons):** [1]
- **Hàm mất mát:** mse, mae, crps
- **Mức ý nghĩa thống kê (Alpha):** 0.05
- **Tổng số cặp kiểm định:** 1148
- **Tổng số quan sát chi tiết:** 622,216

## 2. Bảng kết quả kiểm định tổng hợp (Summary Results)

| province | horizon | model_a | model_b | loss_function | n_observations | hln_stat | hln_p_value | holm_p_value | is_statistically_significant | better_model_by_loss | better_model_statistical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | -0.619 | 0.5364 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +2.548 | 0.0111 | 0.9430 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +0.459 | 0.6462 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +2.668 | 0.0079 | 0.7465 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.317 | 0.0010 | 0.1185 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | -0.238 | 0.8117 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +4.022 | < 0.0001 | 0.0085 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +2.425 | 0.0156 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | -0.615 | 0.5385 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(1) | mse | 542 | +0.814 | 0.4158 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(2) | mse | 542 | -0.036 | 0.9715 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(5) | mse | 542 | +0.280 | 0.7798 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +1.962 | 0.0503 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +3.178 | 0.0016 | 0.1832 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.018 | 0.0441 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +2.746 | 0.0062 | 0.6276 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | +2.784 | 0.0056 | 0.5772 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | +1.118 | 0.2642 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | +2.972 | 0.0031 | 0.3433 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | +3.555 | 0.0004 | 0.0518 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | +0.416 | 0.6778 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +5.290 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | +2.562 | 0.0107 | 0.9270 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | +0.053 | 0.9574 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(1) | mse | 542 | +1.620 | 0.1059 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(2) | mse | 542 | +0.587 | 0.5576 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(5) | mse | 542 | +0.808 | 0.4194 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 1 | mse | 542 | +1.926 | 0.0546 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 2 | mse | 542 | +3.001 | 0.0028 | 0.3203 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 3 | mse | 542 | +1.982 | 0.0479 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | +2.748 | 0.0062 | 0.6276 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -2.475 | 0.0136 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -1.469 | 0.1425 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +2.040 | 0.0418 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -2.711 | 0.0069 | 0.6721 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | -0.994 | 0.3205 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | +0.629 | 0.5297 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -2.800 | 0.0053 | 0.5551 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -2.637 | 0.0086 | 0.7978 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -2.750 | 0.0062 | 0.6276 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -2.660 | 0.0080 | 0.7554 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 1 | mse | 542 | +0.070 | 0.9446 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 2 | mse | 542 | +0.907 | 0.3650 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 3 | mse | 542 | +0.117 | 0.9072 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | +0.901 | 0.3678 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +2.590 | 0.0098 | 0.8764 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +3.232 | 0.0013 | 0.1537 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | -0.419 | 0.6752 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +4.225 | < 0.0001 | 0.0037 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.341 | 0.0196 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | -1.098 | 0.2728 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(1) | mse | 542 | +0.712 | 0.4765 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -0.310 | 0.7569 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(5) | mse | 542 | +0.001 | 0.9995 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +1.857 | 0.0639 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +3.002 | 0.0028 | 0.3203 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +1.910 | 0.0566 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +2.621 | 0.0090 | 0.8197 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +2.638 | 0.0086 | 0.7978 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -2.804 | 0.0052 | 0.5538 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | -0.383 | 0.7022 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +1.364 | 0.1732 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -2.988 | 0.0029 | 0.3284 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -2.717 | 0.0068 | 0.6662 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -2.814 | 0.0051 | 0.5429 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -2.730 | 0.0065 | 0.6472 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 1 | mse | 542 | +0.600 | 0.5485 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +1.855 | 0.0642 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 3 | mse | 542 | +0.666 | 0.5060 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.743 | 0.0818 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.540 | 0.0004 | 0.0544 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | -2.084 | 0.0376 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -1.139 | 0.2550 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.577 | 0.0004 | 0.0480 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.526 | 0.0005 | 0.0568 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -3.605 | 0.0003 | 0.0437 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -3.520 | 0.0005 | 0.0576 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -0.502 | 0.6160 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -0.103 | 0.9178 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -0.485 | 0.6282 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -0.229 | 0.8186 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 2 | mse | 542 | +4.146 | < 0.0001 | 0.0051 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 1 | LSTM 3 | mse | 542 | +2.563 | 0.0107 | 0.9270 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | -0.417 | 0.6765 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(1) | mse | 542 | +0.967 | 0.3342 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(2) | mse | 542 | +0.167 | 0.8674 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(5) | mse | 542 | +0.480 | 0.6317 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.013 | 0.0446 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 2 | mse | 542 | +3.268 | 0.0012 | 0.1372 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.076 | 0.0383 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +2.860 | 0.0044 | 0.4793 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | LSTM 3 | mse | 542 | +0.991 | 0.3220 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -5.317 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(1) | mse | 542 | -5.083 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(2) | mse | 542 | -4.647 | < 0.0001 | 0.0006 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(5) | mse | 542 | -4.537 | < 0.0001 | 0.0009 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VARX 1 | mse | 542 | +0.609 | 0.5425 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 2 | mse | 542 | +1.446 | 0.1486 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 3 | mse | 542 | +0.655 | 0.5130 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | +1.259 | 0.2086 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.572 | 0.0104 | 0.9131 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(1) | mse | 542 | -2.458 | 0.0143 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(2) | mse | 542 | -2.599 | 0.0096 | 0.8654 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(5) | mse | 542 | -2.531 | 0.0117 | 0.9787 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 1 | mse | 542 | -0.088 | 0.9295 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 2 | mse | 542 | +0.782 | 0.4345 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 3 | mse | 542 | -0.042 | 0.9666 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.826 | 0.4092 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | +1.631 | 0.1034 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | +0.587 | 0.5576 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | +0.803 | 0.4220 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +1.927 | 0.0545 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +3.006 | 0.0028 | 0.3185 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +1.984 | 0.0478 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +2.754 | 0.0061 | 0.6267 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(2) | mse | 542 | -1.549 | 0.1219 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(5) | mse | 542 | -0.965 | 0.3349 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 1 | mse | 542 | +1.829 | 0.0680 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 2 | mse | 542 | +3.083 | 0.0022 | 0.2498 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 3 | mse | 542 | +1.892 | 0.0591 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +2.699 | 0.0072 | 0.6876 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VAR(5) | mse | 542 | +0.793 | 0.4279 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 1 | mse | 542 | +2.008 | 0.0451 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 2 | mse | 542 | +3.315 | 0.0010 | 0.1185 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 3 | mse | 542 | +2.073 | 0.0386 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +2.875 | 0.0042 | 0.4613 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 1 | mse | 542 | +1.981 | 0.0481 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 2 | mse | 542 | +3.286 | 0.0011 | 0.1301 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 3 | mse | 542 | +2.044 | 0.0414 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +2.829 | 0.0048 | 0.5223 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 2 | mse | 542 | +0.848 | 0.3966 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 3 | mse | 542 | +0.516 | 0.6060 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +0.762 | 0.4463 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX 3 | mse | 542 | -0.825 | 0.4096 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | -0.271 | 0.7863 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +0.753 | 0.4519 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +1.823 | 0.0689 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +2.980 | 0.0030 | 0.3288 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | +2.676 | 0.0077 | 0.7512 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +2.983 | 0.0030 | 0.3286 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +4.061 | < 0.0001 | 0.0072 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | +0.344 | 0.7307 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +7.230 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +2.580 | 0.0101 | 0.9627 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +1.921 | 0.0552 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(1) | mae | 542 | +1.975 | 0.0488 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(2) | mae | 542 | +1.006 | 0.3147 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VAR(5) | mae | 542 | +1.473 | 0.1413 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +1.863 | 0.0630 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +3.675 | 0.0003 | 0.0324 | True | VARX 2 | VARX 2 |
| dak_lak | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +1.904 | 0.0574 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +3.158 | 0.0017 | 0.1895 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | +2.362 | 0.0185 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -0.599 | 0.5494 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | +2.367 | 0.0183 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | +3.264 | 0.0012 | 0.1355 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -1.197 | 0.2318 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | +7.415 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | +1.987 | 0.0474 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | +1.277 | 0.2023 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(1) | mae | 542 | +0.182 | 0.8555 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -0.964 | 0.3356 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -0.596 | 0.5515 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 1 | mae | 542 | +1.283 | 0.2001 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 2 | mae | 542 | +2.919 | 0.0037 | 0.3920 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX 3 | mae | 542 | +1.321 | 0.1872 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | +2.516 | 0.0122 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -2.535 | 0.0115 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -0.746 | 0.4557 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | +1.314 | 0.1896 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -2.948 | 0.0033 | 0.3599 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | -0.001 | 0.9992 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -1.811 | 0.0706 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -2.337 | 0.0198 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -2.448 | 0.0147 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -2.674 | 0.0077 | 0.7512 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -2.578 | 0.0102 | 0.9627 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -1.683 | 0.0930 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 2 | mae | 542 | +0.662 | 0.5082 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -1.661 | 0.0973 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | +0.405 | 0.6859 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.540 | 0.0113 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.519 | 0.0005 | 0.0564 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | -1.311 | 0.1903 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +7.412 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.146 | 0.0323 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +0.753 | 0.4519 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(1) | mae | 542 | +0.758 | 0.4488 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(2) | mae | 542 | -0.710 | 0.4780 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -0.289 | 0.7728 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +1.425 | 0.1546 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +3.183 | 0.0015 | 0.1758 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +1.465 | 0.1434 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.706 | 0.0070 | 0.7015 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +1.621 | 0.1056 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -2.910 | 0.0038 | 0.3995 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | +0.253 | 0.8001 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | -0.335 | 0.7379 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.338 | 0.0197 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.445 | 0.0148 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.687 | 0.0074 | 0.7360 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.578 | 0.0102 | 0.9627 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 1 | mae | 542 | -1.253 | 0.2109 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 2 | mae | 542 | +1.249 | 0.2123 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -1.210 | 0.2267 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +0.898 | 0.3698 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -4.038 | < 0.0001 | 0.0078 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | -0.841 | 0.4005 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -1.845 | 0.0657 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.241 | 0.0013 | 0.1452 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.395 | 0.0007 | 0.0878 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.603 | 0.0003 | 0.0420 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.529 | 0.0005 | 0.0547 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -2.186 | 0.0292 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -0.627 | 0.5311 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -2.142 | 0.0327 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | -0.776 | 0.4380 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | LSTM 2 | mae | 542 | +6.026 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 1 | LSTM 3 | mae | 542 | +2.548 | 0.0111 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +1.270 | 0.2046 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(1) | mae | 542 | +1.468 | 0.1428 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(2) | mae | 542 | +0.686 | 0.4931 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VAR(5) | mae | 542 | +1.044 | 0.2967 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 1 | mae | 542 | +1.827 | 0.0683 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX 2 | mae | 542 | +3.635 | 0.0003 | 0.0375 | True | VARX 2 | VARX 2 |
| dak_lak | 1 | LSTM 1 | VARX 3 | mae | 542 | +1.871 | 0.0619 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +3.110 | 0.0020 | 0.2203 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | LSTM 3 | mae | 542 | -0.342 | 0.7324 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -7.230 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(1) | mae | 542 | -7.036 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(2) | mae | 542 | -6.663 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VAR(5) | mae | 542 | -6.422 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_lak | 1 | LSTM 2 | VARX 1 | mae | 542 | -0.917 | 0.3593 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 2 | mae | 542 | +0.326 | 0.7449 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX 3 | mae | 542 | -0.873 | 0.3830 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | +0.178 | 0.8591 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -1.962 | 0.0503 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.057 | 0.0402 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.297 | 0.0220 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.205 | 0.0279 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 1 | mae | 542 | -1.341 | 0.1803 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 2 | mae | 542 | +1.447 | 0.1484 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX 3 | mae | 542 | -1.304 | 0.1928 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_lak | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +1.406 | 0.1604 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | +0.042 | 0.9668 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -1.057 | 0.2912 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -0.690 | 0.4907 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +1.259 | 0.2086 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +2.889 | 0.0040 | 0.4220 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +1.297 | 0.1952 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +2.490 | 0.0131 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(2) | mae | 542 | -2.001 | 0.0459 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VAR(5) | mae | 542 | -1.218 | 0.2236 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 1 | mae | 542 | +1.310 | 0.1906 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 2 | mae | 542 | +3.099 | 0.0020 | 0.2265 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX 3 | mae | 542 | +1.353 | 0.1767 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +2.612 | 0.0092 | 0.8872 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VAR(5) | mae | 542 | +1.024 | 0.3065 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 1 | mae | 542 | +1.581 | 0.1145 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 2 | mae | 542 | +3.391 | 0.0007 | 0.0883 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX 3 | mae | 542 | +1.623 | 0.1051 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +2.862 | 0.0044 | 0.4546 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 1 | mae | 542 | +1.495 | 0.1354 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 2 | mae | 542 | +3.288 | 0.0011 | 0.1259 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX 3 | mae | 542 | +1.537 | 0.1249 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +2.765 | 0.0059 | 0.6012 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 2 | mae | 542 | +2.828 | 0.0049 | 0.4997 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX 3 | mae | 542 | +0.518 | 0.6047 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_lak | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +3.708 | 0.0002 | 0.0288 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_lak | 1 | VARX 2 | VARX 3 | mae | 542 | -2.747 | 0.0062 | 0.6285 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | -0.407 | 0.6844 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_lak | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +3.745 | 0.0002 | 0.0251 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | -0.965 | 0.3350 | 0.6038 | False | DeepVAR 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +2.835 | 0.0048 | 0.0476 | True | DeepVAR 3 | DeepVAR 3 |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +1.033 | 0.3019 | 0.6038 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +2.634 | 0.0087 | 0.0608 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +4.274 | < 0.0001 | 0.0003 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | +2.977 | 0.0030 | 0.0365 | True | DeepVAR 3 | DeepVAR 3 |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | +2.160 | 0.0312 | 0.1250 | False | DeepVARX 1 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | +2.877 | 0.0042 | 0.0459 | True | DeepVARX 2 | DeepVARX 2 |
| dak_lak | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | +4.172 | < 0.0001 | 0.0005 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -2.699 | 0.0072 | 0.0573 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | -1.391 | 0.1648 | 0.4943 | False | DeepVAR 3 | None (No significant difference) |
| dak_lak | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +2.411 | 0.0162 | 0.0811 | False | DeepVARX 3 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +2.525 | 0.0118 | 0.0710 | False | DeepVARX 2 | None (No significant difference) |
| dak_lak | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +4.104 | < 0.0001 | 0.0006 | True | DeepVARX 3 | DeepVARX 3 |
| dak_lak | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +2.824 | 0.0049 | 0.0476 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +2.180 | 0.0297 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +0.576 | 0.5649 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +0.124 | 0.9017 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +0.357 | 0.7214 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +1.465 | 0.1435 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | +2.581 | 0.0101 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +2.906 | 0.0038 | 0.4759 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +0.323 | 0.7465 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | +2.150 | 0.0320 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VAR(1) | mse | 542 | +1.465 | 0.1436 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VAR(2) | mse | 542 | +1.555 | 0.1205 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VAR(5) | mse | 542 | +1.771 | 0.0771 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VARX 1 | mse | 542 | -1.024 | 0.3062 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +0.704 | 0.4818 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VARX 3 | mse | 542 | -1.006 | 0.3147 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +0.117 | 0.9066 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | +0.327 | 0.7441 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -1.645 | 0.1005 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | +0.040 | 0.9682 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | +1.166 | 0.2440 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | +0.661 | 0.5090 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +1.774 | 0.0766 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | +0.093 | 0.9258 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | -0.353 | 0.7242 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(1) | mse | 542 | +0.128 | 0.8979 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(2) | mse | 542 | +0.241 | 0.8097 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(5) | mse | 542 | +0.263 | 0.7926 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX 1 | mse | 542 | -1.214 | 0.2253 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX 2 | mse | 542 | +0.397 | 0.6917 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX 3 | mse | 542 | -1.195 | 0.2326 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | -0.116 | 0.9075 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -0.590 | 0.5555 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -0.898 | 0.3696 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +2.482 | 0.0134 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -0.218 | 0.8275 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | +0.228 | 0.8195 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | -1.920 | 0.0554 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -0.360 | 0.7187 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -0.320 | 0.7493 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -0.293 | 0.7699 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -0.292 | 0.7705 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -3.400 | 0.0007 | 0.0932 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VARX 2 | mse | 542 | +0.003 | 0.9972 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -3.433 | 0.0006 | 0.0836 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | -1.391 | 0.1649 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +0.355 | 0.7228 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +1.525 | 0.1279 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | +2.498 | 0.0128 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +3.016 | 0.0027 | 0.3376 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +0.324 | 0.7463 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | +1.984 | 0.0478 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VAR(1) | mse | 542 | +1.603 | 0.1095 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VAR(2) | mse | 542 | +1.783 | 0.0752 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VAR(5) | mse | 542 | +1.957 | 0.0508 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VARX 1 | mse | 542 | -1.070 | 0.2852 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +0.720 | 0.4716 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VARX 3 | mse | 542 | -1.051 | 0.2935 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +0.108 | 0.9143 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +2.805 | 0.0052 | 0.6464 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | +0.134 | 0.8932 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | +0.781 | 0.4354 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +0.183 | 0.8545 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -0.079 | 0.9372 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -0.002 | 0.9984 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VAR(2) | mse | 542 | +0.037 | 0.9709 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VAR(5) | mse | 542 | +0.036 | 0.9710 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VARX 1 | mse | 542 | -2.432 | 0.0153 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +1.176 | 0.2400 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VARX 3 | mse | 542 | -2.408 | 0.0164 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | -0.369 | 0.7120 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -1.128 | 0.2596 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | -0.744 | 0.4570 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -2.732 | 0.0065 | 0.7986 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -1.222 | 0.2224 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -1.238 | 0.2163 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -1.218 | 0.2239 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -1.215 | 0.2249 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -3.741 | 0.0002 | 0.0266 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -1.944 | 0.0524 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -3.742 | 0.0002 | 0.0266 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -2.534 | 0.0116 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | LSTM 2 | mse | 542 | +1.920 | 0.0553 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | LSTM 3 | mse | 542 | -0.025 | 0.9802 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | -1.097 | 0.2730 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(1) | mse | 542 | -0.717 | 0.4735 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(2) | mse | 542 | -0.465 | 0.6418 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(5) | mse | 542 | -0.534 | 0.5936 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX 1 | mse | 542 | -1.398 | 0.1626 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX 2 | mse | 542 | +0.273 | 0.7853 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX 3 | mse | 542 | -1.379 | 0.1684 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | -0.253 | 0.8006 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | LSTM 3 | mse | 542 | -0.457 | 0.6482 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -2.218 | 0.0270 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(1) | mse | 542 | -1.926 | 0.0547 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(2) | mse | 542 | -1.821 | 0.0691 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(5) | mse | 542 | -1.893 | 0.0589 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX 1 | mse | 542 | -1.829 | 0.0679 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX 2 | mse | 542 | -0.279 | 0.7800 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX 3 | mse | 542 | -1.808 | 0.0711 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | -0.703 | 0.4825 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -0.122 | 0.9032 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VAR(1) | mse | 542 | -0.070 | 0.9443 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VAR(2) | mse | 542 | -0.043 | 0.9659 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VAR(5) | mse | 542 | -0.043 | 0.9659 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VARX 1 | mse | 542 | -3.356 | 0.0008 | 0.1075 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VARX 2 | mse | 542 | +0.663 | 0.5073 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VARX 3 | mse | 542 | -3.396 | 0.0007 | 0.0938 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | -0.801 | 0.4234 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | +0.312 | 0.7548 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | +0.439 | 0.6609 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | +0.482 | 0.6298 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | -1.200 | 0.2307 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +0.439 | 0.6610 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | -1.181 | 0.2381 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | -0.089 | 0.9290 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VAR(2) | mse | 542 | +0.731 | 0.4653 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VAR(5) | mse | 542 | +0.695 | 0.4871 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VARX 1 | mse | 542 | -1.298 | 0.1948 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VARX 2 | mse | 542 | +0.394 | 0.6939 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VARX 3 | mse | 542 | -1.281 | 0.2009 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | -0.153 | 0.8783 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VAR(5) | mse | 542 | -0.006 | 0.9954 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX 1 | mse | 542 | -1.328 | 0.1847 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX 2 | mse | 542 | +0.360 | 0.7190 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX 3 | mse | 542 | -1.311 | 0.1905 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | -0.182 | 0.8556 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(5) | VARX 1 | mse | 542 | -1.327 | 0.1849 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | VAR(5) | VARX 2 | mse | 542 | +0.360 | 0.7190 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | VAR(5) | VARX 3 | mse | 542 | -1.310 | 0.1909 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | -0.182 | 0.8558 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | VARX 1 | VARX 2 | mse | 542 | +3.823 | 0.0001 | 0.0197 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | VARX 1 | VARX 3 | mse | 542 | +0.274 | 0.7840 | 1.0000 | False | VARX 3 | None (No significant difference) |
| dak_nong | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +4.781 | < 0.0001 | 0.0003 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_nong | 1 | VARX 2 | VARX 3 | mse | 542 | -3.789 | 0.0002 | 0.0224 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | -1.729 | 0.0844 | 1.0000 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +4.893 | < 0.0001 | 0.0002 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +4.449 | < 0.0001 | 0.0011 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | -2.685 | 0.0075 | 0.3136 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | -0.116 | 0.9076 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | -2.712 | 0.0069 | 0.2964 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | -1.670 | 0.0954 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | +5.852 | < 0.0001 | < 0.0001 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +2.634 | 0.0087 | 0.3407 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | -2.829 | 0.0048 | 0.2190 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +5.305 | < 0.0001 | < 0.0001 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | DeepVAR 1 | VAR(1) | mae | 542 | +5.715 | < 0.0001 | < 0.0001 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | DeepVAR 1 | VAR(2) | mae | 542 | +5.633 | < 0.0001 | < 0.0001 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | DeepVAR 1 | VAR(5) | mae | 542 | +5.829 | < 0.0001 | < 0.0001 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | DeepVAR 1 | VARX 1 | mae | 542 | -4.202 | < 0.0001 | 0.0031 | True | DeepVAR 1 | DeepVAR 1 |
| dak_nong | 1 | DeepVAR 1 | VARX 2 | mae | 542 | -1.822 | 0.0690 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | VARX 3 | mae | 542 | -4.200 | < 0.0001 | 0.0031 | True | DeepVAR 1 | DeepVAR 1 |
| dak_nong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | -2.885 | 0.0041 | 0.1953 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | -3.821 | 0.0001 | 0.0099 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -4.716 | < 0.0001 | 0.0003 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | -3.970 | < 0.0001 | 0.0065 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | -2.979 | 0.0030 | 0.1478 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -1.163 | 0.2452 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | -1.252 | 0.2112 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | -3.845 | 0.0001 | 0.0093 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | +0.570 | 0.5692 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(1) | mae | 542 | +0.069 | 0.9453 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -0.338 | 0.7355 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -0.462 | 0.6439 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX 1 | mae | 542 | -4.877 | < 0.0001 | 0.0002 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | VARX 2 | mae | 542 | -3.202 | 0.0014 | 0.0824 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | VARX 3 | mae | 542 | -4.867 | < 0.0001 | 0.0002 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | -3.931 | < 0.0001 | 0.0072 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | +2.630 | 0.0088 | 0.3407 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | +0.696 | 0.4868 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | +2.666 | 0.0079 | 0.3239 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | +3.807 | 0.0002 | 0.0104 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | +3.843 | 0.0001 | 0.0093 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -2.203 | 0.0280 | 0.9527 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | +3.882 | 0.0001 | 0.0083 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | DeepVAR 3 | VAR(1) | mae | 542 | +4.018 | < 0.0001 | 0.0057 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | DeepVAR 3 | VAR(2) | mae | 542 | +3.935 | < 0.0001 | 0.0071 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | DeepVAR 3 | VAR(5) | mae | 542 | +3.946 | < 0.0001 | 0.0069 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -4.252 | < 0.0001 | 0.0025 | True | DeepVAR 3 | DeepVAR 3 |
| dak_nong | 1 | DeepVAR 3 | VARX 2 | mae | 542 | +3.233 | 0.0013 | 0.0754 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -4.317 | < 0.0001 | 0.0019 | True | DeepVAR 3 | DeepVAR 3 |
| dak_nong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | -0.908 | 0.3645 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | -2.641 | 0.0085 | 0.3407 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | -1.595 | 0.1112 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | +5.305 | < 0.0001 | < 0.0001 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +2.375 | 0.0179 | 0.6437 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | -2.774 | 0.0057 | 0.2518 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +5.394 | < 0.0001 | < 0.0001 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | DeepVARX 1 | VAR(1) | mae | 542 | +5.810 | < 0.0001 | < 0.0001 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | DeepVARX 1 | VAR(2) | mae | 542 | +5.737 | < 0.0001 | < 0.0001 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | DeepVARX 1 | VAR(5) | mae | 542 | +6.096 | < 0.0001 | < 0.0001 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | DeepVARX 1 | VARX 1 | mae | 542 | -4.152 | < 0.0001 | 0.0036 | True | DeepVARX 1 | DeepVARX 1 |
| dak_nong | 1 | DeepVARX 1 | VARX 2 | mae | 542 | -1.767 | 0.0777 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | VARX 3 | mae | 542 | -4.149 | < 0.0001 | 0.0036 | True | DeepVARX 1 | DeepVARX 1 |
| dak_nong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | -2.835 | 0.0048 | 0.2190 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +2.258 | 0.0243 | 0.8517 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | +3.991 | < 0.0001 | 0.0061 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | +4.008 | < 0.0001 | 0.0059 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | -1.622 | 0.1053 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | +4.050 | < 0.0001 | 0.0052 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | DeepVARX 2 | VAR(1) | mae | 542 | +4.210 | < 0.0001 | 0.0030 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | DeepVARX 2 | VAR(2) | mae | 542 | +4.110 | < 0.0001 | 0.0042 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | DeepVARX 2 | VAR(5) | mae | 542 | +4.122 | < 0.0001 | 0.0040 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | DeepVARX 2 | VARX 1 | mae | 542 | -4.177 | < 0.0001 | 0.0033 | True | DeepVARX 2 | DeepVARX 2 |
| dak_nong | 1 | DeepVARX 2 | VARX 2 | mae | 542 | +2.567 | 0.0105 | 0.3898 | False | VARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -4.207 | < 0.0001 | 0.0030 | True | DeepVARX 2 | DeepVARX 2 |
| dak_nong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | -1.273 | 0.2037 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | +3.035 | 0.0025 | 0.1259 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | +3.309 | 0.0010 | 0.0608 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -3.167 | 0.0016 | 0.0880 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | +3.120 | 0.0019 | 0.0991 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(1) | mae | 542 | +3.270 | 0.0011 | 0.0675 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(2) | mae | 542 | +3.188 | 0.0015 | 0.0845 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VAR(5) | mae | 542 | +3.189 | 0.0015 | 0.0845 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -5.052 | < 0.0001 | < 0.0001 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -0.189 | 0.8501 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -5.052 | < 0.0001 | < 0.0001 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | -2.866 | 0.0043 | 0.2028 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | LSTM 2 | mae | 542 | -0.805 | 0.4210 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | LSTM 3 | mae | 542 | -3.858 | 0.0001 | 0.0090 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +1.952 | 0.0514 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(1) | mae | 542 | +2.041 | 0.0417 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(2) | mae | 542 | +1.575 | 0.1159 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VAR(5) | mae | 542 | +1.474 | 0.1410 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX 1 | mae | 542 | -4.991 | < 0.0001 | < 0.0001 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | LSTM 1 | VARX 2 | mae | 542 | -3.163 | 0.0017 | 0.0880 | False | LSTM 1 | None (No significant difference) |
| dak_nong | 1 | LSTM 1 | VARX 3 | mae | 542 | -4.989 | < 0.0001 | < 0.0001 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | -3.929 | < 0.0001 | 0.0072 | True | LSTM 1 | LSTM 1 |
| dak_nong | 1 | LSTM 2 | LSTM 3 | mae | 542 | -3.911 | 0.0001 | 0.0076 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | +1.692 | 0.0912 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(1) | mae | 542 | +1.701 | 0.0895 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(2) | mae | 542 | +1.476 | 0.1405 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VAR(5) | mae | 542 | +1.398 | 0.1626 | 1.0000 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX 1 | mae | 542 | -5.055 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | LSTM 2 | VARX 2 | mae | 542 | -3.071 | 0.0022 | 0.1144 | False | LSTM 2 | None (No significant difference) |
| dak_nong | 1 | LSTM 2 | VARX 3 | mae | 542 | -5.056 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | -3.961 | < 0.0001 | 0.0066 | True | LSTM 2 | LSTM 2 |
| dak_nong | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | +3.909 | 0.0001 | 0.0076 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | LSTM 3 | VAR(1) | mae | 542 | +4.040 | < 0.0001 | 0.0053 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | LSTM 3 | VAR(2) | mae | 542 | +3.967 | < 0.0001 | 0.0065 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | LSTM 3 | VAR(5) | mae | 542 | +3.980 | < 0.0001 | 0.0063 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | LSTM 3 | VARX 1 | mae | 542 | -4.083 | < 0.0001 | 0.0046 | True | LSTM 3 | LSTM 3 |
| dak_nong | 1 | LSTM 3 | VARX 2 | mae | 542 | +3.799 | 0.0002 | 0.0105 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | LSTM 3 | VARX 3 | mae | 542 | -4.173 | < 0.0001 | 0.0034 | True | LSTM 3 | LSTM 3 |
| dak_nong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +0.174 | 0.8616 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | -0.524 | 0.6002 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -0.990 | 0.3228 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -1.064 | 0.2880 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | -4.944 | < 0.0001 | 0.0001 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | -3.290 | 0.0011 | 0.0641 | False | Naive (Random Walk) | None (No significant difference) |
| dak_nong | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | -4.941 | < 0.0001 | 0.0001 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | -3.997 | < 0.0001 | 0.0061 | True | Naive (Random Walk) | Naive (Random Walk) |
| dak_nong | 1 | VAR(1) | VAR(2) | mae | 542 | -1.962 | 0.0502 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VAR(5) | mae | 542 | -1.859 | 0.0636 | 1.0000 | False | VAR(1) | None (No significant difference) |
| dak_nong | 1 | VAR(1) | VARX 1 | mae | 542 | -5.060 | < 0.0001 | < 0.0001 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | VAR(1) | VARX 2 | mae | 542 | -3.409 | 0.0007 | 0.0449 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | VAR(1) | VARX 3 | mae | 542 | -5.053 | < 0.0001 | < 0.0001 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | -4.114 | < 0.0001 | 0.0041 | True | VAR(1) | VAR(1) |
| dak_nong | 1 | VAR(2) | VAR(5) | mae | 542 | -0.644 | 0.5198 | 1.0000 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX 1 | mae | 542 | -5.005 | < 0.0001 | < 0.0001 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | VAR(2) | VARX 2 | mae | 542 | -3.322 | 0.0010 | 0.0592 | False | VAR(2) | None (No significant difference) |
| dak_nong | 1 | VAR(2) | VARX 3 | mae | 542 | -4.999 | < 0.0001 | < 0.0001 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | -4.040 | < 0.0001 | 0.0053 | True | VAR(2) | VAR(2) |
| dak_nong | 1 | VAR(5) | VARX 1 | mae | 542 | -5.035 | < 0.0001 | < 0.0001 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | VAR(5) | VARX 2 | mae | 542 | -3.338 | 0.0009 | 0.0569 | False | VAR(5) | None (No significant difference) |
| dak_nong | 1 | VAR(5) | VARX 3 | mae | 542 | -5.028 | < 0.0001 | < 0.0001 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | -4.060 | < 0.0001 | 0.0050 | True | VAR(5) | VAR(5) |
| dak_nong | 1 | VARX 1 | VARX 2 | mae | 542 | +6.794 | < 0.0001 | < 0.0001 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | VARX 1 | VARX 3 | mae | 542 | -0.067 | 0.9463 | 1.0000 | False | VARX 1 | None (No significant difference) |
| dak_nong | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +6.207 | < 0.0001 | < 0.0001 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_nong | 1 | VARX 2 | VARX 3 | mae | 542 | -6.777 | < 0.0001 | < 0.0001 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | -5.147 | < 0.0001 | < 0.0001 | True | VARX 2 | VARX 2 |
| dak_nong | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +6.324 | < 0.0001 | < 0.0001 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | +2.072 | 0.0388 | 0.3488 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | -1.945 | 0.0523 | 0.3729 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | -2.342 | 0.0196 | 0.2347 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | -1.994 | 0.0466 | 0.3729 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | -0.086 | 0.9315 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | -2.237 | 0.0257 | 0.2566 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | -4.032 | < 0.0001 | 0.0009 | True | DeepVAR 2 | DeepVAR 2 |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | -2.307 | 0.0214 | 0.2356 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | -0.637 | 0.5243 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | +1.492 | 0.1362 | 0.8047 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | +0.249 | 0.8034 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| dak_nong | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +3.826 | 0.0001 | 0.0020 | True | DeepVARX 3 | DeepVARX 3 |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | -1.500 | 0.1341 | 0.8047 | False | DeepVARX 1 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +0.296 | 0.7676 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| dak_nong | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +3.294 | 0.0011 | 0.0137 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +0.502 | 0.6156 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +3.166 | 0.0016 | 0.1504 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +0.738 | 0.4610 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +3.404 | 0.0007 | 0.0748 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.903 | 0.0001 | 0.0130 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | +0.966 | 0.3342 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +4.793 | < 0.0001 | 0.0003 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +3.137 | 0.0018 | 0.1606 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | +0.729 | 0.4664 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(1) | mse | 542 | +0.695 | 0.4871 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(2) | mse | 542 | +0.113 | 0.9105 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(5) | mse | 542 | +0.363 | 0.7166 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +2.894 | 0.0040 | 0.3010 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +3.905 | 0.0001 | 0.0129 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.867 | 0.0043 | 0.3233 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +3.503 | 0.0005 | 0.0542 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | +3.062 | 0.0023 | 0.1987 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -0.190 | 0.8490 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | +3.254 | 0.0012 | 0.1168 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | +3.753 | 0.0002 | 0.0226 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | +0.055 | 0.9562 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +5.753 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | +3.035 | 0.0025 | 0.2146 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | +0.176 | 0.8606 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(1) | mse | 542 | +0.168 | 0.8668 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(2) | mse | 542 | -0.475 | 0.6347 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(5) | mse | 542 | -0.253 | 0.8001 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 1 | mse | 542 | +2.769 | 0.0058 | 0.4071 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 2 | mse | 542 | +3.724 | 0.0002 | 0.0251 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVAR 2 | VARX 3 | mse | 542 | +2.741 | 0.0063 | 0.4360 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | +3.370 | 0.0008 | 0.0829 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -2.994 | 0.0029 | 0.2387 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -1.578 | 0.1150 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +1.044 | 0.2971 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -3.140 | 0.0018 | 0.1606 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | -1.426 | 0.1545 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | +0.536 | 0.5923 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -3.026 | 0.0026 | 0.2180 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -3.156 | 0.0017 | 0.1536 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -3.298 | 0.0010 | 0.1027 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -3.276 | 0.0011 | 0.1099 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -0.262 | 0.7933 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 2 | mse | 542 | -0.010 | 0.9924 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -0.342 | 0.7328 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | +0.754 | 0.4513 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +3.229 | 0.0013 | 0.1225 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +3.692 | 0.0002 | 0.0281 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | +0.227 | 0.8205 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +4.782 | < 0.0001 | 0.0003 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.967 | 0.0031 | 0.2541 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | +0.474 | 0.6359 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(1) | mse | 542 | +0.362 | 0.7176 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -0.340 | 0.7343 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(5) | mse | 542 | -0.100 | 0.9203 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +2.734 | 0.0065 | 0.4392 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +3.676 | 0.0003 | 0.0294 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +2.706 | 0.0070 | 0.4640 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +3.312 | 0.0010 | 0.0988 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +2.209 | 0.0276 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -3.348 | 0.0009 | 0.0888 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | -0.915 | 0.3608 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +1.655 | 0.0984 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -3.254 | 0.0012 | 0.1168 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -3.374 | 0.0008 | 0.0825 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -3.512 | 0.0005 | 0.0529 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -3.490 | 0.0005 | 0.0560 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 1 | mse | 542 | +0.852 | 0.3945 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +2.091 | 0.0370 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 3 | mse | 542 | +0.837 | 0.4028 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +2.228 | 0.0263 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.936 | < 0.0001 | 0.0115 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | -2.299 | 0.0219 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -0.787 | 0.4316 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.757 | 0.0002 | 0.0225 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.945 | < 0.0001 | 0.0112 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -4.097 | < 0.0001 | 0.0062 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -4.086 | < 0.0001 | 0.0064 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -0.800 | 0.4241 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -0.864 | 0.3880 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -0.868 | 0.3855 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -0.273 | 0.7849 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 2 | mse | 542 | +4.393 | < 0.0001 | 0.0017 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 1 | LSTM 3 | mse | 542 | +3.123 | 0.0019 | 0.1660 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | +0.004 | 0.9965 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(1) | mse | 542 | +0.051 | 0.9597 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(2) | mse | 542 | -0.486 | 0.6274 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(5) | mse | 542 | -0.313 | 0.7543 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.846 | 0.0046 | 0.3403 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 2 | mse | 542 | +3.902 | 0.0001 | 0.0130 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.824 | 0.0049 | 0.3591 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +3.493 | 0.0005 | 0.0559 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | LSTM 3 | mse | 542 | +1.448 | 0.1483 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -5.268 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(1) | mse | 542 | -5.323 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(2) | mse | 542 | -5.504 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(5) | mse | 542 | -5.569 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VARX 1 | mse | 542 | +1.102 | 0.2708 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 2 | mse | 542 | +1.803 | 0.0719 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 3 | mse | 542 | +1.075 | 0.2827 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | +1.746 | 0.0813 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.992 | 0.0029 | 0.2387 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(1) | mse | 542 | -3.110 | 0.0020 | 0.1711 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(2) | mse | 542 | -3.255 | 0.0012 | 0.1168 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(5) | mse | 542 | -3.241 | 0.0013 | 0.1187 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 1 | mse | 542 | -0.433 | 0.6654 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 2 | mse | 542 | -0.182 | 0.8558 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 3 | mse | 542 | -0.544 | 0.5866 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.658 | 0.5111 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | +0.093 | 0.9256 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | -0.719 | 0.4725 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | -0.431 | 0.6666 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +2.729 | 0.0066 | 0.4392 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +3.685 | 0.0003 | 0.0287 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +2.703 | 0.0071 | 0.4640 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +3.327 | 0.0009 | 0.0947 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(2) | mse | 542 | -1.649 | 0.0998 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(5) | mse | 542 | -0.745 | 0.4566 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 1 | mse | 542 | +2.805 | 0.0052 | 0.3754 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 2 | mse | 542 | +3.854 | 0.0001 | 0.0155 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(1) | VARX 3 | mse | 542 | +2.783 | 0.0056 | 0.3956 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +3.457 | 0.0006 | 0.0625 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VAR(5) | mse | 542 | +0.721 | 0.4715 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 1 | mse | 542 | +2.938 | 0.0034 | 0.2753 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 2 | mse | 542 | +4.020 | < 0.0001 | 0.0083 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(2) | VARX 3 | mse | 542 | +2.921 | 0.0036 | 0.2842 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +3.603 | 0.0003 | 0.0385 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| gia_lai | 1 | VAR(5) | VARX 1 | mse | 542 | +2.924 | 0.0036 | 0.2842 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 2 | mse | 542 | +4.025 | < 0.0001 | 0.0082 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(5) | VARX 3 | mse | 542 | +2.905 | 0.0038 | 0.2943 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +3.592 | 0.0004 | 0.0397 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| gia_lai | 1 | VARX 1 | VARX 2 | mse | 542 | +0.357 | 0.7213 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 3 | mse | 542 | -0.339 | 0.7345 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +1.520 | 0.1291 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX 3 | mse | 542 | -0.440 | 0.6602 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.189 | 0.2350 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +1.901 | 0.0578 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +0.390 | 0.6967 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +3.380 | 0.0008 | 0.0855 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | +2.164 | 0.0309 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +3.291 | 0.0011 | 0.1136 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +4.084 | < 0.0001 | 0.0065 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | +1.494 | 0.1358 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +6.192 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +3.222 | 0.0014 | 0.1402 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +1.834 | 0.0671 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(1) | mae | 542 | +1.918 | 0.0556 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(2) | mae | 542 | +1.372 | 0.1707 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VAR(5) | mae | 542 | +1.258 | 0.2088 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +2.856 | 0.0045 | 0.3920 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +4.233 | < 0.0001 | 0.0035 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +2.803 | 0.0053 | 0.4410 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +3.710 | 0.0002 | 0.0274 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | +3.173 | 0.0016 | 0.1624 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | +0.841 | 0.4009 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | +3.100 | 0.0020 | 0.2037 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | +3.594 | 0.0004 | 0.0416 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | +0.242 | 0.8085 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | +7.775 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | +3.022 | 0.0026 | 0.2472 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | +2.487 | 0.0132 | 0.9359 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(1) | mae | 542 | +2.153 | 0.0318 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(2) | mae | 542 | +1.199 | 0.2312 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VAR(5) | mae | 542 | +0.868 | 0.3857 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 1 | mae | 542 | +2.651 | 0.0082 | 0.6435 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX 2 | mae | 542 | +3.943 | < 0.0001 | 0.0113 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVAR 2 | VARX 3 | mae | 542 | +2.597 | 0.0097 | 0.7255 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | +3.464 | 0.0006 | 0.0644 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -3.056 | 0.0024 | 0.2243 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -1.320 | 0.1874 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | +0.306 | 0.7594 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -3.222 | 0.0013 | 0.1402 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | -1.063 | 0.2882 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -0.692 | 0.4890 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -2.778 | 0.0057 | 0.4693 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -2.827 | 0.0049 | 0.4191 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -2.958 | 0.0032 | 0.2972 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -3.058 | 0.0023 | 0.2243 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -0.671 | 0.5028 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 2 | mae | 542 | +0.489 | 0.6251 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -0.804 | 0.4218 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | +0.513 | 0.6080 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.968 | 0.0031 | 0.2909 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.665 | 0.0003 | 0.0324 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | -0.466 | 0.6412 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +6.373 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.906 | 0.0038 | 0.3390 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +1.034 | 0.3015 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(1) | mae | 542 | +1.189 | 0.2351 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(2) | mae | 542 | +0.315 | 0.7525 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -0.020 | 0.9843 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +2.543 | 0.0113 | 0.8334 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +3.894 | 0.0001 | 0.0137 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +2.492 | 0.0130 | 0.9359 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +3.384 | 0.0008 | 0.0849 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +1.029 | 0.3039 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -3.093 | 0.0021 | 0.2061 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | -0.717 | 0.4735 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | +0.882 | 0.3783 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.681 | 0.0076 | 0.6073 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.719 | 0.0068 | 0.5538 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.854 | 0.0045 | 0.3920 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.955 | 0.0033 | 0.2974 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 1 | mae | 542 | +0.057 | 0.9544 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 2 | mae | 542 | +1.596 | 0.1111 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -0.040 | 0.9684 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +1.502 | 0.1338 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -3.970 | < 0.0001 | 0.0102 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | -1.432 | 0.1526 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -0.492 | 0.6229 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.279 | 0.0011 | 0.1176 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.358 | 0.0008 | 0.0907 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.492 | 0.0005 | 0.0587 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.613 | 0.0003 | 0.0390 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -0.727 | 0.4675 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 2 | mae | 542 | +0.049 | 0.9608 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -0.806 | 0.4205 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | +0.042 | 0.9665 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | LSTM 2 | mae | 542 | +5.384 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 1 | LSTM 3 | mae | 542 | +3.071 | 0.0022 | 0.2195 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +0.849 | 0.3964 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(1) | mae | 542 | +0.991 | 0.3220 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(2) | mae | 542 | +0.528 | 0.5979 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VAR(5) | mae | 542 | +0.366 | 0.7148 | 1.0000 | False | VAR(5) | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 1 | mae | 542 | +2.683 | 0.0075 | 0.6073 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX 2 | mae | 542 | +4.038 | < 0.0001 | 0.0078 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | LSTM 1 | VARX 3 | mae | 542 | +2.631 | 0.0087 | 0.6700 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +3.536 | 0.0004 | 0.0508 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | LSTM 3 | mae | 542 | +0.936 | 0.3498 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -6.115 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(1) | mae | 542 | -6.359 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(2) | mae | 542 | -6.305 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VAR(5) | mae | 542 | -6.411 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| gia_lai | 1 | LSTM 2 | VARX 1 | mae | 542 | +0.614 | 0.5394 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 2 | mae | 542 | +1.510 | 0.1317 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX 3 | mae | 542 | +0.561 | 0.5750 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | +1.291 | 0.1971 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -2.633 | 0.0087 | 0.6700 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.684 | 0.0075 | 0.6073 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.821 | 0.0050 | 0.4222 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.920 | 0.0037 | 0.3286 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 1 | mae | 542 | -0.535 | 0.5932 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 2 | mae | 542 | +0.749 | 0.4542 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX 3 | mae | 542 | -0.683 | 0.4948 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| gia_lai | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +0.897 | 0.3701 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | +0.252 | 0.8013 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -0.722 | 0.4706 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -0.935 | 0.3503 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +2.284 | 0.0228 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +3.498 | 0.0005 | 0.0578 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +2.232 | 0.0260 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +3.065 | 0.0023 | 0.2217 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(2) | mae | 542 | -1.536 | 0.1252 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VAR(5) | mae | 542 | -1.650 | 0.0996 | 1.0000 | False | VAR(1) | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 1 | mae | 542 | +2.310 | 0.0213 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX 2 | mae | 542 | +3.590 | 0.0004 | 0.0418 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(1) | VARX 3 | mae | 542 | +2.259 | 0.0243 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +3.124 | 0.0019 | 0.1899 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VAR(5) | mae | 542 | -0.730 | 0.4656 | 1.0000 | False | VAR(2) | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 1 | mae | 542 | +2.445 | 0.0148 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX 2 | mae | 542 | +3.760 | 0.0002 | 0.0228 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(2) | VARX 3 | mae | 542 | +2.396 | 0.0169 | 1.0000 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +3.269 | 0.0011 | 0.1204 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 1 | mae | 542 | +2.539 | 0.0114 | 0.8334 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX 2 | mae | 542 | +3.892 | 0.0001 | 0.0137 | True | VARX 2 | VARX 2 |
| gia_lai | 1 | VAR(5) | VARX 3 | mae | 542 | +2.490 | 0.0131 | 0.9359 | False | VARX 3 | None (No significant difference) |
| gia_lai | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +3.378 | 0.0008 | 0.0855 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 2 | mae | 542 | +1.252 | 0.2110 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX 3 | mae | 542 | -0.605 | 0.5457 | 1.0000 | False | VARX 1 | None (No significant difference) |
| gia_lai | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +1.843 | 0.0659 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX 3 | mae | 542 | -1.375 | 0.1698 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | -0.010 | 0.9917 | 1.0000 | False | VARX 2 | None (No significant difference) |
| gia_lai | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +2.164 | 0.0309 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | -1.510 | 0.1316 | 0.2847 | False | DeepVAR 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +3.333 | 0.0009 | 0.0092 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +1.808 | 0.0712 | 0.2847 | False | DeepVARX 1 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +3.078 | 0.0022 | 0.0175 | True | DeepVARX 2 | DeepVARX 2 |
| gia_lai | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +4.553 | < 0.0001 | < 0.0001 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | +3.622 | 0.0003 | 0.0038 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | +2.869 | 0.0043 | 0.0287 | True | DeepVARX 1 | DeepVARX 1 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | +3.447 | 0.0006 | 0.0067 | True | DeepVARX 2 | DeepVARX 2 |
| gia_lai | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | +4.506 | < 0.0001 | 0.0001 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -3.149 | 0.0017 | 0.0156 | True | DeepVAR 3 | DeepVAR 3 |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | -1.794 | 0.0734 | 0.2847 | False | DeepVAR 3 | None (No significant difference) |
| gia_lai | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +1.465 | 0.1434 | 0.2847 | False | DeepVARX 3 | None (No significant difference) |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +2.883 | 0.0041 | 0.0287 | True | DeepVARX 2 | DeepVARX 2 |
| gia_lai | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +4.411 | < 0.0001 | 0.0002 | True | DeepVARX 3 | DeepVARX 3 |
| gia_lai | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +2.375 | 0.0179 | 0.0895 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | mse | 542 | +0.193 | 0.8473 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | mse | 542 | +2.980 | 0.0030 | 0.2935 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | mse | 542 | +0.045 | 0.9644 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | mse | 542 | +2.914 | 0.0037 | 0.3456 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | mse | 542 | +3.728 | 0.0002 | 0.0264 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 1 | LSTM 1 | mse | 542 | -0.342 | 0.7328 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 2 | mse | 542 | +4.954 | < 0.0001 | 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVAR 1 | LSTM 3 | mse | 542 | +2.740 | 0.0064 | 0.5208 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | Naive (Random Walk) | mse | 542 | +0.163 | 0.8710 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(1) | mse | 542 | +0.030 | 0.9763 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(2) | mse | 542 | -0.656 | 0.5120 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(5) | mse | 542 | -0.492 | 0.6232 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 1 | mse | 542 | +2.028 | 0.0431 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 2 | mse | 542 | +3.404 | 0.0007 | 0.0814 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 3 | mse | 542 | +2.107 | 0.0356 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mse | 542 | +3.150 | 0.0017 | 0.1809 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | mse | 542 | +2.910 | 0.0038 | 0.3459 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | mse | 542 | -0.183 | 0.8549 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | mse | 542 | +2.863 | 0.0044 | 0.3815 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | mse | 542 | +3.644 | 0.0003 | 0.0362 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 2 | LSTM 1 | mse | 542 | -0.344 | 0.7313 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 2 | mse | 542 | +5.028 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVAR 2 | LSTM 3 | mse | 542 | +2.668 | 0.0079 | 0.6366 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | Naive (Random Walk) | mse | 542 | -0.581 | 0.5617 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(1) | mse | 542 | -0.119 | 0.9053 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(2) | mse | 542 | -0.775 | 0.4389 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(5) | mse | 542 | -0.624 | 0.5331 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 1 | mse | 542 | +1.966 | 0.0499 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 2 | mse | 542 | +3.276 | 0.0011 | 0.1253 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 3 | mse | 542 | +2.042 | 0.0416 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mse | 542 | +3.056 | 0.0024 | 0.2402 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | mse | 542 | -2.785 | 0.0055 | 0.4704 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | mse | 542 | -1.865 | 0.0628 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | mse | 542 | +1.518 | 0.1295 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 1 | mse | 542 | -2.982 | 0.0030 | 0.2935 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 2 | mse | 542 | -1.218 | 0.2236 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 3 | mse | 542 | -0.609 | 0.5426 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | Naive (Random Walk) | mse | 542 | -2.909 | 0.0038 | 0.3459 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(1) | mse | 542 | -3.088 | 0.0021 | 0.2182 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(2) | mse | 542 | -3.183 | 0.0015 | 0.1657 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(5) | mse | 542 | -3.128 | 0.0019 | 0.1932 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 1 | mse | 542 | -1.473 | 0.1413 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 2 | mse | 542 | -0.393 | 0.6943 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 3 | mse | 542 | -1.372 | 0.1706 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mse | 542 | +0.159 | 0.8736 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | mse | 542 | +2.756 | 0.0061 | 0.5022 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | mse | 542 | +3.473 | 0.0006 | 0.0645 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 1 | mse | 542 | -0.316 | 0.7524 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 2 | mse | 542 | +4.369 | < 0.0001 | 0.0019 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVARX 1 | LSTM 3 | mse | 542 | +2.569 | 0.0105 | 0.8154 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | Naive (Random Walk) | mse | 542 | +0.152 | 0.8792 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(1) | mse | 542 | +0.001 | 0.9989 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(2) | mse | 542 | -0.647 | 0.5177 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(5) | mse | 542 | -0.496 | 0.6198 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 1 | mse | 542 | +1.926 | 0.0546 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 2 | mse | 542 | +3.184 | 0.0015 | 0.1657 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 3 | mse | 542 | +1.999 | 0.0461 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mse | 542 | +2.963 | 0.0032 | 0.3022 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | mse | 542 | +2.589 | 0.0099 | 0.7799 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 1 | mse | 542 | -2.895 | 0.0039 | 0.3548 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 2 | mse | 542 | -0.354 | 0.7237 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 3 | mse | 542 | +1.474 | 0.1410 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | Naive (Random Walk) | mse | 542 | -2.865 | 0.0043 | 0.3815 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(1) | mse | 542 | -2.977 | 0.0030 | 0.2935 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(2) | mse | 542 | -3.056 | 0.0024 | 0.2402 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(5) | mse | 542 | -3.020 | 0.0026 | 0.2620 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 1 | mse | 542 | -0.230 | 0.8184 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 2 | mse | 542 | +1.848 | 0.0652 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 3 | mse | 542 | -0.071 | 0.9437 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mse | 542 | +2.051 | 0.0407 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 1 | mse | 542 | -3.735 | 0.0002 | 0.0259 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | LSTM 2 | mse | 542 | -2.180 | 0.0297 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 3 | mse | 542 | -1.455 | 0.1462 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | Naive (Random Walk) | mse | 542 | -3.643 | 0.0003 | 0.0362 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VAR(1) | mse | 542 | -3.886 | 0.0001 | 0.0145 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VAR(2) | mse | 542 | -3.970 | < 0.0001 | 0.0105 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VAR(5) | mse | 542 | -3.922 | < 0.0001 | 0.0126 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VARX 1 | mse | 542 | -1.836 | 0.0669 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 2 | mse | 542 | -1.317 | 0.1885 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 3 | mse | 542 | -1.745 | 0.0816 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mse | 542 | -0.770 | 0.4414 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 2 | mse | 542 | +4.310 | < 0.0001 | 0.0025 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 1 | LSTM 3 | mse | 542 | +2.762 | 0.0059 | 0.4995 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | Naive (Random Walk) | mse | 542 | +0.330 | 0.7415 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(1) | mse | 542 | +0.265 | 0.7913 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(2) | mse | 542 | -0.352 | 0.7252 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(5) | mse | 542 | -0.133 | 0.8941 | 1.0000 | False | LSTM 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 1 | mse | 542 | +2.077 | 0.0383 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 2 | mse | 542 | +3.449 | 0.0006 | 0.0699 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 3 | mse | 542 | +2.157 | 0.0314 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mse | 542 | +3.174 | 0.0016 | 0.1687 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | LSTM 3 | mse | 542 | +1.022 | 0.3070 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | Naive (Random Walk) | mse | 542 | -4.992 | < 0.0001 | 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(1) | mse | 542 | -4.954 | < 0.0001 | 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(2) | mse | 542 | -4.958 | < 0.0001 | 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(5) | mse | 542 | -5.073 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VARX 1 | mse | 542 | +0.094 | 0.9255 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 2 | mse | 542 | +1.185 | 0.2367 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 3 | mse | 542 | +0.189 | 0.8505 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mse | 542 | +1.261 | 0.2079 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | Naive (Random Walk) | mse | 542 | -2.667 | 0.0079 | 0.6366 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(1) | mse | 542 | -2.823 | 0.0049 | 0.4246 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(2) | mse | 542 | -2.932 | 0.0035 | 0.3302 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(5) | mse | 542 | -2.880 | 0.0041 | 0.3684 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 1 | mse | 542 | -1.473 | 0.1413 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 2 | mse | 542 | -0.187 | 0.8518 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 3 | mse | 542 | -1.363 | 0.1733 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mse | 542 | +0.396 | 0.6925 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(1) | mse | 542 | -0.096 | 0.9237 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(2) | mse | 542 | -0.757 | 0.4492 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(5) | mse | 542 | -0.603 | 0.5470 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 1 | mse | 542 | +1.967 | 0.0497 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 2 | mse | 542 | +3.276 | 0.0011 | 0.1253 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 3 | mse | 542 | +2.043 | 0.0415 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mse | 542 | +3.055 | 0.0024 | 0.2402 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(2) | mse | 542 | -1.561 | 0.1192 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(5) | mse | 542 | -0.942 | 0.3468 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 1 | mse | 542 | +2.059 | 0.0400 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 2 | mse | 542 | +3.507 | 0.0005 | 0.0574 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 3 | mse | 542 | +2.141 | 0.0327 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX-L 2 (Ridge) | mse | 542 | +3.216 | 0.0014 | 0.1501 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VAR(5) | mse | 542 | +0.830 | 0.4070 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 1 | mse | 542 | +2.195 | 0.0286 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 2 | mse | 542 | +3.621 | 0.0003 | 0.0388 | True | VARX 2 | VARX 2 |
| lam_dong | 1 | VAR(2) | VARX 3 | mse | 542 | +2.277 | 0.0232 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX-L 2 (Ridge) | mse | 542 | +3.320 | 0.0010 | 0.1087 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 1 | mse | 542 | +2.143 | 0.0326 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 2 | mse | 542 | +3.573 | 0.0004 | 0.0462 | True | VARX 2 | VARX 2 |
| lam_dong | 1 | VAR(5) | VARX 3 | mse | 542 | +2.224 | 0.0265 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX-L 2 (Ridge) | mse | 542 | +3.272 | 0.0011 | 0.1253 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 2 | mse | 542 | +1.751 | 0.0805 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 3 | mse | 542 | +1.527 | 0.1274 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX-L 2 (Ridge) | mse | 542 | +3.568 | 0.0004 | 0.0465 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| lam_dong | 1 | VARX 2 | VARX 3 | mse | 542 | -1.551 | 0.1215 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX-L 2 (Ridge) | mse | 542 | +1.051 | 0.2935 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 3 | VARX-L 2 (Ridge) | mse | 542 | +3.513 | 0.0005 | 0.0566 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | mae | 542 | +1.375 | 0.1698 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | mae | 542 | +2.982 | 0.0030 | 0.3138 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | mae | 542 | +2.042 | 0.0416 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | mae | 542 | +2.636 | 0.0086 | 0.8195 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | mae | 542 | +3.943 | < 0.0001 | 0.0115 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 1 | LSTM 1 | mae | 542 | -0.579 | 0.5627 | 1.0000 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | LSTM 2 | mae | 542 | +6.373 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVAR 1 | LSTM 3 | mae | 542 | +2.487 | 0.0132 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | Naive (Random Walk) | mae | 542 | +1.435 | 0.1518 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(1) | mae | 542 | +1.410 | 0.1591 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(2) | mae | 542 | +0.559 | 0.5762 | 1.0000 | False | VAR(2) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VAR(5) | mae | 542 | +0.730 | 0.4656 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 1 | mae | 542 | +1.974 | 0.0489 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX 2 | mae | 542 | +3.677 | 0.0003 | 0.0317 | True | VARX 2 | VARX 2 |
| lam_dong | 1 | DeepVAR 1 | VARX 3 | mae | 542 | +2.047 | 0.0411 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | VARX-L 2 (Ridge) | mae | 542 | +3.230 | 0.0013 | 0.1457 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | mae | 542 | +2.573 | 0.0104 | 0.9530 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | mae | 542 | -0.145 | 0.8848 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | mae | 542 | +2.242 | 0.0253 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | mae | 542 | +3.423 | 0.0007 | 0.0774 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 1 | mae | 542 | -1.197 | 0.2318 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | LSTM 2 | mae | 542 | +6.384 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVAR 2 | LSTM 3 | mae | 542 | +2.088 | 0.0373 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | Naive (Random Walk) | mae | 542 | +1.269 | 0.2049 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(1) | mae | 542 | +0.121 | 0.9038 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(2) | mae | 542 | -0.709 | 0.4786 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VAR(5) | mae | 542 | -0.638 | 0.5235 | 1.0000 | False | DeepVAR 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 1 | mae | 542 | +1.584 | 0.1138 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 2 | mae | 542 | +3.152 | 0.0017 | 0.1881 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX 3 | mae | 542 | +1.652 | 0.0992 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | VARX-L 2 (Ridge) | mae | 542 | +2.785 | 0.0055 | 0.5655 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | mae | 542 | -2.662 | 0.0080 | 0.7686 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | mae | 542 | -1.494 | 0.1357 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | mae | 542 | +1.211 | 0.2266 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 1 | mae | 542 | -3.091 | 0.0021 | 0.2266 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 2 | mae | 542 | -0.485 | 0.6279 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | LSTM 3 | mae | 542 | -2.310 | 0.0213 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | Naive (Random Walk) | mae | 542 | -2.556 | 0.0108 | 0.9872 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(1) | mae | 542 | -2.633 | 0.0087 | 0.8195 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(2) | mae | 542 | -2.757 | 0.0060 | 0.6022 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VAR(5) | mae | 542 | -2.757 | 0.0060 | 0.6022 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 1 | mae | 542 | -1.551 | 0.1214 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 2 | mae | 542 | +0.595 | 0.5518 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX 3 | mae | 542 | -1.438 | 0.1510 | 1.0000 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | VARX-L 2 (Ridge) | mae | 542 | +0.458 | 0.6472 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | mae | 542 | +2.324 | 0.0205 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | mae | 542 | +3.621 | 0.0003 | 0.0386 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 1 | LSTM 1 | mae | 542 | -1.612 | 0.1075 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | LSTM 2 | mae | 542 | +6.186 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | DeepVARX 1 | LSTM 3 | mae | 542 | +2.172 | 0.0303 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | Naive (Random Walk) | mae | 542 | +0.238 | 0.8118 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(1) | mae | 542 | +0.287 | 0.7742 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(2) | mae | 542 | -0.778 | 0.4371 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VAR(5) | mae | 542 | -0.780 | 0.4355 | 1.0000 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 1 | mae | 542 | +1.663 | 0.0969 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 2 | mae | 542 | +3.342 | 0.0009 | 0.1013 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX 3 | mae | 542 | +1.736 | 0.0830 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | VARX-L 2 (Ridge) | mae | 542 | +2.912 | 0.0037 | 0.3885 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | mae | 542 | +1.925 | 0.0548 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 1 | mae | 542 | -2.729 | 0.0066 | 0.6433 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 2 | mae | 542 | -0.034 | 0.9729 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | LSTM 3 | mae | 542 | -0.062 | 0.9509 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | Naive (Random Walk) | mae | 542 | -2.225 | 0.0265 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(1) | mae | 542 | -2.293 | 0.0222 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(2) | mae | 542 | -2.434 | 0.0153 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VAR(5) | mae | 542 | -2.430 | 0.0154 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 1 | mae | 542 | -0.814 | 0.4158 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 2 | mae | 542 | +1.653 | 0.0989 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX 3 | mae | 542 | -0.684 | 0.4943 | 1.0000 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 2 | VARX-L 2 (Ridge) | mae | 542 | +1.429 | 0.1535 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 1 | mae | 542 | -4.123 | < 0.0001 | 0.0055 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | LSTM 2 | mae | 542 | -1.275 | 0.2027 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | LSTM 3 | mae | 542 | -2.019 | 0.0440 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | Naive (Random Walk) | mae | 542 | -3.411 | 0.0007 | 0.0800 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(1) | mae | 542 | -3.549 | 0.0004 | 0.0500 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VAR(2) | mae | 542 | -3.674 | 0.0003 | 0.0317 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VAR(5) | mae | 542 | -3.702 | 0.0002 | 0.0290 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 3 | VARX 1 | mae | 542 | -2.007 | 0.0453 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 2 | mae | 542 | -0.559 | 0.5763 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX 3 | mae | 542 | -1.892 | 0.0590 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 3 | VARX-L 2 (Ridge) | mae | 542 | -0.602 | 0.5473 | 1.0000 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | LSTM 2 | mae | 542 | +5.453 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 1 | LSTM 3 | mae | 542 | +2.605 | 0.0094 | 0.8776 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | Naive (Random Walk) | mae | 542 | +1.244 | 0.2142 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(1) | mae | 542 | +1.406 | 0.1603 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(2) | mae | 542 | +0.806 | 0.4208 | 1.0000 | False | VAR(2) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VAR(5) | mae | 542 | +0.951 | 0.3420 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 1 | mae | 542 | +2.095 | 0.0366 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX 2 | mae | 542 | +3.819 | 0.0001 | 0.0187 | True | VARX 2 | VARX 2 |
| lam_dong | 1 | LSTM 1 | VARX 3 | mae | 542 | +2.171 | 0.0304 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 1 | VARX-L 2 (Ridge) | mae | 542 | +3.332 | 0.0009 | 0.1043 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | LSTM 3 | mae | 542 | +0.012 | 0.9906 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | Naive (Random Walk) | mae | 542 | -6.289 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(1) | mae | 542 | -6.057 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(2) | mae | 542 | -5.834 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VAR(5) | mae | 542 | -5.854 | < 0.0001 | < 0.0001 | True | LSTM 2 | LSTM 2 |
| lam_dong | 1 | LSTM 2 | VARX 1 | mae | 542 | -0.413 | 0.6799 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 2 | mae | 542 | +0.828 | 0.4078 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX 3 | mae | 542 | -0.333 | 0.7396 | 1.0000 | False | LSTM 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 2 | VARX-L 2 (Ridge) | mae | 542 | +0.680 | 0.4969 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | Naive (Random Walk) | mae | 542 | -2.072 | 0.0388 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(1) | mae | 542 | -2.138 | 0.0329 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(2) | mae | 542 | -2.285 | 0.0227 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VAR(5) | mae | 542 | -2.285 | 0.0227 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 1 | mae | 542 | -0.964 | 0.3356 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 2 | mae | 542 | +1.689 | 0.0918 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX 3 | mae | 542 | -0.811 | 0.4180 | 1.0000 | False | LSTM 3 | None (No significant difference) |
| lam_dong | 1 | LSTM 3 | VARX-L 2 (Ridge) | mae | 542 | +1.798 | 0.0727 | 1.0000 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(1) | mae | 542 | +0.038 | 0.9695 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(2) | mae | 542 | -0.768 | 0.4429 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VAR(5) | mae | 542 | -0.696 | 0.4866 | 1.0000 | False | Naive (Random Walk) | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 1 | mae | 542 | +1.568 | 0.1174 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 2 | mae | 542 | +3.134 | 0.0018 | 0.1982 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX 3 | mae | 542 | +1.636 | 0.1023 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | Naive (Random Walk) | VARX-L 2 (Ridge) | mae | 542 | +2.768 | 0.0058 | 0.5888 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(2) | mae | 542 | -1.608 | 0.1084 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VAR(5) | mae | 542 | -1.280 | 0.2009 | 1.0000 | False | VAR(1) | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 1 | mae | 542 | +1.608 | 0.1084 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 2 | mae | 542 | +3.294 | 0.0011 | 0.1180 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX 3 | mae | 542 | +1.681 | 0.0933 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(1) | VARX-L 2 (Ridge) | mae | 542 | +2.848 | 0.0046 | 0.4697 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VAR(5) | mae | 542 | +0.192 | 0.8479 | 1.0000 | False | VAR(5) | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 1 | mae | 542 | +1.790 | 0.0741 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 2 | mae | 542 | +3.464 | 0.0006 | 0.0673 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX 3 | mae | 542 | +1.861 | 0.0633 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(2) | VARX-L 2 (Ridge) | mae | 542 | +2.995 | 0.0029 | 0.3045 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 1 | mae | 542 | +1.789 | 0.0742 | 1.0000 | False | VARX 1 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 2 | mae | 542 | +3.472 | 0.0006 | 0.0658 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX 3 | mae | 542 | +1.862 | 0.0632 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VAR(5) | VARX-L 2 (Ridge) | mae | 542 | +3.001 | 0.0028 | 0.3016 | False | VARX-L 2 (Ridge) | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 2 | mae | 542 | +2.703 | 0.0071 | 0.6874 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX 3 | mae | 542 | +0.955 | 0.3401 | 1.0000 | False | VARX 3 | None (No significant difference) |
| lam_dong | 1 | VARX 1 | VARX-L 2 (Ridge) | mae | 542 | +3.963 | < 0.0001 | 0.0106 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| lam_dong | 1 | VARX 2 | VARX 3 | mae | 542 | -2.492 | 0.0130 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 2 | VARX-L 2 (Ridge) | mae | 542 | -0.191 | 0.8482 | 1.0000 | False | VARX 2 | None (No significant difference) |
| lam_dong | 1 | VARX 3 | VARX-L 2 (Ridge) | mae | 542 | +3.801 | 0.0002 | 0.0199 | True | VARX-L 2 (Ridge) | VARX-L 2 (Ridge) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 2 | crps | 542 | -1.796 | 0.0731 | 0.1718 | False | DeepVAR 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVAR 3 | crps | 542 | +2.982 | 0.0030 | 0.0329 | True | DeepVAR 3 | DeepVAR 3 |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 1 | crps | 542 | +0.568 | 0.5704 | 0.5704 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 2 | crps | 542 | +2.121 | 0.0344 | 0.1718 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 1 | DeepVARX 3 | crps | 542 | +4.311 | < 0.0001 | 0.0003 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 2 | DeepVAR 3 | crps | 542 | +3.255 | 0.0012 | 0.0145 | True | DeepVAR 3 | DeepVAR 3 |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 1 | crps | 542 | +2.542 | 0.0113 | 0.0731 | False | DeepVARX 1 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 2 | crps | 542 | +2.570 | 0.0104 | 0.0731 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 2 | DeepVARX 3 | crps | 542 | +4.291 | < 0.0001 | 0.0003 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 1 | crps | 542 | -2.944 | 0.0034 | 0.0329 | True | DeepVAR 3 | DeepVAR 3 |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 2 | crps | 542 | -2.668 | 0.0079 | 0.0629 | False | DeepVAR 3 | None (No significant difference) |
| lam_dong | 1 | DeepVAR 3 | DeepVARX 3 | crps | 542 | +2.004 | 0.0455 | 0.1718 | False | DeepVARX 3 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 2 | crps | 542 | +2.094 | 0.0367 | 0.1718 | False | DeepVARX 2 | None (No significant difference) |
| lam_dong | 1 | DeepVARX 1 | DeepVARX 3 | crps | 542 | +4.306 | < 0.0001 | 0.0003 | True | DeepVARX 3 | DeepVARX 3 |
| lam_dong | 1 | DeepVARX 2 | DeepVARX 3 | crps | 542 | +2.962 | 0.0032 | 0.0329 | True | DeepVARX 3 | DeepVARX 3 |

## 3. Tổng hợp các cặp có ưu thế thống kê vượt trội (Statistically Significant Differences)

Có **213** phép kiểm định xác nhận sự khác biệt có ý nghĩa thống kê:

- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.022, p-value Holm=0.0085).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+5.290, p-value Holm=0.0000).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.225, p-value Holm=0.0037).
- **dak_lak** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.577, p-value Holm=0.0480).
- **dak_lak** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.605, p-value Holm=0.0437).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.146, p-value Holm=0.0051).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.317, p-value Holm=0.0000).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.083, p-value Holm=0.0001).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-4.647, p-value Holm=0.0006).
- **dak_lak** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-4.537, p-value Holm=0.0009).
- **dak_lak** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+4.061, p-value Holm=0.0072).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+7.230, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.675, p-value Holm=0.0324).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+7.415, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+7.412, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-4.038, p-value Holm=0.0078).
- **dak_lak** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.603, p-value Holm=0.0420).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.026, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.635, p-value Holm=0.0375).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-7.230, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-7.036, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.663, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.422, p-value Holm=0.0000).
- **dak_lak** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.708, p-value Holm=0.0288).
- **dak_lak** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.745, p-value Holm=0.0251).
- **dak_lak** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+2.835, p-value Holm=0.0476).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.274, p-value Holm=0.0003).
- **dak_lak** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+2.977, p-value Holm=0.0365).
- **dak_lak** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+2.877, p-value Holm=0.0459).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.172, p-value Holm=0.0005).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.104, p-value Holm=0.0006).
- **dak_lak** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+2.824, p-value Holm=0.0476).
- **dak_nong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.741, p-value Holm=0.0266).
- **dak_nong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.742, p-value Holm=0.0266).
- **dak_nong** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.823, p-value Holm=0.0197).
- **dak_nong** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+4.781, p-value Holm=0.0003).
- **dak_nong** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=-3.789, p-value Holm=0.0224).
- **dak_nong** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+4.893, p-value Holm=0.0002).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=+4.449, p-value Holm=0.0011).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=+5.852, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=+5.305, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=+5.715, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=+5.633, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=+5.829, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **DeepVAR 1** vượt trội hơn (HLN stat=-4.202, p-value Holm=0.0031).
- **dak_nong** (h=1, MAE): **DeepVAR 1** vượt trội hơn (HLN stat=-4.200, p-value Holm=0.0031).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.821, p-value Holm=0.0099).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-4.716, p-value Holm=0.0003).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.970, p-value Holm=0.0065).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.845, p-value Holm=0.0093).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-4.877, p-value Holm=0.0002).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-4.867, p-value Holm=0.0002).
- **dak_nong** (h=1, MAE): **DeepVAR 2** vượt trội hơn (HLN stat=-3.931, p-value Holm=0.0072).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=+3.807, p-value Holm=0.0104).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+3.843, p-value Holm=0.0093).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=+3.882, p-value Holm=0.0083).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=+4.018, p-value Holm=0.0057).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=+3.935, p-value Holm=0.0071).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=+3.946, p-value Holm=0.0069).
- **dak_nong** (h=1, MAE): **DeepVAR 3** vượt trội hơn (HLN stat=-4.252, p-value Holm=0.0025).
- **dak_nong** (h=1, MAE): **DeepVAR 3** vượt trội hơn (HLN stat=-4.317, p-value Holm=0.0019).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=+5.305, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=+5.394, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=+5.810, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=+5.737, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=+6.096, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **DeepVARX 1** vượt trội hơn (HLN stat=-4.152, p-value Holm=0.0036).
- **dak_nong** (h=1, MAE): **DeepVARX 1** vượt trội hơn (HLN stat=-4.149, p-value Holm=0.0036).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=+3.991, p-value Holm=0.0061).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+4.008, p-value Holm=0.0059).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=+4.050, p-value Holm=0.0052).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=+4.210, p-value Holm=0.0030).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=+4.110, p-value Holm=0.0042).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=+4.122, p-value Holm=0.0040).
- **dak_nong** (h=1, MAE): **DeepVARX 2** vượt trội hơn (HLN stat=-4.177, p-value Holm=0.0033).
- **dak_nong** (h=1, MAE): **DeepVARX 2** vượt trội hơn (HLN stat=-4.207, p-value Holm=0.0030).
- **dak_nong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-5.052, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-5.052, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=-3.858, p-value Holm=0.0090).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=-4.991, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=-4.989, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **LSTM 1** vượt trội hơn (HLN stat=-3.929, p-value Holm=0.0072).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-3.911, p-value Holm=0.0076).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-5.055, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-5.056, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-3.961, p-value Holm=0.0066).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=+3.909, p-value Holm=0.0076).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=+4.040, p-value Holm=0.0053).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=+3.967, p-value Holm=0.0065).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=+3.980, p-value Holm=0.0063).
- **dak_nong** (h=1, MAE): **LSTM 3** vượt trội hơn (HLN stat=-4.083, p-value Holm=0.0046).
- **dak_nong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.799, p-value Holm=0.0105).
- **dak_nong** (h=1, MAE): **LSTM 3** vượt trội hơn (HLN stat=-4.173, p-value Holm=0.0034).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=-4.944, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=-4.941, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **Naive (Random Walk)** vượt trội hơn (HLN stat=-3.997, p-value Holm=0.0061).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=-5.060, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=-3.409, p-value Holm=0.0449).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=-5.053, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(1)** vượt trội hơn (HLN stat=-4.114, p-value Holm=0.0041).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=-5.005, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=-4.999, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(2)** vượt trội hơn (HLN stat=-4.040, p-value Holm=0.0053).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=-5.035, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=-5.028, p-value Holm=0.0001).
- **dak_nong** (h=1, MAE): **VAR(5)** vượt trội hơn (HLN stat=-4.060, p-value Holm=0.0050).
- **dak_nong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+6.794, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+6.207, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=-6.777, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=-5.147, p-value Holm=0.0000).
- **dak_nong** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+6.324, p-value Holm=0.0000).
- **dak_nong** (h=1, CRPS): **DeepVAR 2** vượt trội hơn (HLN stat=-4.032, p-value Holm=0.0009).
- **dak_nong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.826, p-value Holm=0.0020).
- **dak_nong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+3.294, p-value Holm=0.0137).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.903, p-value Holm=0.0130).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.793, p-value Holm=0.0003).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.905, p-value Holm=0.0129).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.753, p-value Holm=0.0226).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+5.753, p-value Holm=0.0000).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.724, p-value Holm=0.0251).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.692, p-value Holm=0.0281).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.782, p-value Holm=0.0003).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.676, p-value Holm=0.0294).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.936, p-value Holm=0.0115).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.757, p-value Holm=0.0225).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.945, p-value Holm=0.0112).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-4.097, p-value Holm=0.0062).
- **gia_lai** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-4.086, p-value Holm=0.0064).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.393, p-value Holm=0.0017).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.902, p-value Holm=0.0130).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.268, p-value Holm=0.0000).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.323, p-value Holm=0.0000).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.504, p-value Holm=0.0000).
- **gia_lai** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.569, p-value Holm=0.0000).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.685, p-value Holm=0.0287).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.854, p-value Holm=0.0155).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+4.020, p-value Holm=0.0083).
- **gia_lai** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.603, p-value Holm=0.0385).
- **gia_lai** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+4.025, p-value Holm=0.0082).
- **gia_lai** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.592, p-value Holm=0.0397).
- **gia_lai** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+4.084, p-value Holm=0.0065).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.192, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+4.233, p-value Holm=0.0035).
- **gia_lai** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.710, p-value Holm=0.0274).
- **gia_lai** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.594, p-value Holm=0.0416).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+7.775, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.943, p-value Holm=0.0113).
- **gia_lai** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.665, p-value Holm=0.0324).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.373, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.894, p-value Holm=0.0137).
- **gia_lai** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.970, p-value Holm=0.0102).
- **gia_lai** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.613, p-value Holm=0.0390).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+5.384, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+4.038, p-value Holm=0.0078).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.115, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.359, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.305, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.411, p-value Holm=0.0000).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.590, p-value Holm=0.0418).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.760, p-value Holm=0.0228).
- **gia_lai** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.892, p-value Holm=0.0137).
- **gia_lai** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.333, p-value Holm=0.0092).
- **gia_lai** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+3.078, p-value Holm=0.0175).
- **gia_lai** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.553, p-value Holm=0.0001).
- **gia_lai** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.622, p-value Holm=0.0038).
- **gia_lai** (h=1, CRPS): **DeepVARX 1** vượt trội hơn (HLN stat=+2.869, p-value Holm=0.0287).
- **gia_lai** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+3.447, p-value Holm=0.0067).
- **gia_lai** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.506, p-value Holm=0.0001).
- **gia_lai** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=-3.149, p-value Holm=0.0156).
- **gia_lai** (h=1, CRPS): **DeepVARX 2** vượt trội hơn (HLN stat=+2.883, p-value Holm=0.0287).
- **gia_lai** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.411, p-value Holm=0.0002).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.728, p-value Holm=0.0264).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.954, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.644, p-value Holm=0.0362).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+5.028, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.369, p-value Holm=0.0019).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.735, p-value Holm=0.0259).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.643, p-value Holm=0.0362).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.886, p-value Holm=0.0145).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.970, p-value Holm=0.0105).
- **lam_dong** (h=1, MSE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.922, p-value Holm=0.0126).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=+4.310, p-value Holm=0.0025).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-4.992, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-4.954, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-4.958, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **LSTM 2** vượt trội hơn (HLN stat=-5.073, p-value Holm=0.0001).
- **lam_dong** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.621, p-value Holm=0.0388).
- **lam_dong** (h=1, MSE): **VARX 2** vượt trội hơn (HLN stat=+3.573, p-value Holm=0.0462).
- **lam_dong** (h=1, MSE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.568, p-value Holm=0.0465).
- **lam_dong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.943, p-value Holm=0.0115).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.373, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.677, p-value Holm=0.0317).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.384, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=+3.621, p-value Holm=0.0386).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+6.186, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-4.123, p-value Holm=0.0055).
- **lam_dong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.674, p-value Holm=0.0317).
- **lam_dong** (h=1, MAE): **DeepVARX 3** vượt trội hơn (HLN stat=-3.702, p-value Holm=0.0290).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=+5.453, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **VARX 2** vượt trội hơn (HLN stat=+3.819, p-value Holm=0.0187).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.289, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-6.057, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-5.834, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **LSTM 2** vượt trội hơn (HLN stat=-5.854, p-value Holm=0.0000).
- **lam_dong** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.963, p-value Holm=0.0106).
- **lam_dong** (h=1, MAE): **VARX-L 2 (Ridge)** vượt trội hơn (HLN stat=+3.801, p-value Holm=0.0199).
- **lam_dong** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+2.982, p-value Holm=0.0329).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.311, p-value Holm=0.0003).
- **lam_dong** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=+3.255, p-value Holm=0.0145).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.291, p-value Holm=0.0003).
- **lam_dong** (h=1, CRPS): **DeepVAR 3** vượt trội hơn (HLN stat=-2.944, p-value Holm=0.0329).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+4.306, p-value Holm=0.0003).
- **lam_dong** (h=1, CRPS): **DeepVARX 3** vượt trội hơn (HLN stat=+2.962, p-value Holm=0.0329).

---
*Báo cáo được sinh tự động bởi Statistical Evaluation Module (Diebold-Mariano Suite).*
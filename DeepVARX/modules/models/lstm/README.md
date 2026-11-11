# Hệ thống mô hình LSTM dự báo giá cà phê Robusta Việt Nam

Thư mục này đóng gói độc lập mã nguồn, dữ liệu đầu vào, bảng kết quả và biểu đồ trực quan cho 3 phiên bản mô hình **LSTM** đối chứng với **Naive (Random Walk)** và **LSTM-Level Raw**, kèm kiểm định thống kê chuẩn hóa và Ablation Study.

## 1. Cấu trúc thư mục
- `lstm_coffee_models.py`: Mô-đun xử lý chuỗi dừng (`log-return`), kiểm định chẩn đoán trên Train (`ADF` đầy đủ 3 tỉnh & ngoại sinh, `VIF`), kiến trúc PyTorch `CoffeeLSTM`, mô hình đối chứng mức giá thô `evaluate_raw_price_level_lstm`, kiểm định `Diebold-Mariano (HAC Newey-West + HLN)` hỗ trợ chuỗi đơn tỉnh và tổng hợp panel $\bar{d}_t$ (không dùng `flatten()`), `Permutation Importance` đơn biến và `Group Permutation Importance`.
- `run_lstm_experiments.py`: Chương trình thực nghiệm chính (5 seeds: `42, 123, 456, 789, 2024`, kiểm định DM cặp theo panel và từng tỉnh, Ablation Study theo nhóm biến, xuất toàn bộ CSV và 4 biểu đồ).
- `data_daily_panel_clean_v2.csv`: Tập dữ liệu panel ngày (2013–2026) cho Đắk Lắk, Lâm Đồng, Gia Lai (và Đắk Nông, Tây Nguyên) cùng các biến thời tiết/kinh tế/mùa vụ.
- `benchmark_model_comparison_results.csv`: Bảng benchmark VAR/VARX/VARX-L/DeepVAR/DeepVARX cùng tập Test 542 ngày để đối chiếu.
- `lstm_model_comparison_results.csv`: Bảng kết quả tổng hợp (`Naive`, `LSTM-Level Raw`, `LSTM 1`, `LSTM 2`, `LSTM 3`).
- `lstm_province_results.csv`: Sai số và kiểm định DM-HLN-HAC chi tiết theo từng tỉnh (`dak_lak`, `lam_dong`, `gia_lai`).
- `lstm_dm_tests.csv`: Bảng kiểm định Diebold-Mariano (HAC Newey-West $L=5$, HLN) cho cả chuỗi tổng hợp theo ngày $\bar{d}_t$ và từng tỉnh riêng biệt.
- `lstm_feature_importance.csv` & `lstm_group_importance.csv`: Tầm quan trọng đặc trưng đơn biến và theo nhóm (`Group Permutation Importance`) của `LSTM 2` và `LSTM 3`.
- `lstm_ablation_results.csv`: Kết quả Ablation Study qua 10 cấu hình nhóm biến (cùng `seq_len=10`, 5 seeds).
- `lstm_diagnostics_adf.csv` & `lstm_diagnostics_vif.csv`: Kết quả kiểm định tính dừng `ADF` (30 chuỗi) và đa cộng tuyến `VIF` trên tập Train.
- `figures_lstm/`:
  - `fig1_price_regime_and_forecast.png`
  - `fig2_lstm_versions_comparison.png`
  - `fig3_learning_curves_and_importance.png`
  - `fig4_ablation_and_group_importance.png`

## 2. Chạy lại toàn bộ thực nghiệm
```powershell
cd D:\TLCN\lstm
python -u run_lstm_experiments.py
```

## 3. Chiến lược đánh giá & Tóm tắt kết quả (542 ngày Test: `2024-10-03` → `2026-10-30`, 5 seeds)
- **Chiến lược dự báo:** *Holdout-trained Rolling One-step-ahead Forecasting* ($h=1$). Trọng số và bộ chuẩn hóa (`StandardScaler`) được khớp cố định trên Train (`2013-01-02` → `2022-09-06`, 2.525 ngày) và chọn điểm dừng trên Validation (`2022-09-07` → `2024-10-02`, 541 ngày). Trên Test, tại mỗi ngày $t$, mô hình dùng cửa sổ lịch sử thực tế $[t - \text{seq\_len}, \dots, t - 1]$ và neo trên giá thực tế $P_{t-1}$.
- **Kiểm định DM:** Dùng `Diebold-Mariano` với phương sai dài hạn **HAC / Newey-West** ($L = \lfloor 4(n/100)^{2/9} \rfloor = 5$ với $n=542, h=1$) và hiệu chỉnh **Harvey-Leybourne-Newbold (1997)** trên chuỗi $\bar{d}_t$ tổng hợp theo ngày (không `flatten()` 3 tỉnh) và riêng từng tỉnh.
- **Diễn giải DA của Naive:** Naive Random Walk ($\hat{P}_t = P_{t-1}$) dự báo $\hat{r}_t = 0$, nên công thức `sign(0) == sign(r_t)` trên các ngày có biến động ($r_t \ne 0$) cho `0.00%` theo định nghĩa toán học (không đưa ra dự báo chiều). Do đó bảng dưới ghi `N/A (ΔP̂=0)`.

| Mô hình | Đầu vào | Cấu hình | RMSE (VND) | MAPE (%) | $R^2_{\text{ret}}$ | DA (%) | DM-stat (HAC) vs Naive | DM $p_{\text{2-sided}}$ vs Naive |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Naive (RW)** | $\hat{P}_t = P_{t-1}$ | — | 1922.89 | 1.119% | -0.0013 | N/A ($\Delta \hat{P}=0$) | Benchmark | Benchmark |
| **LSTM-Level (Raw MLM)** | Giá thô $P_t$ (3 tỉnh) | `seq=10`, MinMax trên Train | 17370.43 | 13.708% | -86.2435 | 50.57% | -10.670 | 0.0000 (kém hơn) |
| **LSTM 1** | Giá quá khứ (3 tỉnh, `log-return`) | `seq=30`, 1L×64u, Adam `1e-3`, `drop=0.0`, batch `16` | 1925.40 ± 8.9 | 1.129% | -0.0061 | 52.43% | -0.143 | 0.6688 |
| **LSTM 2** | Giá + Thời tiết + Kinh tế (11 biến) | `seq=10`, 1L×64u, AdamW `1e-3`, `out_drop=0.1`, batch `16` | **1607.15 ± 6.9** | **0.966%** | **0.3008** | **76.32%** | **+3.542** | **0.0005** |
| **LSTM 3** | Giá + Toàn bộ ngoại sinh (20 biến) | `seq=10`, 1L×64u, AdamW `1e-3`, `in/out_drop=0.2`, batch `16` | 1666.97 ± 10.8 | 0.982% | 0.2528 | 74.78% | +3.165 | 0.0021 |

## 4. Lưu ý phương pháp luận đã kiểm chứng
1. **ADF & VIF:** Các kiểm định ADF/VIF chỉ chạy trên Train để kiểm chứng bộ biến đã thiết kế sẵn (loại `temp_max`, `temp_min` có VIF 82.9/46.1; sai phân `soil_moisture_top` có Train ADF $p=0.0614$). Riêng `cos_day_of_year` có Train VIF sau sàng lọc là `10.05` (> 10 do tương quan với `is_harvest_season_vn` và `is_harvest_season_brazil`), nhưng vẫn được giữ trong `LSTM 3` và được gắn cờ minh bạch trong `lstm_diagnostics_vif.csv`.
2. **Đóng góp thực tế của từng nhóm biến (Group Permutation & Ablation Study):** Động lực cải thiện dự báo đến gần như hoàn toàn từ nhóm kỳ hạn cà phê (`ret_london`, `ret_arabica`; `Target + Core_Econ` đạt Test RMSE `1579.54 VND`, `Target + London_Only` đạt `1579.70 VND`). Nhóm thời tiết ngày và biến ngoại sinh mở rộng trong `LSTM 3` không cải thiện hiệu suất ngoài mẫu trên Test so với mô hình chỉ dùng giá + kinh tế cốt lõi.
3. **Khả dụng ngoại sinh tại thời điểm dự báo:** Các biến thị trường ngày $t-1$ (London/NY) chỉ khả dụng khi gốc dự báo đặt vào sáng ngày $t$ (giờ Việt Nam, sau khi sàn London/NY đóng cửa); dữ liệu thời tiết vệ tinh NASA POWER thực tế có độ trễ công bố ~2–7 ngày nên cần thay bằng dự báo thời tiết hoặc độ trễ khả dụng khi triển khai thực tế.

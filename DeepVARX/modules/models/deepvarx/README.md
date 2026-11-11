# DeepVARX (TimeXer-DeepVARX) Dự Báo Xác Suất Đa Biến Giá Cà Phê Robusta Việt Nam

Thư mục độc lập chứa mã nguồn mô hình **DeepVARX thế hệ mới (`TimeXer-DeepVARX` + `TFT GRN` + `Low-Rank Covariance` + `Causal Conformal Calibration`)**, dữ liệu đầu vào, bộ kiểm toán toàn diện các vấn đề độ tin cậy (Mục 5 & Mục 6 trước khi chốt khóa luận), toàn bộ bảng CSV định lượng và 5 biểu đồ chẩn đoán cho 3 tỉnh **Đắk Lắk, Lâm Đồng, Gia Lai**.

---

## 1. Kiến Trúc & Thực Nghiệm Chính (`542` Ngày Test: `2024-10-03 -> 2026-10-30`)

Dữ liệu panel ngày gồm `3,608` ngày log-return (`Train = 2,525`, `Val = 541`, `Test = 542`, `seq_len = 10`, `5 seeds: 42, 123, 456, 789, 2024`). [`CoffeeDeepVARX`](file:///D:/TLCN/deepvarx/deepvarx_coffee_models.py) tách nhánh nội sinh (`LSTM(3,64)`) và nhánh ngoại sinh (nhúng độc lập từng biến → `MultiheadAttention(4 heads)` → `GatedResidualNetwork`), kết hợp nhánh hiệp phương sai hạng thấp `Sigma_t = diag(d_t) + V_t V_t^T` (`rank=2`, `min_std=0.20`) huấn luyện bằng $\beta$-NLL ($\beta=0.5$) và Early Stopping theo CRPS trên Validation.

| Mô hình | Biến | Seed RMSE (VND) | Ensemble RMSE | MAE (VND) | DA (%) | CRPS_Gauss | CRPS_Conformal | Cov95_Gauss | Cov95_Conformal | MPIW95_Conf | DM $p_{2s}$ vs Naive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Naive (RW Prob)` | 1 | 1922.89 | 1922.89 | 1201.63 | 0.00% | 0.00865 | — | 90.2% | — | — | Benchmark |
| `DeepVARX-Level` | 11 | 46873.87 | 46873.87 | 44672.79 | — | — | — | — | — | — | 1.0000 |
| `DeepVARX-Legacy` | 11 | 1842.18 ± 18.9 | 1678.85 | 1251.59 ± 19.8 | 65.49% | — | — | — | — | — | 0.0003 |
| **DeepVARX 1** | 3 | 1908.82 ± 1.8 | 1908.51 | 1196.91 ± 1.7 | 51.64% | 0.00857 | 0.00859 | 92.0% | 94.2% | 7146 | 0.1610 |
| **DeepVARX 2** | 11 | 1687.95 ± 27.8 | 1663.76 | **1046.67 ± 9.6** | **75.08%** | **0.00730** | **0.00710** | 89.4% | 94.5% | **5594** | **< 0.0001** |
| **DeepVARX 3** | 20 | **1686.88 ± 31.4** | **1662.61** | 1054.48 ± 5.9 | 74.68% | 0.00740 | 0.00718 | 87.7% | **94.7%** | 5727 | **< 0.0001** |

---

## 2. Xác Minh Toàn Bộ 8 Điểm Mục 6 & Mục 5.1–5.2

1. **Rolling Conformal Zero-Leakage (`deepvarx_conformal_leakage_and_window_audit.csv`):**
   - Kiểm thử đối kháng *Future-Poisoning* (gán `true_test_r[t:] = +500.0` tại các mốc Test và kiểm tra timestamp `max_hist_date < pred_date[t]`) xác nhận khoảng cách nhân quả đúng `1` ngày (`t-1`) và sai khác dự báo **`0.0e+00`** tuyệt đối.
   - Độ nhạy cửa sổ trượt $W \in \{60, 90, 120, 180\}$ ổn định (`DVX 3` `Cov95` đạt `95.5% / 94.5% / 94.7% / 95.6%`; Đắk Lắk `KS p = 0.388–0.743`), trong đó $W=120$ cho `MPIW95` sắc nét nhất (`5,594 VND` ở DVX 2, `5,727 VND` ở DVX 3).
2. **PIT KS & Tính độc lập chuỗi thời gian (`deepvarx_province_calibration.csv`, `fig4`):**
   - Tách kiểm định **riêng từng tỉnh ($N=542$)** thay vì gộp chéo `1,626` quan sát, kết hợp **Berkowitz (2001) AR(1) Likelihood-Ratio LR3** và **Ljung-Box Q(5)**.
   - Với `Gauss + Val tau`, cả 3 tỉnh bị bác bỏ ngay trên kiểm định KS đơn tỉnh (`p = 0.0002–0.0035`). Khi dùng `Causal Conformal (W=120)`, **cả 3 tỉnh đều đạt chuẩn đồng đều biên KS** (`DVX 2`: `p = 0.5601 / 0.8501 / 0.5969`; `DVX 3`: `p = 0.7432 / 0.8794 / 0.8684`). Tuy vậy, `Berkowitz AR(1)` và `LB_var_p` vẫn có ý nghĩa (`AR1_rho = -0.13..-0.16`, `LB_var_p <= 0.0002`), phản ánh **tự tương quan âm nhẹ bậc 1 và cụm phương sai đuôi còn sót trong chuỗi PIT thực nghiệm** — được báo cáo minh bạch để tránh giả định i.i.d. quá mức.
3. **DM Test Đa Biến & Tổng hợp 3 tỉnh (`deepvarx_dm_test_audit.csv`):**
   - Kiểm chứng đồng thời **3 cách tổng hợp/kiểm định trên `N=542` ngày, `h=1`, Newey-West HAC (`max_lag=5`)**: (a) Daily Mean across 3 provinces, (b) **Mariano-Preve Multivariate 3D Wald (`MDM_Wald_chi2(3)`, `MDM_p_F(3,539)`)** bảo toàn hiệp phương sai liên tỉnh mà không ép trọng số bằng nhau, và (c) từng tỉnh riêng lẻ.
   - `DVX 2/3` vượt `Naive` rất mạnh ở cả 3 phép đo (`DM_HAC >= 3.71, MDM p_F <= 0.0019, p_2s < 0.0001` trên Ret-MSE, Price-MSE, CRPS và từng tỉnh). So với `Legacy OLS Ensemble (1,678.85 VND)`, vượt trội có ý nghĩa tại Lâm Đồng (`p_2s = 0.0124`) và qua `Mariano-Preve MDM` đa biến (`p_F = 0.0068 / 0.0033`), dù `Daily Mean` chưa qua `0.05` (`p_2s = 0.2123 / 0.1836`).
4. **DM Test `V3 Dense-Concat No-Attn` vs `TimeXer` (`deepvarx_dm_test_audit.csv`):**
   - Dù `V3 Dense-Concat` (`1,642.99 VND`) có điểm RMSE thấp hơn `DVX 3 TimeXer` (`1,662.61 VND`) ~`19.6 VND`, kiểm định **DM HAC (`p_2s = 0.7535` Ret-MSE, `0.5322` Price-MSE, `0.2863` CRPS)** và **MDM đa biến (`p_F = 0.3206`)** lẫn cả 3 tỉnh riêng lẻ (`p_2s = 0.5755–0.9124`) đều cho thấy **không có sự khác biệt có ý nghĩa thống kê** về mất mát dự báo hàng ngày.
5. **Bootstrap Đa Tỉnh Đồng Bộ & Lý giải Khoảng CI rộng (`deepvarx_bootstrap_correlation_check.csv`, `deepvarx_bootstrap_structure_audit.csv`):**
   - Xác nhận Circular Moving-Block Bootstrap lấy mẫu **đồng bộ theo ngày `t` trên cả 3 tỉnh & biến ngoại sinh trễ**: sai lệch tương quan `Corr(DL, LD) = 0.8440` vs Boot `0.8473`, `Corr(DL, GL) = 0.8224` vs Boot `0.8256`, `Corr(|Err_DL|, |London_lag1|) = 0.3028` vs Boot `0.3040` (`< 0.004`).
   - Khoảng Marginal CI rộng (`[1,288, 2,092]` tại `B=10`) xuất phát từ biên độ biến động theo chế độ thị trường (`Naive` cũng dao động `[1,528, 2,371]`). Trên cùng khối ngày đồng bộ, **Paired Differential 95% CI ($\Delta\text{RMSE} = \text{RMSE}_{\text{Naive}} - \text{RMSE}_{\text{Model}}$)** dương chặt chẽ trên mọi `B = 5, 10, 20` (`DVX 2`: **`+259.63 VND` [`+148.39, +361.55`]**; `DVX 3`: **`+260.52 VND` [`+138.78, +371.68`]**).
6. **Calibration Riêng Từng Tỉnh (`deepvarx_province_calibration.csv`):**
   - Báo cáo chi tiết `Đắk Lắk`, `Lâm Đồng`, `Gia Lai` với `Causal Conformal (W=120)`: `DVX 2` đạt `Cov95 = 95.0% / 94.8% / 93.7%`, `DVX 3` đạt `Cov95 = 94.6% / 95.0% / 94.5%` và `Cov80` cân bằng quanh `82.3%–83.8%`.
7. **Chính sách về VIF (`deepvarx_exog_vif_summary.csv`):**
   - Không loại bỏ biến chỉ vì VIF cao (đặc biệt cặp Fourier mùa vụ `sin/cos_day_of_year` có `cos_day_of_year VIF = 9.995` do tương quan với chỉ báo vụ mùa `-0.8293`); VIF được dùng để giải thích hiện tượng chia sẻ phân bổ quan trọng giữa các biến tương quan cao.
8. **Controlled Ablation Công Bằng Siêu Tham Số & Ngân Sách (`deepvarx_feature_ablation.csv`, `fig5`):**
   - Track A đồng nhất 100% (`AdamW, wd=1e-3, dropout=0.15, in_dropout=0.10, max_epochs=50, patience=10`) cho 6 tập biến: `Endog 3f (1,916.52)`, `Weather Only 6f (1,926.19)` (không cải thiện so với `Naive`), `London Only 4f (1,653.42, Ens=1,643.01)`, `Core Econ 8f (1,682.73)`, `DVX 2 11f (1,687.95)`, `Full TimeXer 20f (1,680.93, Val_CRPS=0.51445)`.
   - Track B đồng nhất (`drop=0.20`) đối chiếu `TimeXer Full (78,892 params, 245.6s, Val_CRPS=0.51288)` với `Dense-Concat No-Attn (29,420 params, 116.7s, Val_CRPS=0.52073, Seed=1,648.67±11.7)`.

---

## 3. Chạy Tái Lập Toàn Bộ Kiểm Toán

```powershell
cd D:\TLCN\deepvarx
python -u run_deepvarx_experiments.py
```

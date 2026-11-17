# BÁO CÁO KHẢO SÁT & KIỂM TOÁN DỮ LIỆU DỰ BÁO
## (DATA AUDIT REPORT FOR DIEBOLD–MARIANO TEST SUITE)

**Dự án:** Nghiên cứu & Dự báo Giá Cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)  
**Phân hệ:** Đánh giá Thống kê Độc lập (Statistical Evaluation Module)  
**Thời điểm kiểm toán:** Tháng 11/2026  
**Trạng thái kiểm toán:** HOÀN THÀNH (Dữ liệu sẵn có ở mức tổng hợp; cần Adapter xuất chuỗi dự báo chi tiết theo thời điểm để thỏa mãn điều kiện DM Test)

---

## 1. Cấu trúc thư mục và dữ liệu thực tế của dự án

Khảo sát toàn bộ cây thư mục dự án bên ngoài môi trường ảo (`venv/` và `.git/`), hệ thống bao gồm:

```text
DeepVARX/
├── Home.py                             # Giao diện chính Streamlit
├── requirements.txt                    # Danh mục thư viện phụ thuộc
├── data/
│   ├── data_daily_panel_clean_v2.csv   # Dữ liệu bảng (Panel) giá & ngoại sinh 5 khu vực (18,045 dòng x 52 cột)
│   ├── data_daily_clean_v2.csv         # Dữ liệu chuỗi thời gian gộp/trung bình (3,609 dòng x 47 cột)
│   ├── coffee_price_cleaned.csv        # Dữ liệu TSV gốc (54,414 dòng)
│   ├── processed/                      # 3 gói dữ liệu cắt lát từ Dispatcher (A_base, B_diff, C_scaled)
│   └── raw/                            # Dữ liệu thô và file master xác thực
├── modules/
│   ├── data_utils.py, panel_data.py    # Tiền xử lý dữ liệu, chuẩn hóa, tách train/val/test
│   ├── dispatcher.py                   # Bộ định tuyến cắt gói A, B, C
│   ├── feature_selection.py            # Kiểm định đa cộng tuyến VIF, Granger, CCF
│   ├── model_specs.py                  # Đặc tả danh mục mô hình & cấu hình biến
│   ├── stat_tests.py                   # Thư viện kiểm định thống kê hiện có
│   └── models/                         # Triển khai mô hình & kịch bản thực nghiệm
│       ├── var/                        # Mô hình VAR (Vector Autoregression)
│       ├── varx/                       # Mô hình VARX / VARX-L
│       ├── lstm/                       # Mô hình LSTM (Univariate, Multivariate, Full Exog)
│       ├── deepvar/                    # Mô hình DeepVAR (Probabilistic Low-Rank)
│       └── deepvarx/                   # Mô hình DeepVARX (TimeXer Dual-Stream + Cross-Attn)
├── pages/                              # Các trang giao diện luồng nghiệp vụ Streamlit (1 -> 6)
└── utils/                              # Cốt lõi toán học và class triển khai 5 mô hình chuẩn
    ├── var_model.py, varx_model.py
    ├── lstm_model.py, deepvar_model.py, deepvarx_model.py
```

---

## 2. Kiểm toán chi tiết các tệp CSV hiện có

### 2.1. Nhóm tệp dữ liệu đầu vào (Input Datasets)
- **`data/data_daily_panel_clean_v2.csv`**:
  - **Kích thước:** 18,045 dòng × 52 cột.
  - **Phạm vi không gian:** 5 khu vực gồm Tây Nguyên (Trung bình), Đắk Lắk, Lâm Đồng, Gia Lai, Đắk Nông (mỗi tỉnh 3,609 quan sát hàng ngày từ `2013-01-01` đến `2026-10-30`).
  - **Các trường chính:** `date`, `province`, `price_vnd_kg`, `ret_price`, các trễ thời tiết (`temp_mean`, `precip_nasa`, `humidity_air`,...), các biến vĩ mô quốc tế (`london_price_usd_ton`, `arabica_price_usd_lb`, `brent_price_usd_barrel`, `dxy_index`, `usd_vnd_rate`, `usd_brl_rate`), lịch mùa vụ (`sin_day_of_year`, `cos_day_of_year`, `is_harvest_season_vn`, `is_harvest_season_brazil`).
  - **Đánh giá:** Đây là tập dữ liệu chân lý mặt đất (ground truth) chuẩn xác, có tính dừng đã kiểm chứng qua ADF/KPSS trên chuỗi log-return `ret_price`. Dữ liệu đầy đủ $y_{true}$ và có thể sinh ngay giá trị dự báo Naive Benchmark ($\hat{y}_{t}^{Naive} = y_{t-1}$).

- **`data/processed/train_test_A_base.csv`, `train_test_B_expanded_diff.csv`, `train_test_C_expanded_scaled.csv`**:
  - **Kích thước:** ~3,608 dòng × 16 cột.
  - **Mục đích:** Là các ma trận đặc trưng đầu vào phục vụ huấn luyện mô hình ở Trang 4/5. Các tệp này chứa cột phân vùng `split` (`train` hoặc `test`) và biến mục tiêu `y_level`, nhưng **không chứa giá trị dự báo của các mô hình**.

### 2.2. Nhóm tệp kết quả thực nghiệm trong `modules/models/`
Khảo sát toàn bộ 38 tệp CSV trong các thư mục con của `modules/models/`:

| Thư mục | Tệp CSV kết quả chính | Cấu trúc dữ liệu thực tế | Có chứa chuỗi dự báo theo ngày ($y_t, \hat{y}_t$)? | Đủ điều kiện làm đầu vào DM Test? |
| :--- | :--- | :--- | :---: | :---: |
| `modules/models/var/` | `var_model_comparison_results.csv` | Bảng tổng hợp 6 dòng × 11 cột (RMSE, MAE, MAPE, DA, DM vs Naive). | ❌ Không (Chỉ có metric gộp) | ❌ Không |
| | `var_province_results.csv` | Bảng tổng hợp 18 dòng × 7 cột (Metric phân rã theo 3 tỉnh). | ❌ Không | ❌ Không |
| `modules/models/varx/` | `varx_model_comparison_results.csv` | Bảng tổng hợp 7 dòng × 12 cột (RMSE, MAE, MAPE, DA, DM vs Naive). | ❌ Không (Chỉ có metric gộp) | ❌ Không |
| | `varx_province_results.csv` | Bảng tổng hợp 21 dòng × 7 cột (Metric phân rã theo 3 tỉnh). | ❌ Không | ❌ Không |
| `modules/models/lstm/` | `lstm_model_comparison_results.csv` | Bảng tổng hợp 5 dòng × 15 cột (RMSE, MAE, R², DA, DM vs Naive). | ❌ Không | ❌ Không |
| | `lstm_province_results.csv` | Bảng tổng hợp 12 dòng × 10 cột (Metric phân rã theo 3 tỉnh). | ❌ Không | ❌ Không |
| | `lstm_dm_tests.csv` | Bảng 20 dòng × 8 cột: Báo cáo kết quả kiểm định DM đã chạy trước đó. | ❌ Không (Chỉ lưu DM stat & p-val) | ❌ Không |
| `modules/models/deepvar/` | `deepvar_model_comparison_results.csv` | Bảng tổng hợp 6 dòng × 16 cột (RMSE, CRPS, NLL, Cov80/95, MPIW). | ❌ Không | ❌ Không |
| | `deepvar_province_results.csv` | Bảng tổng hợp 9 dòng × 8 cột (Metric phân rã theo 3 tỉnh). | ❌ Không | ❌ Không |
| `modules/models/deepvarx/` | `deepvarx_model_comparison_results.csv` | Bảng tổng hợp 6 dòng × 18 cột (RMSE, CRPS, Cov95, Winkler, DM). | ❌ Không | ❌ Không |
| | `deepvarx_province_results.csv` | Bảng tổng hợp 9 dòng × 9 cột (Metric phân rã theo 3 tỉnh). | ❌ Không | ❌ Không |
| | `deepvarx_dm_test_audit.csv` | Bảng 31 dòng × 12 cột: Thống kê DM_HAC(5), p-value, MDM Wald giữa các cặp. | ❌ Không (Chỉ lưu DM stat & p-val) | ❌ Không |

---

## 3. Những phát hiện cốt lõi (Key Findings) & Thách thức kỹ thuật

### Phát hiện 1: Không có tệp CSV lưu chuỗi dự báo điểm/phân phối theo thời điểm (Point-in-Time Predictions)
- **Thực trạng:** Khi chạy các kịch bản thực nghiệm trước đây (`run_var_experiments.py`, `run_lstm_experiments.py`, `run_deepvarx_experiments.py`), tác giả đã khởi tạo mảng dự báo `model_preds` hoặc `seed_pred_p_list` **hoàn toàn trong bộ nhớ RAM**, thực hiện tính toán loss differential $d_t$ và xuất biểu đồ hình ảnh (`figures_*/`), sau đó chỉ ghi ra đĩa các bảng **chỉ số tổng hợp (aggregated metrics)** như RMSE, MAE hoặc giá trị kiểm định DM đã tính sẵn.
- **Tại giao diện Streamlit (`pages/5...` và `pages/6...`):** Bảng `pred` (chứa `date`, `province`, `y_true`, `y_pred`, `y_prev`) chỉ tồn tại trong `st.session_state["results"]` thuộc bộ nhớ phiên duyệt web của người dùng và sẽ biến mất khi tắt ứng dụng, không hề được tự động lưu thành tệp CSV độc lập trên ổ đĩa.
- **Hệ quả đối với DM Test:** Theo nguyên lý toán học thống kê của Diebold & Mariano (1995), kiểm định DM bắt buộc phải nhận đầu vào là chuỗi sai số dự báo $e_{A,t}, e_{B,t}$ và chuỗi vi phân hàm mất mát theo từng bước thời gian:
  $$d_t = L(e_{A,t}) - L(e_{B,t}), \quad t = 1, \dots, T$$
  Tuyệt đối **không thể** thực hiện kiểm định DM mới dựa trên các bảng kết quả tổng hợp RMSE hay MAE có sẵn.

### Phát hiện 2: Tính đồng bộ và nhất quán hoàn hảo của quy hoạch thực nghiệm
Mặc dù chưa có tệp CSV lưu chuỗi dự báo $y_{pred}$, kiểm toán mã nguồn xác nhận một ưu điểm then chốt:
1. **Cùng tập dữ liệu gốc:** Toàn bộ 5 họ mô hình (VAR, VARX, LSTM, DeepVAR, DeepVARX) đều kế thừa chung hàm `load_and_prepare_coffee_data()` từ tệp chuẩn hóa `data/data_daily_panel_clean_v2.csv`.
2. **Cùng mốc phân chia Train / Validation / Test:**
   - Tổng quan sát chuỗi: $T = 3608$ ngày.
   - `train_end = 2525` (70% đầu: từ `2013-01-02` đến `2022-11-28`).
   - `val_end = 3066` (15% tiếp theo: từ `2022-11-29` đến `2024-10-14`).
   - `te_idx`: đúng $N = 542$ ngày Test (từ `2024-10-15` đến `2026-10-30`).
3. **Cùng 3 tỉnh mục tiêu:** `TARGET_PROVINCES = ["dak_lak", "lam_dong", "gia_lai"]`.
4. **Cùng chuỗi giá thực tế ($P_{true}$) và giá hôm trước ($P_{prev}$):**
   - Sự đồng nhất giữa các họ mô hình là $100.000000\%$ (đã kiểm chứng bằng `np.allclose`).
5. **Cùng bước dự báo (Forecast Horizon):** Toàn bộ thực nghiệm tập trung vào dự báo 1 bước tới kỳ rolling one-step-ahead ($h = 1$).
6. **Khả năng tái lập hoàn toàn (Full Deterministic Reproducibility):**
   - Chạy thử nghiệm độc lập trên mô hình VAR(5) và VARX 2 cho kết quả RMSE và MAE khớp chính xác đến 6 chữ số thập phân so với bảng báo cáo lưu trong `var_model_comparison_results.csv` và `varx_model_comparison_results.csv`!

---

## 4. Đánh giá tính đủ điều kiện để thực hiện DM Test

| Tiêu chí kiểm tra | Đánh giá thực tế | Kết luận điều kiện |
| :--- | :--- | :---: |
| **Khoảng thời gian Test** | Hoàn toàn trùng khớp giữa mọi mô hình: 542 ngày liên tục (10/2024 – 10/2026). | ✅ ĐẠT |
| **Tập ngày dự báo (Dates)** | Đồng bộ $100\%$ không bị lệch ngày hay khuyết thiếu. | ✅ ĐẠT |
| **Giá trị thực tế ($y_{true}$)** | Nhất quán tuyệt đối giữa các mô hình. | ✅ ĐẠT |
| **Phân chia theo tỉnh** | 3 tỉnh trọng điểm Tây Nguyên: Đắk Lắk, Lâm Đồng, Gia Lai. | ✅ ĐẠT |
| **Dự báo bước $h$** | Đồng nhất $h = 1$ (rolling one-step). | ✅ ĐẠT |
| **Tệp CSV chuỗi $y_{pred}$ sẵn có** | Chưa tồn tại tệp CSV lưu chi tiết $y_{pred}$ cho từng ngày. | ⚠️ CHƯA ĐỦ ĐIỀU KIỆN ĐẦU VÀO TRỰC TIẾP |

**KẾT LUẬN KIỂM TOÁN:**
Dữ liệu dự án **đủ điều kiện về mặt phương pháp luận, tính đồng bộ và chất lượng toán học**, nhưng **thiếu tệp vật lý lưu chuỗi quan sát chi tiết theo thời điểm**.

---

## 5. Đề xuất phương án chuẩn hóa và xử lý (Standardization & Action Plan)

Để phân hệ DM Test hoạt động độc lập, không xâm phạm code hiện tại, không sửa file gốc, giải pháp tối ưu được đề xuất gồm:

### 5.1. Định nghĩa Chuẩn dữ liệu đầu vào (Standardized Forecast Input Schema)
Phân hệ `statistical_evaluation/` quy định cấu trúc tệp CSV đầu vào chuẩn dạng bảng dài (Long-format) hoặc bảng rộng (Wide-format):

**Định dạng Long-Format chuẩn (`predictions_long.csv`):**
- `date` (hoặc `target_date`): Chuỗi định dạng `YYYY-MM-DD` (Khóa thời gian).
- `province`: Tên định danh tỉnh (`dak_lak`, `lam_dong`, `gia_lai`).
- `model`: Tên mô hình dự báo (`Naive`, `VAR(5)`, `VARX 2`, `LSTM 2`, `DeepVARX 2`,...).
- `horizon`: Bước dự báo ($h \ge 1$, mặc định $h = 1$).
- `y_true`: Giá trị thực tế thu mua cà phê (VNĐ/kg).
- `y_pred`: Giá trị dự báo của mô hình (VNĐ/kg).
- `y_prev` (tùy chọn): Giá trị phiên trước đó $y_{t-1}$ dùng cho Naive Benchmark.
- `pred_std` (tùy chọn): Độ lệch chuẩn dự báo dùng cho hàm mất mát xác suất (CRPS).

### 5.2. Xây dựng Bộ chuyển đổi & Xuất dữ liệu độc lập (Forecast Data Export Utility)
- Xây dựng một script độc lập trong `statistical_evaluation/` (ví dụ `export_model_forecasts.py` hoặc tích hợp vào runner).
- Tận dụng mã nguồn các mô hình thuần túy (`CoffeeVAR`, `CoffeeVARX`,...) và tệp `data/data_daily_panel_clean_v2.csv` để sinh chuỗi dự báo chuẩn xác cho tập Test (542 ngày × 3 tỉnh = 1,626 quan sát/mô hình).
- Lưu các tệp dự báo chuẩn hóa vào thư mục độc lập `statistical_evaluation/data/input/`.
- Đồng thời cung cấp cơ chế cho phép người dùng tự nạp bất kỳ tệp CSV dự báo tùy ý từ bên ngoài, hệ thống sẽ tự động xác thực (validate) tính hợp lệ trước khi thực hiện kiểm định DM.

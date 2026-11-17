# PHÂN HỆ KIỂM ĐỊNH THỐNG KÊ DIEBOLD–MARIANO (DM TEST SUITE)
## Đánh giá So sánh Hiệu suất Mô hình Dự báo Giá Cà phê Robusta Việt Nam

Phân hệ đánh giá thống kê độc lập (**Statistical Evaluation Module**) được thiết kế chuyên biệt để thực hiện kiểm định **Diebold–Mariano (DM Test)** kết hợp ước lượng phương sai dài hạn **Newey–West HAC**, hiệu chỉnh mẫu hữu hạn **Harvey–Leybourne–Newbold (HLN)**, và hiệu chỉnh kiểm định đa giả thuyết **Holm–Bonferroni**.

Phân hệ hoạt động **hoàn toàn độc lập** với luồng Train/Test và giao diện Streamlit hiện tại, không tác động hay chỉnh sửa mã nguồn gốc, đồng thời đảm bảo tính minh bạch, có thể truy xuất và tái lập $100\%$.

---

## 1. Cấu trúc thư mục phân hệ

```text
statistical_evaluation/
├── README.md                           # Hướng dẫn cài đặt, cấu hình và sử dụng chi tiết
├── DATA_AUDIT_REPORT.md                # Báo cáo khảo sát & kiểm toán dữ liệu dự báo gốc
├── DM_TEST_THEORY.md                   # Cơ sở lý thuyết toán học & phương pháp luận chuẩn luận văn
├── requirements.txt                    # Danh mục thư viện phụ thuộc (numpy, pandas, scipy)
├── run_dm_test.py                      # Chương trình thực thi CLI độc lập
│
├── config/
│   ├── __init__.py
│   └── settings.py                     # Cấu hình mặc định (alpha=0.05, loss=['mse','mae'],...)
│
├── data/
│   ├── input/                          # Chứa tệp CSV dự báo đầu vào (chuẩn hóa Long-format)
│   │   └── coffee_forecasts_test_542days.csv
│   ├── processed/                      # Dữ liệu tiền xử lý phục vụ kiểm định
│   └── output/                         # Kết quả kiểm định thực tế
│       ├── dm_summary_results.csv      # Bảng tổng hợp các phép kiểm định cặp
│       ├── dm_detailed_observations.csv# Chuỗi vi phân mất mát chi tiết từng thời điểm (16 MB)
│       └── DM_EVALUATION_REPORT.md     # Báo cáo tổng kết Markdown tự động
│
├── src/                                # Mã nguồn thuật toán thống kê
│   ├── __init__.py
│   ├── loss_functions.py               # Hàm mất mát (MSE, MAE, CRPS) & Loss Differential
│   ├── hac_estimator.py                # Ước lượng Newey-West HAC & Bartlett Kernel
│   ├── hln_correction.py               # Hiệu chỉnh mẫu nhỏ Harvey-Leybourne-Newbold (1997)
│   ├── holm_correction.py              # Hiệu chỉnh p-value Holm-Bonferroni (1979)
│   ├── dm_test.py                      # Thuật toán Diebold-Mariano lõi
│   ├── data_validator.py               # Xác thực tính toàn vẹn và tính nhất quán dữ liệu
│   ├── data_alignment.py               # Đồng bộ chuỗi dự báo theo ngày, tỉnh, horizon
│   ├── data_loader.py                  # Trình tải dữ liệu linh hoạt (Long, Wide, Thư mục)
│   ├── comparison.py                   # Bộ điều phối so sánh đa mô hình & cảnh báo Nested
│   ├── report_generator.py             # Xuất CSV và sinh báo cáo kết quả
│   └── coffee_data_adapter.py          # Trích xuất chuỗi dự báo chuẩn xác từ mô hình dự án
│
└── tests/                              # Bộ kiểm thử tự động toàn diện (23 test cases)
    ├── conftest.py
    ├── test_loss.py                    # Kiểm thử hàm mất mát
    ├── test_hac.py                     # Kiểm thử HAC & độ rộng trễ
    ├── test_hln.py                     # Kiểm thử hiệu chỉnh HLN & phân phối Student-t
    ├── test_holm.py                    # Kiểm thử thuật toán Holm-Bonferroni
    ├── test_dm_test.py                 # Kiểm thử DM lõi & tính toán thủ công
    ├── test_alignment.py               # Kiểm thử đồng bộ & làm sạch dữ liệu
    └── test_integration.py             # Kiểm thử tích hợp toàn diện trên dữ liệu thực
```

---

## 2. Cài đặt môi trường

Phân hệ sử dụng các thư viện tính toán khoa học chuẩn mực:
```bash
pip install -r statistical_evaluation/requirements.txt
```
*(Yêu cầu tối thiểu: `numpy>=1.23.0`, `pandas>=1.5.0`, `scipy>=1.9.0`)*.

---

## 3. Hướng dẫn sử dụng chương trình chạy độc lập (`run_dm_test.py`)

### 3.1. Chạy nhanh mặc định (Đầy đủ 8 mô hình trên 3 tỉnh Tây Nguyên)
```bash
python statistical_evaluation/run_dm_test.py
```
Lệnh này sẽ:
1. Đọc dữ liệu dự báo 542 ngày test tại `statistical_evaluation/data/input/coffee_forecasts_test_542days.csv`.
2. Kiểm tra tính hợp lệ và đồng bộ ground truth giữa các mô hình.
3. Tự động sinh $28$ cặp so sánh $\times$ $3$ tỉnh $\times$ $2$ hàm mất mát (MSE, MAE) = **$168$ phép kiểm định**.
4. Áp dụng Newey-West HAC (lag=5) và hiệu chỉnh mẫu hữu hạn HLN (Student-t df=541).
5. Áp dụng hiệu chỉnh Holm-Bonferroni theo từng nhóm kiểm định.
6. Xuất bảng chi tiết từng thời điểm `dm_detailed_observations.csv` và bảng tổng hợp `dm_summary_results.csv`.
7. In báo cáo tóm tắt ra màn hình dòng lệnh.

### 3.2. So sánh các mô hình cụ thể
```bash
python statistical_evaluation/run_dm_test.py --models "Naive (Random Walk)" "VAR(5)" "VARX 2"
```

### 3.3. Lựa chọn tỉnh, hàm mất mát và tham số thống kê
```bash
python statistical_evaluation/run_dm_test.py --provinces dak_lak lam_dong --loss mse mae --alpha 0.05 --hac-lag 5
```

### 3.4. Chỉ định tệp dự báo tùy ý từ bên ngoài
```bash
python statistical_evaluation/run_dm_test.py --input-file duong/dan/toi/file_du_bao.csv --output-dir ket_qua/
```

### 3.5. Trích xuất lại dữ liệu dự báo chuẩn từ mô hình cà phê
```bash
python statistical_evaluation/run_dm_test.py --generate-coffee-data
```

---

## 4. Đặc tả chuẩn dữ liệu đầu vào (Input Schema)

Tệp CSV đầu vào có thể ở định dạng Long-format với các trường sau:

| Tên cột | Kiểu dữ liệu | Bắt buộc? | Ý nghĩa |
| :--- | :--- | :---: | :--- |
| `date` | String (`YYYY-MM-DD`) | **Có** | Ngày mốc mục tiêu dự báo |
| `province` | String | **Có** | Định danh tỉnh/khu vực (`dak_lak`, `lam_dong`, `gia_lai`) |
| `horizon` | Integer | Không (mặc định 1) | Bước dự báo ($h \ge 1$) |
| `model` | String | **Có** | Tên mô hình dự báo |
| `y_true` | Float | **Có** | Mức giá cà phê thực tế (VNĐ/kg) |
| `y_pred` | Float | **Có** | Mức giá dự báo của mô hình (VNĐ/kg) |
| `pred_std` | Float | Tùy chọn | Độ lệch chuẩn dự báo (dùng khi kiểm định bằng CRPS) |

---

## 5. Đặc tả dữ liệu đầu ra (Output Schema)

### 5.1. Tệp CSV chi tiết theo thời điểm (`dm_detailed_observations.csv`)
Lưu toàn bộ chuỗi quan sát từng ngày làm cơ sở thực hiện kiểm định:
- `date`: Ngày quan sát.
- `province`: Tỉnh.
- `horizon`: Bước dự báo.
- `loss_function`: Hàm mất mát (`mse` hoặc `mae`).
- `model_a`: Tên Mô hình A.
- `model_b`: Tên Mô hình B.
- `actual_value`: Giá thực tế $y_t$.
- `pred_a`: Giá dự báo $\hat{y}_{A,t}$.
- `pred_b`: Giá dự báo $\hat{y}_{B,t}$.
- `error_a`: Sai số $e_{A,t} = y_t - \hat{y}_{A,t}$.
- `error_b`: Sai số $e_{B,t} = y_t - \hat{y}_{B,t}$.
- `loss_a`: Mất mát $L(e_{A,t})$.
- `loss_b`: Mất mát $L(e_{B,t})$.
- `loss_differential`: Vi phân mất mát $d_t = L(e_{A,t}) - L(e_{B,t})$.

### 5.2. Tệp CSV tổng hợp kiểm định (`dm_summary_results.csv`)
Lưu kết quả kiểm định thống kê cho từng cặp:
- `province`, `horizon`, `loss_function`.
- `model_a`, `model_b`: Cặp mô hình so sánh.
- `n_observations`: Số quan sát thời gian ($T = 542$).
- `mean_loss_a`, `mean_loss_b`: Mất mát trung bình của từng mô hình.
- `mean_loss_diff`: $\bar{d} = \overline{L(A)} - \overline{L(B)}$.
- `dm_stat`, `dm_p_value`: Thống kê DM tiệm cận và p-value chuẩn tắc.
- `hln_stat`, `hln_p_value`: Thống kê HLN hiệu chỉnh và p-value Student-$t$ ($T-1$).
- `holm_p_value`: p-value sau hiệu chỉnh Holm–Bonferroni.
- `significance_level`: Ngưỡng $\alpha$ (0.05).
- `is_statistically_significant`: `True` / `False`.
- `better_model_by_loss`: Mô hình có loss thấp hơn trên mẫu.
- `better_model_statistical`: Mô hình có ưu thế thống kê vượt trội (sau khi xét cả p-value).
- `final_interpretation`: Diễn giải kết luận khoa học bằng lời.
- `hac_lag`: Số trễ Newey-West sử dụng.
- `is_nested`, `nested_warning`: Cảnh báo đối với cặp mô hình lồng nhau.

---

## 6. Hướng dẫn chạy kiểm thử tự động (Unit & Integration Tests)

Chạy toàn bộ 23 ca kiểm thử bằng `pytest`:
```bash
python -m pytest statistical_evaluation/tests -v
```

Nội dung kiểm thử bao gồm:
1. **Sanity checks & Hand calculations:** Đối chiếu với ví dụ tính toán tay chuẩn xác.
2. **Identical models:** Hai mô hình dự báo giống hệt nhau $\implies DM = 0, p = 1.0$.
3. **Model A better than B:** $\bar{d} < 0 \implies DM < 0, p < 0.001$.
4. **Model B better than A:** $\bar{d} > 0 \implies DM > 0, p < 0.001$.
5. **Symmetry & Signs:** Đổi chỗ Mô hình A và B $\implies$ đảo dấu chính xác $DM_{AB} = -DM_{BA}$.
6. **Newey-West HAC:** Trọng số Bartlett, chọn lag tự động, tính bán xác định dương.
7. **HLN correction:** Hệ số hiệu chỉnh $\sqrt{(T-1)/T}$ và phân phối Student-$t(T-1)$.
8. **Holm-Bonferroni step-down:** Kiểm tra tính đơn điệu không giảm và kiểm soát FWER.
9. **Data alignment & validation:** Xử lý lệch ngày, dữ liệu khuyết và trùng lặp.
10. **Full Pipeline Integration:** Kiểm thử đầu cuối với dữ liệu thực 542 ngày test.

---

## 7. Tài liệu lý thuyết chi tiết

Xem tài liệu lý thuyết toán học chuyên sâu trình bày bằng tiếng Việt tại:  
👉 [`statistical_evaluation/DM_TEST_THEORY.md`](DM_TEST_THEORY.md)

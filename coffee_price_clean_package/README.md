# GÓI DỮ LIỆU GIÁ CÀ PHÊ VIỆT NAM (2010 - 2026) VÀ MÃ NGUỒN XỬ LÝ LỖI
## VIETNAM COFFEE PRICE DATASET & PROCESSING PIPELINE

## Cấu trúc thư mục (Directory Structure)

```
coffee_price_clean_package/
├── data/
│   ├── coffee_price_cleaned.csv          # Tệp dữ liệu chính đã clean 100% (12.3 MB, 54.414 dòng)
│   ├── coffee_price_cleaned_backup.csv   # Tệp dữ liệu gốc chưa xử lý (dùng để so sánh đối chứng)
│   └── macro_yahoo_cache.csv             # Dữ liệu vĩ mô chốt phiên thực tế từ Yahoo Finance
├── code/
│   ├── crawlers/                         # Module cào dữ liệu đa nguồn
│   │   ├── __init__.py
│   │   ├── giacaphe_crawler.py           # Giải mã CSS ::after chống cào của GiaCaPhe.com
│   │   ├── giacafe_crawler.py            # Cào bảng giá liên tỉnh từ GiaCafe.vn
│   │   ├── vietnambiz_crawler.py         # Bóc tách tin tức thị trường VietnamBiz
│   │   ├── yahoo_crawler.py              # Lấy dữ liệu vĩ mô quốc tế (Arabica, Brent, DXY, FX)
│   │   └── multi_crawler.py              # Bộ điều phối cào đa luồng
│   ├── main.py                           # Điểm khởi chạy pipeline dòng lệnh CLI
│   ├── scanner.py                        # Quét và phát hiện 7 nhóm lỗi dữ liệu
│   ├── patcher.py                        # Khử gai nhọn cuối tuần & đồng bộ giá liên tỉnh
│   ├── fix_all_indicators.py             # Chuẩn hóa toàn bộ 37 chỉ số vĩ mô & địa lý
│   ├── test_against_reality.py           # Bộ 26 bài test đối soát thực tế lịch sử (100% ĐẠT)
│   └── generate_report_charts.py         # Vẽ biểu đồ kiểm định chất lượng 300 DPI
├── requirements.txt                      # Danh sách thư viện Python cần thiết
└── README.md                             # Tài liệu hướng dẫn sử dụng trọn gói
```

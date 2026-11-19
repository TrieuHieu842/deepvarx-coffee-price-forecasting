#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH THỰC THI KIỂM ĐỊNH DIEBOLD–MARIANO ĐỘC LẬP (CLI RUNNER)
Phân hệ đánh giá thống kê độc lập cho dự án dự báo giá cà phê.

Usage examples:
    python run_dm_test.py
    python run_dm_test.py --models "Naive (Random Walk)" "VAR(5)" "VARX 2" --loss mse mae
    python run_dm_test.py --provinces dak_lak lam_dong --alpha 0.05 --hac-lag 5
    python run_dm_test.py --generate-coffee-data
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Configure utf-8 stdout/stderr for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add current module to path
_cur_dir = Path(__file__).resolve().parent
if str(_cur_dir) not in sys.path:
    sys.path.insert(0, str(_cur_dir))

from src.comparison import run_model_comparisons
from src.data_loader import load_forecast_csv, load_from_directory
from src.data_validator import validate_forecast_dataframe
from src.report_generator import export_evaluation_results, generate_model_discovery_report
from src.coffee_data_adapter import generate_coffee_test_forecasts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DMRunner")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chương trình kiểm định thống kê Diebold-Mariano độc lập cho dự báo giá cà phê."
    )
    all_models_p = _cur_dir / "data" / "input" / "all_models_forecasts_test_542days.csv"
    coffee_p = _cur_dir / "data" / "input" / "coffee_forecasts_test_542days.csv"
    default_input = str(all_models_p if all_models_p.exists() else coffee_p)

    parser.add_argument(
        "--input-file",
        type=str,
        default=default_input,
        help="Đường dẫn tới tệp CSV dự báo đầu vào.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Thư mục chứa các tệp CSV dự báo (nếu tải nhiều file).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_cur_dir / "data" / "output"),
        help="Thư mục xuất kết quả CSV và báo cáo.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Danh sách các mô hình cần so sánh (mặc định: toàn bộ mô hình trong dữ liệu).",
    )
    parser.add_argument(
        "--provinces",
        nargs="+",
        default=None,
        help="Danh sách tỉnh cần kiểm định (mặc định: toàn bộ tỉnh trong dữ liệu).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=None,
        help="Danh sách bước dự báo horizon h (mặc định: toàn bộ horizon trong dữ liệu).",
    )
    parser.add_argument(
        "--loss",
        nargs="+",
        default=["mse", "mae", "crps"],
        help="Danh sách các hàm mất mát: 'mse', 'mae', 'crps' (mặc định: mse mae crps).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Mức ý nghĩa thống kê (mặc định: 0.05).",
    )
    parser.add_argument(
        "--hac-lag",
        type=int,
        default=None,
        help="Độ rộng trễ Newey-West HAC (mặc định: tự động chọn theo mẫu và horizon).",
    )
    parser.add_argument(
        "--no-holm",
        action="store_true",
        help="Tắt hiệu chỉnh Holm-Bonferroni cho kiểm định đa giả thuyết.",
    )
    parser.add_argument(
        "--no-detailed",
        action="store_true",
        help="Không xuất tệp CSV chi tiết từng thời điểm.",
    )
    parser.add_argument(
        "--generate-coffee-data",
        action="store_true",
        help="Tự động trích xuất lại dữ liệu dự báo test của các mô hình cà phê vào data/input/.",
    )
    return parser.parse_args()


def main() -> int:
    t0 = time.time()
    args = parse_arguments()

    print("=" * 88)
    print("PHÂN HỆ KIỂM ĐỊNH THỐNG KÊ DIEBOLD–MARIANO (DM TEST SUITE)")
    print("=" * 88)

    # 1. Option: Re-generate coffee forecasts if requested or if default file missing
    input_path = Path(args.input_file)
    if args.generate_coffee_data or (not input_path.exists() and args.data_dir is None):
        logger.info("Đang sinh tệp dự báo chuẩn hóa từ mô hình cà phê...")
        try:
            generate_coffee_test_forecasts(output_csv_path=str(input_path))
        except Exception as e:
            logger.error(f"Không thể sinh dữ liệu cà phê tự động: {e}")
            return 1

    # 2. Load forecast data
    if args.data_dir:
        logger.info(f"Đang nạp dữ liệu từ thư mục: {args.data_dir}")
        df_raw = load_from_directory(args.data_dir)
    else:
        logger.info(f"Đang nạp dữ liệu từ tệp: {input_path}")
        df_raw = load_forecast_csv(input_path)

    # 3. Validate forecast data
    val_report = validate_forecast_dataframe(df_raw)
    if not val_report.is_valid:
        logger.error("Dữ liệu đầu vào KHÔNG HỢP LỆ:")
        for err in val_report.errors:
            logger.error(f"  - {err}")
        return 1

    if val_report.warnings:
        for warn in val_report.warnings:
            logger.warning(f"  [Cảnh báo] {warn}")

    logger.info(f"Xác thực thành công {val_report.num_rows:,} dòng dữ liệu.")
    logger.info(f"Khoảng thời gian: {val_report.date_range[0]} đến {val_report.date_range[1]}")
    logger.info(f"Mô hình phát hiện: {val_report.models_found}")
    logger.info(f"Tỉnh phát hiện: {val_report.provinces_found}")
    logger.info(f"Horizon phát hiện: {val_report.horizons_found}")

    # Generate Model Discovery Report
    disc_report_path = generate_model_discovery_report(df_raw, args.output_dir)
    logger.info(f"Đã xuất báo cáo khảo sát mô hình: {disc_report_path}")

    # Filter selections
    selected_models = args.models or val_report.models_found
    selected_provs = args.provinces or val_report.provinces_found
    selected_horizons = args.horizons or val_report.horizons_found

    # Check validity of selections
    invalid_m = [m for m in selected_models if m not in val_report.models_found]
    if invalid_m:
        logger.error(f"Các mô hình yêu cầu không tồn tại trong dữ liệu: {invalid_m}")
        return 1

    invalid_p = [p for p in selected_provs if p not in val_report.provinces_found]
    if invalid_p:
        logger.error(f"Các tỉnh yêu cầu không tồn tại trong dữ liệu: {invalid_p}")
        return 1

    # 4. Execute DM Comparisons
    logger.info("\nĐang thực thi các phép kiểm định Diebold-Mariano...")
    results = run_model_comparisons(
        df=df_raw,
        models=selected_models,
        provinces=selected_provs,
        horizons=selected_horizons,
        loss_functions=args.loss,
        alpha=args.alpha,
        hac_lag=args.hac_lag,
        apply_holm=not args.no_holm,
        warn_nested=True,
    )

    if results.summary_df.empty:
        logger.warning("Không có phép kiểm định nào hoàn thành (do thiếu quan sát hoặc lọc rỗng).")
        return 0

    # 5. Export Output CSVs and Report
    logger.info(f"\nĐang xuất kết quả ra thư mục: {args.output_dir}")
    exported = export_evaluation_results(
        results,
        output_dir=args.output_dir,
        export_detailed=not args.no_detailed,
    )

    for k, p in exported.items():
        logger.info(f"  -> Đã lưu {k}: {p}")

    # 6. Print Console Summary
    print("\n" + "=" * 88)
    print("BẢNG TỔNG HỢP KẾT QUẢ KIỂM ĐỊNH DIEBOLD–MARIANO (MẪU ĐẠI DIỆN)")
    print("=" * 88)

    disp_cols = [
        "province",
        "model_a",
        "model_b",
        "loss_function",
        "mean_loss_diff",
        "hln_stat",
        "hln_p_value",
        "holm_p_value",
        "better_model_by_loss",
        "better_model_statistical",
    ]
    disp_cols = [c for c in disp_cols if c in results.summary_df.columns]
    sample_df = results.summary_df[disp_cols].copy()

    # Format numbers for console
    sample_df["mean_loss_diff"] = sample_df["mean_loss_diff"].map(lambda x: f"{x:+,.1f}")
    sample_df["hln_stat"] = sample_df["hln_stat"].map(lambda x: f"{x:+.3f}")
    sample_df["hln_p_value"] = sample_df["hln_p_value"].map(lambda x: f"{x:.4f}" if x >= 1e-4 else "< 0.0001")
    sample_df["holm_p_value"] = sample_df["holm_p_value"].map(lambda x: f"{x:.4f}" if x >= 1e-4 else "< 0.0001")

    print(sample_df.to_string(index=False))

    # Print summary statistics
    total_tests = len(results.summary_df)
    sig_tests = int(results.summary_df["is_statistically_significant"].sum())
    print("\n" + "-" * 88)
    print(f"Hoàn thành {total_tests} phép kiểm định cặp trong {time.time() - t0:.2f} giây.")
    print(f"Số cặp có sự khác biệt có ý nghĩa thống kê (alpha={args.alpha}): {sig_tests} / {total_tests} ({sig_tests/max(1, total_tests):.1%})")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    sys.exit(main())

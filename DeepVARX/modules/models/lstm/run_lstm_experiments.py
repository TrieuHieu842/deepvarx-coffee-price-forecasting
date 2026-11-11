# -*- coding: utf-8 -*-
"""
========================================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM TOÀN DIỆN CÁC PHIÊN BẢN LSTM (LSTM 1, LSTM 2, LSTM 3)
Dự báo giá thu mua cà phê Robusta tại Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
========================================================================================
Tệp đầu vào:
  - data_daily_panel_clean_v2.csv
  - benchmark_model_comparison_results.csv (tùy chọn, dùng đối chiếu với VAR/VARX/VARX-L)

Các phiên bản thực nghiệm:
  * Đối chứng 0a: Naive / Random Walk (P_hat_t = P_{t-1})
  * Đối chứng 0b: LSTM-Level (Raw MLM PyTorch trên mức giá gốc - minh họa bão hòa miền giá)
  * LSTM 1      : Giá cà phê quá khứ (Baseline)
  * LSTM 2      : Giá + Thời tiết cơ bản + Kinh tế cơ bản (Multivariate LSTM)
  * LSTM 3      : Giá + Toàn bộ biến ngoại sinh phù hợp (Đánh giá khả năng khai thác dữ liệu)
========================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_coffee_models import (
    TARGET_PROVINCES,
    compute_diagnostics,
    compute_group_permutation_importance,
    compute_metrics,
    compute_permutation_importance,
    diebold_mariano_hln,
    diebold_mariano_panel_hln,
    evaluate_raw_price_level_lstm,
    fit_single_lstm_seed,
    load_and_prepare_coffee_data,
    select_hac_lag,
)

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SEEDS: List[int] = [42, 123, 456, 789, 2024]
FIG_DIR = "figures_lstm"
os.makedirs(FIG_DIR, exist_ok=True)

# Cấu hình đã tuyển chọn nghiêm ngặt trên tập Validation (không chạm Test khi tuning)
MODEL_CONFIGS: Dict[str, Dict] = {
    "LSTM 1": {
        "purpose": "Baseline (Giá cà phê quá khứ)",
        "seq_len": 30,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "in_dropout": 0.0,
        "batch_size": 16,
        "lr": 1e-3,
        "optimizer_name": "adam",
        "weight_decay": 1e-4,
        "max_epochs": 50,
        "patience": 10,
        "scaler_type": "standard",
        "scale_target": True,
    },
    "LSTM 2": {
        "purpose": "Multivariate (Giá + Thời tiết + Kinh tế)",
        "seq_len": 10,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.1,
        "in_dropout": 0.0,
        "batch_size": 16,
        "lr": 1e-3,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
        "max_epochs": 50,
        "patience": 10,
        "scaler_type": "standard",
        "scale_target": True,
    },
    "LSTM 3": {
        "purpose": "Full Exogenous (Giá + Toàn bộ biến ngoại sinh phù hợp)",
        "seq_len": 10,
        "hidden_dim": 64,
        "num_layers": 1,
        "dropout": 0.2,
        "in_dropout": 0.2,
        "batch_size": 16,
        "lr": 1e-3,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
        "max_epochs": 50,
        "patience": 10,
        "scaler_type": "standard",
        "scale_target": True,
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def main() -> None:
    t_start = time.time()
    bundle = load_and_prepare_coffee_data("data_daily_panel_clean_v2.csv")

    print_header("BƯỚC 1: CHẨN ĐOÁN DỮ LIỆU, TÍNH DỪNG (ADF), ĐA CỘNG TUYẾN (VIF) & ĐỘ TRỄ T+1")
    diag = compute_diagnostics(bundle)

    print("\n--- 1.1. Phân bổ miền giá theo 3 tập Train / Validation / Test ---")
    print(diag["split_stats"].to_string(index=False))

    print("\n--- 1.2. Kiểm định tính dừng ADF trên tập Train ---")
    print(diag["adf"].to_string(index=False))

    print("\n--- 1.3. Kiểm định đa cộng tuyến VIF trước và sau sàng lọc biến ngoại sinh ---")
    print(diag["vif"].to_string(index=False))
    diag["vif"].to_csv("lstm_diagnostics_vif.csv", index=False)
    diag["adf"].to_csv("lstm_diagnostics_adf.csv", index=False)

    print("\n--- 1.4. Tương quan theo độ trễ (t+1, t, t-1, t-2) với ret_price (Đắk Lắk, Train) ---")
    print(diag["lead_lag_corr"].sort_values("Corr_Lag1_t_minus_1", key=abs, ascending=False).head(12).to_string(index=False))

    print("\n--- 1.5. Xác minh chiến lược đánh giá & nguồn rò rỉ tiềm ẩn ---")
    n_tr, n_va, n_te = bundle.train_end, bundle.val_end - bundle.train_end, len(bundle.ret_df) - bundle.val_end
    print(
        f"  * Phân chia thời gian (không chạm Val/Test khi fit scaler hay chọn biến):\n"
        f"    - Train: {n_tr} ngày ({bundle.dates[0].date()} -> {bundle.dates[bundle.train_end - 1].date()})\n"
        f"    - Val  : {n_va} ngày ({bundle.dates[bundle.train_end].date()} -> {bundle.dates[bundle.val_end - 1].date()})\n"
        f"    - Test : {n_te} ngày ({bundle.dates[bundle.val_end].date()} -> {bundle.dates[-1].date()})\n"
        f"  * Chiến lược dự báo: Holdout-trained Rolling One-step-ahead (h=1).\n"
        f"    Mô hình huấn luyện cố định trên Train/Val, tại mỗi ngày Test t dùng lịch sử thực tế\n"
        f"    [t - seq_len, ..., t - 1] và neo trên P_{{t-1}} thực tế (KHÔNG phải Fixed-origin Multi-step).\n"
        f"  * Lưu ý khả dụng ngoại sinh tại thời điểm dự báo: dữ liệu thị trường t-1 (London/NY)\n"
        f"    chỉ khả dụng nếu dự báo lập vào sáng ngày t (ICT); biến thời tiết vệ tinh NASA POWER\n"
        f"    thực tế có độ trễ công bố 2-7 ngày (cần thay bằng dự báo thời tiết/trễ phù hợp khi triển khai)."
    )

    print("\n--- 1.6. Cấu hình đầu vào và siêu tham số chốt từ Validation cho LSTM 1 / 2 / 3 ---")
    for name, cfg in MODEL_CONFIGS.items():
        cols = list(bundle.feature_sets[name].columns)
        print(
            f"  * {name} ({len(cols)} biến) | {cfg['purpose']}\n"
            f"    seq_len={cfg['seq_len']}, layers={cfg['num_layers']}, units={cfg['hidden_dim']}, "
            f"opt={cfg['optimizer_name'].upper()}(lr={cfg['lr']}, wd={cfg['weight_decay']}), "
            f"batch={cfg['batch_size']}, in_drop={cfg['in_dropout']}, out_drop={cfg['dropout']}, "
            f"epochs<={cfg['max_epochs']} (ES patience={cfg['patience']})"
        )
        print(f"    Features: {cols}")

    # ==================================================================================
    # BƯỚC 2: ĐÁNH GIÁ ĐỐI CHỨNG TRÊN TẬP TEST (542 NGÀY: 2024-10-03 -> 2026-10-30)
    # ==================================================================================
    print_header("BƯỚC 2: ĐÁNH GIÁ OUT-OF-SAMPLE TRÊN TẬP TEST (542 NGÀY, 5 SEEDS ĐỘC LẬP)")

    val_end = bundle.val_end
    test_true_p = bundle.price_df.iloc[val_end:].values
    test_prev_p = bundle.price_df.shift(1).iloc[val_end:].values
    test_true_r = bundle.ret_df.iloc[val_end:].values
    test_dates = bundle.dates[val_end:]

    # 2.0a. Naive (Random Walk): P_hat_t = P_{t-1}, r_hat_t = 0
    naive_pred_r = np.zeros_like(test_true_r)
    naive_pred_p = test_prev_p.copy()
    naive_ret_err = test_true_r - naive_pred_r
    naive_price_err = test_true_p - naive_pred_p

    naive_metrics = compute_metrics(test_true_p, naive_pred_p, test_true_r, naive_pred_r)
    summary_rows = [
        {
            "Model": "Naive (Random Walk)",
            "Purpose": "P_hat_t = P_{t-1} (Chuẩn đối chứng)",
            "Num_Features": 1,
            "Seq_Len": 1,
            "RMSE (VND)": f"{naive_metrics['RMSE (VND)']:.2f}",
            "MAE (VND)": f"{naive_metrics['MAE (VND)']:.2f}",
            "MAPE (%)": f"{naive_metrics['MAPE (%)']:.3f}%",
            "R2_price": f"{naive_metrics['R2_price']:.4f}",
            "R2_ret": f"{naive_metrics['R2_ret']:.4f}",
            "DA (%)": "N/A (ΔP̂=0)",
            "RMSE_ret": f"{naive_metrics['RMSE_ret']:.5f}",
            "DM_stat vs Naive": "Benchmark",
            "DM_p_1s vs Naive": "Benchmark",
            "DM_p_2s vs Naive": "Benchmark",
            "Avg_Best_Epoch": "-",
            "_rmse_num": naive_metrics["RMSE (VND)"],
        }
    ]
    print(
        f"-> [Benchmark] Naive (RW)      : RMSE = {naive_metrics['RMSE (VND)']:.2f} VND | "
        f"MAE = {naive_metrics['MAE (VND)']:.2f} | MAPE = {naive_metrics['MAPE (%)']:.3f}% | "
        f"R2_price = {naive_metrics['R2_price']:.4f} | DA = N/A (ΔP̂=0; công thức sign(0)==±1 cho 0.00%)"
    )

    # 2.0b. LSTM-Level (Raw MLM PyTorch trên price_vnd_kg trực tiếp - seed=42)
    raw_lvl_res = evaluate_raw_price_level_lstm(
        bundle.price_df,
        train_end=bundle.train_end,
        val_end=bundle.val_end,
        seq_len=10,
        hidden_dim=64,
        batch_size=16,
        lr=1e-3,
        max_epochs=50,
        patience=10,
        scaler_type="minmax",
        seed=42,
    )
    raw_lvl_metrics = compute_metrics(
        raw_lvl_res["test_true_price"],
        raw_lvl_res["test_pred_price"],
        raw_lvl_res["test_true_ret"],
        raw_lvl_res["test_pred_ret"],
        bench_ret_err=naive_ret_err,
        bench_price_err=naive_price_err,
    )
    summary_rows.append(
        {
            "Model": "LSTM-Level (Raw MLM)",
            "Purpose": "Chẩn đoán: dự báo trực tiếp mức giá P_t (MinMax)",
            "Num_Features": 3,
            "Seq_Len": 10,
            "RMSE (VND)": f"{raw_lvl_metrics['RMSE (VND)']:.2f}",
            "MAE (VND)": f"{raw_lvl_metrics['MAE (VND)']:.2f}",
            "MAPE (%)": f"{raw_lvl_metrics['MAPE (%)']:.3f}%",
            "R2_price": f"{raw_lvl_metrics['R2_price']:.4f}",
            "R2_ret": f"{raw_lvl_metrics['R2_ret']:.4f}",
            "DA (%)": f"{raw_lvl_metrics['DA (%)']:.2f}%",
            "RMSE_ret": f"{raw_lvl_metrics['RMSE_ret']:.5f}",
            "DM_stat vs Naive": f"{raw_lvl_metrics['DM_stat_price']:+.3f}",
            "DM_p_1s vs Naive": f"{raw_lvl_metrics['DM_p_price']:.4f}",
            "DM_p_2s vs Naive": f"{raw_lvl_metrics['DM_p_price_2sided']:.4f}",
            "Avg_Best_Epoch": "-",
            "_rmse_num": raw_lvl_metrics["RMSE (VND)"],
        }
    )
    print(
        f"-> [Diagnostic] LSTM-Level Raw : RMSE = {raw_lvl_metrics['RMSE (VND)']:.2f} VND | "
        f"MAPE = {raw_lvl_metrics['MAPE (%)']:.3f}% | R2 = {raw_lvl_metrics['R2_price']:.4f} "
        f"| DM_stat(price)={raw_lvl_metrics['DM_stat_price']:+.2f} (Bão hòa ngưỡng Train 28k-50k!)"
    )

    # 2.1. Chạy LSTM 1, LSTM 2, LSTM 3 qua 5 seeds độc lập
    FEATURE_GROUPS = {
        "LSTM 2": {
            "Target_Returns": ["dak_lak", "lam_dong", "gia_lai"],
            "Weather_Core": ["temp_mean", "precip_nasa", "humidity_air"],
            "Econ_Coffee_Futures": ["ret_london", "ret_arabica"],
            "Econ_Macro_FX": ["ret_brent", "ret_dxy", "ret_usd_vnd"],
        },
        "LSTM 3": {
            "Target_Returns": ["dak_lak", "lam_dong", "gia_lai"],
            "Weather_All": ["temp_mean", "precip_nasa", "humidity_air", "d_soil_moisture", "solar_radiation", "wind_speed_2m"],
            "Econ_Coffee_Futures": ["ret_london", "ret_arabica"],
            "Econ_Macro_FX": ["ret_brent", "ret_dxy", "ret_usd_vnd", "ret_usd_brl"],
            "Calendar_Friction": ["sin_day_of_year", "cos_day_of_year", "is_harvest_season_vn", "is_harvest_season_brazil", "exog_stale_days"],
        },
    }

    model_outputs: Dict[str, Dict] = {}
    province_rows = []
    importance_frames = []
    group_importance_frames = []

    for model_name, cfg in MODEL_CONFIGS.items():
        X_df = bundle.feature_sets[model_name]
        seed_metrics_list = []
        seed_pred_r_list = []
        seed_pred_p_list = []
        seed_val_rmse_list = []
        seed_runs = []

        for seed in SEEDS:
            run = fit_single_lstm_seed(
                X_df=X_df,
                Y_df=bundle.ret_df,
                price_df=bundle.price_df,
                train_end=bundle.train_end,
                val_end=bundle.val_end,
                seq_len=cfg["seq_len"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                in_dropout=cfg["in_dropout"],
                batch_size=cfg["batch_size"],
                lr=cfg["lr"],
                optimizer_name=cfg["optimizer_name"],
                weight_decay=cfg["weight_decay"],
                max_epochs=cfg["max_epochs"],
                patience=cfg["patience"],
                scaler_type=cfg["scaler_type"],
                scale_target=cfg["scale_target"],
                seed=seed,
            )
            seed_runs.append(run)
            seed_pred_r_list.append(run["test_pred_ret"])
            seed_pred_p_list.append(run["test_pred_price"])

            val_m = compute_metrics(
                run["val_true_price"], run["val_pred_price"], run["val_true_ret"], run["val_pred_ret"]
            )
            seed_val_rmse_list.append(val_m["RMSE (VND)"])

            m = compute_metrics(
                run["test_true_price"],
                run["test_pred_price"],
                run["test_true_ret"],
                run["test_pred_ret"],
                bench_ret_err=naive_ret_err,
                bench_price_err=naive_price_err,
            )
            m["best_epoch"] = run["best_epoch"]
            seed_metrics_list.append(m)

        # Tổng hợp Mean ± Std qua 5 seeds
        mean_m = {k: float(np.mean([r[k] for r in seed_metrics_list])) for k in seed_metrics_list[0]}
        std_m = {k: float(np.std([r[k] for r in seed_metrics_list])) for k in seed_metrics_list[0]}

        # Ensemble dự báo trung bình 5 seeds
        ens_pred_r = np.mean(seed_pred_r_list, axis=0)
        ens_pred_p = test_prev_p * np.exp(ens_pred_r)
        ens_m = compute_metrics(
            test_true_p,
            ens_pred_p,
            test_true_r,
            ens_pred_r,
            bench_ret_err=naive_ret_err,
            bench_price_err=naive_price_err,
        )

        model_outputs[model_name] = {
            "runs": seed_runs,
            "mean_metrics": mean_m,
            "std_metrics": std_m,
            "ens_metrics": ens_m,
            "ens_pred_r": ens_pred_r,
            "ens_pred_p": ens_pred_p,
            "val_rmse_mean": float(np.mean(seed_val_rmse_list)),
            "val_rmse_std": float(np.std(seed_val_rmse_list)),
        }

        summary_rows.append(
            {
                "Model": model_name,
                "Purpose": cfg["purpose"],
                "Num_Features": X_df.shape[1],
                "Seq_Len": cfg["seq_len"],
                "RMSE (VND)": f"{mean_m['RMSE (VND)']:.2f} ± {std_m['RMSE (VND)']:.1f}",
                "MAE (VND)": f"{mean_m['MAE (VND)']:.2f} ± {std_m['MAE (VND)']:.1f}",
                "MAPE (%)": f"{mean_m['MAPE (%)']:.3f}% ± {std_m['MAPE (%)']:.2f}%",
                "R2_price": f"{mean_m['R2_price']:.4f}",
                "R2_ret": f"{mean_m['R2_ret']:.4f}",
                "DA (%)": f"{mean_m['DA (%)']:.2f}% ± {std_m['DA (%)']:.2f}%",
                "RMSE_ret": f"{mean_m['RMSE_ret']:.5f}",
                "DM_stat vs Naive": f"{mean_m['DM_stat_ret']:+.3f}",
                "DM_p_1s vs Naive": f"{mean_m['DM_p_ret']:.4f}",
                "DM_p_2s vs Naive": f"{mean_m['DM_p_ret_2sided']:.4f}",
                "Avg_Best_Epoch": f"{mean_m['best_epoch']:.1f}",
                "_rmse_num": mean_m["RMSE (VND)"],
            }
        )

        print(
            f"-> {model_name:<12} ({X_df.shape[1]:2d}f, L={cfg['seq_len']:2d}): "
            f"Val RMSE={np.mean(seed_val_rmse_list):.2f} | "
            f"Test RMSE={mean_m['RMSE (VND)']:.2f} ± {std_m['RMSE (VND)']:.1f} | "
            f"MAPE={mean_m['MAPE (%)']:.3f}% | DA={mean_m['DA (%)']:.2f}% | "
            f"R2_ret={mean_m['R2_ret']:.4f} | DM(HAC)={mean_m['DM_stat_ret']:+.2f} "
            f"(p2s={mean_m['DM_p_ret_2sided']:.4f}) | Ens RMSE={ens_m['RMSE (VND)']:.2f}"
        )

        # Chi tiết từng tỉnh (kiểm định DM-HLN-HAC riêng từng tỉnh trên chuỗi 1D 542 ngày)
        for p_idx, prov in enumerate(TARGET_PROVINCES):
            p_rmses, p_maes, p_mapes, p_das, p_dmstats, p_dmps, p_dmp2s = [], [], [], [], [], [], []
            for run in seed_runs:
                pm = compute_metrics(
                    run["test_true_price"][:, p_idx : p_idx + 1],
                    run["test_pred_price"][:, p_idx : p_idx + 1],
                    run["test_true_ret"][:, p_idx : p_idx + 1],
                    run["test_pred_ret"][:, p_idx : p_idx + 1],
                    bench_ret_err=naive_ret_err[:, p_idx : p_idx + 1],
                )
                p_rmses.append(pm["RMSE (VND)"])
                p_maes.append(pm["MAE (VND)"])
                p_mapes.append(pm["MAPE (%)"])
                p_das.append(pm["DA (%)"])
                p_dmstats.append(pm["DM_stat_ret"])
                p_dmps.append(pm["DM_p_ret"])
                p_dmp2s.append(pm["DM_p_ret_2sided"])
            province_rows.append(
                {
                    "Model": model_name,
                    "Province": prov,
                    "RMSE_Mean": round(float(np.mean(p_rmses)), 2),
                    "RMSE_Std": round(float(np.std(p_rmses)), 2),
                    "MAE_Mean": round(float(np.mean(p_maes)), 2),
                    "MAPE_Mean (%)": round(float(np.mean(p_mapes)), 3),
                    "DA_Mean (%)": round(float(np.mean(p_das)), 2),
                    "DM_stat_vs_Naive": round(float(np.mean(p_dmstats)), 3),
                    "DM_p_1s_vs_Naive": round(float(np.mean(p_dmps)), 4),
                    "DM_p_2s_vs_Naive": round(float(np.mean(p_dmp2s)), 4),
                }
            )

        # Permutation Importance (đơn biến & theo nhóm) cho LSTM 2 và LSTM 3 (trung bình 5 seeds)
        if model_name in ("LSTM 2", "LSTM 3"):
            feat_names = list(X_df.columns)
            per_seed_imp = []
            per_seed_gimp = []
            for s_idx, run in enumerate(seed_runs):
                imp = compute_permutation_importance(
                    model=run["model"],
                    X_tensor=run["X_te_tensor"],
                    y_scaler=run["y_scaler"],
                    prev_prices=run["test_prev_price"],
                    true_prices=run["test_true_price"],
                    feature_names=feat_names,
                    n_repeats=10,
                    seed=42 + s_idx,
                ).set_index("Feature")
                per_seed_imp.append(imp[["RMSE_Increase_VND", "Relative_Share (%)"]])

                gimp = compute_group_permutation_importance(
                    model=run["model"],
                    X_tensor=run["X_te_tensor"],
                    y_scaler=run["y_scaler"],
                    prev_prices=run["test_prev_price"],
                    true_prices=run["test_true_price"],
                    feature_names=feat_names,
                    feature_groups=FEATURE_GROUPS[model_name],
                    n_repeats=10,
                    seed=42 + s_idx,
                ).set_index("Group")
                per_seed_gimp.append(gimp[["Num_Features", "RMSE_Increase_VND", "Relative_Share (%)"]])

            avg_imp = sum(per_seed_imp) / len(per_seed_imp)
            avg_imp = avg_imp.sort_values("RMSE_Increase_VND", ascending=False).reset_index()
            avg_imp.insert(0, "Model", model_name)
            importance_frames.append(avg_imp)

            avg_gimp = sum(per_seed_gimp) / len(per_seed_gimp)
            avg_gimp = avg_gimp.sort_values("RMSE_Increase_VND", ascending=False).reset_index()
            avg_gimp.insert(0, "Model", model_name)
            group_importance_frames.append(avg_gimp)

    # ==================================================================================
    # BƯỚC 3: KIỂM ĐỊNH CẶP DIEBOLD-MARIANO (HAC/NEWEY-WEST + PANEL), BẢNG & ABLATION STUDY
    # ==================================================================================
    hac_lag = select_hac_lag(len(test_dates), h=1)
    print_header(
        f"BƯỚC 3: KIỂM ĐỊNH DIEBOLD-MARIANO (HAC Newey-West L={hac_lag}, HLN) & ABLATION STUDY"
    )

    err_l1 = test_true_r - model_outputs["LSTM 1"]["ens_pred_r"]
    err_l2 = test_true_r - model_outputs["LSTM 2"]["ens_pred_r"]
    err_l3 = test_true_r - model_outputs["LSTM 3"]["ens_pred_r"]

    pair_checks = [
        ("LSTM 1 (ens) vs Naive (RW)", naive_ret_err, err_l1),
        ("LSTM 2 (ens) vs Naive (RW)", naive_ret_err, err_l2),
        ("LSTM 3 (ens) vs Naive (RW)", naive_ret_err, err_l3),
        ("LSTM 2 (ens) vs LSTM 1 (ens) [Đóng góp Thời tiết + KT cơ bản]", err_l1, err_l2),
        ("LSTM 3 (ens) vs LSTM 2 (ens) [Đóng góp Ngoại sinh mở rộng]", err_l2, err_l3),
    ]
    dm_rows = []
    for label, e_b, e_m in pair_checks:
        res_dm = diebold_mariano_panel_hln(
            e_b, e_m, h=1, max_lag=hac_lag, alternative="greater", province_names=TARGET_PROVINCES
        )
        st_agg = res_dm["agg_stat"]
        p1_agg = res_dm["agg_p"]
        p2_agg = res_dm["agg_p_two_sided"]
        if p2_agg < 0.05 and st_agg > 0:
            verdict = "Model tốt hơn Bench (p_2s < 0.05, DM > 0)"
        elif p2_agg < 0.05 and st_agg < 0:
            verdict = "Model KÉM hơn Bench (p_2s < 0.05, DM < 0)"
        else:
            verdict = "Không khác biệt có ý nghĩa (p_2s >= 0.05)"
        print(
            f"  * {label:<62}: Panel d_bar DM={st_agg:+6.3f}, p_1s={p1_agg:.4f}, p_2s={p2_agg:.4f} -> {verdict}"
        )
        dm_rows.append(
            {
                "Comparison": label,
                "Scope": "Panel_Daily_Agg (d_bar_t)",
                "N_Time": res_dm["n"],
                "HAC_Lag": res_dm["hac_lag"],
                "DM_stat_HLN": round(st_agg, 4),
                "p_one_sided_model_better": round(p1_agg, 6),
                "p_two_sided": round(p2_agg, 6),
                "Interpretation": verdict,
            }
        )
        for prov in TARGET_PROVINCES:
            pr = res_dm["per_province"][prov]
            if pr["p_two_sided"] < 0.05 and pr["dm_stat"] > 0:
                p_verdict = "Model tốt hơn Bench"
            elif pr["p_two_sided"] < 0.05 and pr["dm_stat"] < 0:
                p_verdict = "Model KÉM hơn Bench"
            else:
                p_verdict = "Không có ý nghĩa"
            print(
                f"      - {prov:<8}: DM={pr['dm_stat']:+6.3f}, p_1s={pr['p_value']:.4f}, "
                f"p_2s={pr['p_two_sided']:.4f} ({p_verdict})"
            )
            dm_rows.append(
                {
                    "Comparison": label,
                    "Scope": prov,
                    "N_Time": res_dm["n"],
                    "HAC_Lag": res_dm["hac_lag"],
                    "DM_stat_HLN": round(pr["dm_stat"], 4),
                    "p_one_sided_model_better": round(pr["p_value"], 6),
                    "p_two_sided": round(pr["p_two_sided"], 6),
                    "Interpretation": p_verdict,
                }
            )
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv("lstm_dm_tests.csv", index=False)

    print("\n--- Bảng tổng hợp các phiên bản LSTM (542 ngày Test) ---")
    summary_df = pd.DataFrame(summary_rows).drop(columns=["_rmse_num"])
    print(summary_df.to_string(index=False))
    summary_df.to_csv("lstm_model_comparison_results.csv", index=False)

    print("\n--- Bảng chi tiết theo từng tỉnh (Naive DA = NaN do r_hat = 0) ---")
    for p_idx, prov in enumerate(TARGET_PROVINCES):
        nm = compute_metrics(
            test_true_p[:, p_idx : p_idx + 1],
            naive_pred_p[:, p_idx : p_idx + 1],
            test_true_r[:, p_idx : p_idx + 1],
            naive_pred_r[:, p_idx : p_idx + 1],
        )
        province_rows.insert(
            p_idx,
            {
                "Model": "Naive (RW)",
                "Province": prov,
                "RMSE_Mean": round(nm["RMSE (VND)"], 2),
                "RMSE_Std": 0.0,
                "MAE_Mean": round(nm["MAE (VND)"], 2),
                "MAPE_Mean (%)": round(nm["MAPE (%)"], 3),
                "DA_Mean (%)": np.nan,
                "DM_stat_vs_Naive": np.nan,
                "DM_p_1s_vs_Naive": np.nan,
                "DM_p_2s_vs_Naive": np.nan,
            },
        )
    prov_df = pd.DataFrame(province_rows)
    print(prov_df.to_string(index=False))
    prov_df.to_csv("lstm_province_results.csv", index=False)

    imp_all_df = pd.concat(importance_frames, ignore_index=True)
    imp_all_df.to_csv("lstm_feature_importance.csv", index=False)
    print("\n--- Top 12 đặc trưng đơn biến của LSTM 3 (Permutation Importance - lưu ý tương quan chéo) ---")
    print(imp_all_df[imp_all_df["Model"] == "LSTM 3"].head(12).to_string(index=False))

    gimp_all_df = pd.concat(group_importance_frames, ignore_index=True)
    gimp_all_df.to_csv("lstm_group_importance.csv", index=False)
    print("\n--- Group Permutation Importance (LSTM 2 & LSTM 3, trung bình 5 seeds) ---")
    print(gimp_all_df.to_string(index=False))

    # ==================================================================================
    # BƯỚC 3B: ABLATION STUDY (CỐ ĐỊNH CẤU HÌNH CƠ BẢN LSTM 2: seq_len=10, 5 SEEDS)
    # ==================================================================================
    print("\n--- 3B. Ablation Study: Đóng góp thực tế của từng nhóm biến (seq_len=10, 5 seeds) ---")
    df_all_feats = bundle.feature_sets["LSTM 3"]
    base_cfg = MODEL_CONFIGS["LSTM 2"]
    ablation_specs = [
        ("A0: Target_Only (seq=10)", ["dak_lak", "lam_dong", "gia_lai"], base_cfg),
        ("A1: Target + Core_Weather", ["dak_lak", "lam_dong", "gia_lai", "temp_mean", "precip_nasa", "humidity_air"], base_cfg),
        ("A2: Target + London_Only", ["dak_lak", "lam_dong", "gia_lai", "ret_london"], base_cfg),
        ("A3: Target + Core_Econ", ["dak_lak", "lam_dong", "gia_lai", "ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd"], base_cfg),
        ("A4: LSTM 2 (Core Weather+Econ)", list(bundle.feature_sets["LSTM 2"].columns), base_cfg),
        ("A5: LSTM 2 + Ext_Weather", list(bundle.feature_sets["LSTM 2"].columns) + ["d_soil_moisture", "solar_radiation", "wind_speed_2m"], base_cfg),
        ("A6: LSTM 2 + Ext_Econ (USD/BRL)", list(bundle.feature_sets["LSTM 2"].columns) + ["ret_usd_brl"], base_cfg),
        ("A7: LSTM 2 + Calendar_Friction", list(bundle.feature_sets["LSTM 2"].columns) + ["sin_day_of_year", "cos_day_of_year", "is_harvest_season_vn", "is_harvest_season_brazil", "exog_stale_days"], base_cfg),
        ("A8: All_Exog (Base Drop=0.1)", list(df_all_feats.columns), base_cfg),
        ("A9: LSTM 3 (All_Exog, Drop=0.2)", list(df_all_feats.columns), MODEL_CONFIGS["LSTM 3"]),
    ]

    ablation_rows = []
    ablation_ens_errs = {}
    for spec_name, cols, cfg_use in ablation_specs:
        if spec_name.startswith("A4:"):
            runs_ab = model_outputs["LSTM 2"]["runs"]
            ens_r_ab = model_outputs["LSTM 2"]["ens_pred_r"]
            val_rmses = [compute_metrics(r["val_true_price"], r["val_pred_price"], r["val_true_ret"], r["val_pred_ret"])["RMSE (VND)"] for r in runs_ab]
        elif spec_name.startswith("A9:"):
            runs_ab = model_outputs["LSTM 3"]["runs"]
            ens_r_ab = model_outputs["LSTM 3"]["ens_pred_r"]
            val_rmses = [compute_metrics(r["val_true_price"], r["val_pred_price"], r["val_true_ret"], r["val_pred_ret"])["RMSE (VND)"] for r in runs_ab]
        else:
            X_sub = df_all_feats[cols]
            runs_ab = []
            val_rmses = []
            pred_r_list = []
            for seed in SEEDS:
                r = fit_single_lstm_seed(
                    X_df=X_sub,
                    Y_df=bundle.ret_df,
                    price_df=bundle.price_df,
                    train_end=bundle.train_end,
                    val_end=bundle.val_end,
                    seq_len=cfg_use["seq_len"],
                    hidden_dim=cfg_use["hidden_dim"],
                    num_layers=cfg_use["num_layers"],
                    dropout=cfg_use["dropout"],
                    in_dropout=cfg_use["in_dropout"],
                    batch_size=cfg_use["batch_size"],
                    lr=cfg_use["lr"],
                    optimizer_name=cfg_use["optimizer_name"],
                    weight_decay=cfg_use["weight_decay"],
                    max_epochs=cfg_use["max_epochs"],
                    patience=cfg_use["patience"],
                    scaler_type=cfg_use["scaler_type"],
                    scale_target=cfg_use["scale_target"],
                    seed=seed,
                )
                runs_ab.append(r)
                pred_r_list.append(r["test_pred_ret"])
                vm = compute_metrics(r["val_true_price"], r["val_pred_price"], r["val_true_ret"], r["val_pred_ret"])
                val_rmses.append(vm["RMSE (VND)"])
            ens_r_ab = np.mean(pred_r_list, axis=0)

        ab_ms = [
            compute_metrics(
                r["test_true_price"], r["test_pred_price"], r["test_true_ret"], r["test_pred_ret"],
                bench_ret_err=naive_ret_err, bench_price_err=naive_price_err,
            )
            for r in runs_ab
        ]
        ens_err_ab = test_true_r - ens_r_ab
        ablation_ens_errs[spec_name] = ens_err_ab
        ab_rmse_mean = float(np.mean([x["RMSE (VND)"] for x in ab_ms]))
        ab_rmse_std = float(np.std([x["RMSE (VND)"] for x in ab_ms]))
        print(f"  [Ablation] {spec_name:<34} ({len(cols):2d}f): Val={np.mean(val_rmses):.2f} | Test={ab_rmse_mean:.2f} ± {ab_rmse_std:.1f}")
        ablation_rows.append(
            {
                "Ablation_Config": spec_name,
                "Num_Features": len(cols),
                "Val_RMSE_Mean": round(float(np.mean(val_rmses)), 2),
                "Test_RMSE_Mean": round(ab_rmse_mean, 2),
                "Test_RMSE_Std": round(ab_rmse_std, 2),
                "Test_MAPE (%)": round(float(np.mean([x["MAPE (%)"] for x in ab_ms])), 3),
                "Test_DA (%)": round(float(np.mean([x["DA (%)"] for x in ab_ms])), 2),
                "Test_R2_ret": round(float(np.mean([x["R2_ret"] for x in ab_ms])), 4),
            }
        )

    base_a0_err = ablation_ens_errs["A0: Target_Only (seq=10)"]
    base_l2_err = ablation_ens_errs["A4: LSTM 2 (Core Weather+Econ)"]
    for row in ablation_rows:
        name = row["Ablation_Config"]
        err = ablation_ens_errs[name]
        if name == "A0: Target_Only (seq=10)":
            row["DM_stat_vs_A0"] = 0.0
            row["DM_p2s_vs_A0"] = 1.0
        else:
            dm_a0 = diebold_mariano_panel_hln(base_a0_err, err, h=1, max_lag=hac_lag, alternative="two-sided")
            row["DM_stat_vs_A0"] = round(dm_a0["agg_stat"], 3)
            row["DM_p2s_vs_A0"] = round(dm_a0["agg_p_two_sided"], 4)
        if name == "A4: LSTM 2 (Core Weather+Econ)":
            row["DM_stat_vs_LSTM2"] = 0.0
            row["DM_p2s_vs_LSTM2"] = 1.0
        else:
            dm_l2 = diebold_mariano_panel_hln(base_l2_err, err, h=1, max_lag=hac_lag, alternative="two-sided")
            row["DM_stat_vs_LSTM2"] = round(dm_l2["agg_stat"], 3)
            row["DM_p2s_vs_LSTM2"] = round(dm_l2["agg_p_two_sided"], 4)

    ablation_df = pd.DataFrame(ablation_rows)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv("lstm_ablation_results.csv", index=False)

    if os.path.exists("benchmark_model_comparison_results.csv"):
        print("\n--- Đối chiếu cùng bộ benchmark đã có trong hệ thống (cùng 542 ngày Test) ---")
        bench_df = pd.read_csv("benchmark_model_comparison_results.csv")
        print(bench_df.to_string(index=False))

    # ==================================================================================
    # BƯỚC 4: TRỰC QUAN HÓA KẾT QUẢ & LƯU BIỂU ĐỒ
    # ==================================================================================
    print_header("BƯỚC 4: XUẤT BIỂU ĐỒ TRỰC QUAN")

    # Fig 1: Dịch chuyển phân phối + sụp đổ của LSTM-Level + Zoom Test Đắk Lắk
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    ax1, ax2 = axes
    dl_prices = bundle.price_df["dak_lak"]
    tr_end = bundle.train_end
    ax1.plot(bundle.dates[:tr_end], dl_prices.iloc[:tr_end], color="#2b5c8f", lw=1.1, label="Train (28.9k-50.1k)")
    ax1.plot(bundle.dates[tr_end:val_end], dl_prices.iloc[tr_end:val_end], color="#e68a00", lw=1.1, label="Validation (39.4k-134k)")
    ax1.plot(test_dates, dl_prices.iloc[val_end:], color="#1a1a1a", lw=1.4, label="Test Actual (83.9k-139k)")
    ax1.plot(test_dates, raw_lvl_res["test_pred_price"][:, 0], color="#d62728", ls="--", lw=1.6, label="LSTM-Level Raw (Suy thoái)")
    ax1.plot(test_dates, model_outputs["LSTM 2"]["ens_pred_p"][:, 0], color="#2ca02c", lw=1.2, alpha=0.85, label="LSTM 2 (Log-Return + Naive Anchor)")
    ax1.set_title("Dịch chuyển phân phối & Suy thoái của LSTM mức giá thô (Đắk Lắk)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Giá cà phê (VND/kg)")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(alpha=0.25)

    # Zoom 60 ngày cuối trên Test
    zoom = 60
    zd = test_dates[-zoom:]
    ax2.plot(zd, test_true_p[-zoom:, 0], color="#111111", lw=2.0, label="Actual (Thực tế)")
    ax2.plot(zd, naive_pred_p[-zoom:, 0], color="#7f7f7f", ls=":", lw=1.5, label="Naive (P_{t-1})")
    ax2.plot(zd, model_outputs["LSTM 1"]["ens_pred_p"][-zoom:, 0], color="#ff7f0e", ls="--", lw=1.3, label="LSTM 1 (Baseline)")
    ax2.plot(zd, model_outputs["LSTM 2"]["ens_pred_p"][-zoom:, 0], color="#2ca02c", lw=1.6, label="LSTM 2 (Weather+Econ)")
    ax2.plot(zd, model_outputs["LSTM 3"]["ens_pred_p"][-zoom:, 0], color="#1f77b4", ls="-.", lw=1.5, label="LSTM 3 (Full Exog)")
    ax2.set_title("So sánh dự báo 60 ngày cuối trên Test (Đắk Lắk)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Giá cà phê (VND/kg)")
    ax2.tick_params(axis="x", rotation=25)
    ax2.legend(loc="best", fontsize=8.5)
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_price_regime_and_forecast.png")
    plt.savefig(p1, dpi=180)
    plt.close(fig)

    # Fig 2: So sánh RMSE & DA giữa Naive, LSTM 1, LSTM 2, LSTM 3
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = ["Naive (RW)", "LSTM 1\n(Giá QK)", "LSTM 2\n(Thời tiết+KT)", "LSTM 3\n(Toàn bộ NS)"]
    rmses = [
        naive_metrics["RMSE (VND)"],
        model_outputs["LSTM 1"]["mean_metrics"]["RMSE (VND)"],
        model_outputs["LSTM 2"]["mean_metrics"]["RMSE (VND)"],
        model_outputs["LSTM 3"]["mean_metrics"]["RMSE (VND)"],
    ]
    rmse_errs = [
        0.0,
        model_outputs["LSTM 1"]["std_metrics"]["RMSE (VND)"],
        model_outputs["LSTM 2"]["std_metrics"]["RMSE (VND)"],
        model_outputs["LSTM 3"]["std_metrics"]["RMSE (VND)"],
    ]
    das = [
        0.0,
        model_outputs["LSTM 1"]["mean_metrics"]["DA (%)"],
        model_outputs["LSTM 2"]["mean_metrics"]["DA (%)"],
        model_outputs["LSTM 3"]["mean_metrics"]["DA (%)"],
    ]
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#1f77b4"]

    b1 = axes[0].bar(labels, rmses, yerr=rmse_errs, capsize=5, color=colors, edgecolor="#222222", alpha=0.9)
    axes[0].axhline(1669.96, color="#9467bd", ls="--", lw=1.3, label="VARX-L Benchmark (1669.96)")
    axes[0].set_title("Test RMSE trên giá (VND/kg) - Thấp hơn là tốt hơn", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("RMSE (VND/kg)")
    axes[0].set_ylim(1200, max(rmses) + 220)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)
    for bar, val in zip(b1, rmses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 25, f"{val:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    b2 = axes[1].bar(labels[1:], das[1:], color=colors[1:], edgecolor="#222222", alpha=0.9)
    axes[1].axhline(50.0, color="#d62728", ls=":", lw=1.2, label="Ngẫu nhiên (50%)")
    axes[1].axhline(75.02, color="#9467bd", ls="--", lw=1.3, label="VARX-L DA (75.02%)")
    axes[1].set_title("Độ chính xác dự báo hướng (DA %)\n(Naive RW: r_hat=0 nên DA không xác định)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Directional Accuracy (%)")
    axes[1].set_ylim(0, 95)
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.25)
    for bar, val in zip(b2, das[1:]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_lstm_versions_comparison.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    # Fig 3: Learning curves & Top Feature Importance (LSTM 3)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for mname, col in [("LSTM 1", "#ff7f0e"), ("LSTM 2", "#2ca02c"), ("LSTM 3", "#1f77b4")]:
        r0 = model_outputs[mname]["runs"][0]
        ep_axis = range(1, len(r0["val_losses"]) + 1)
        axes[0].plot(ep_axis, r0["train_losses"], color=col, ls="--", alpha=0.55, label=f"{mname} Train")
        axes[0].plot(ep_axis, r0["val_losses"], color=col, lw=1.8, label=f"{mname} Val (best={r0['best_epoch']})")
    axes[0].set_title("Quá trình hội tụ & Early Stopping (Seed 42)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (Standardized Return)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.25)

    top_imp = imp_all_df[imp_all_df["Model"] == "LSTM 3"].head(12).iloc[::-1]
    axes[1].barh(top_imp["Feature"], top_imp["RMSE_Increase_VND"], color="#1f77b4", edgecolor="#222222", alpha=0.9)
    axes[1].axvline(0.0, color="#333333", lw=0.8)
    axes[1].set_title("Permutation Importance đơn biến trong LSTM 3\n(Cảnh báo: hoán vị đơn lẻ bị ảnh hưởng đa cộng tuyến)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Mức tăng Test RMSE khi hoán vị biến (VND/kg)")
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig3_learning_curves_and_importance.png")
    plt.savefig(p3, dpi=180)
    plt.close(fig)

    # Fig 4: Group Permutation Importance & Ablation Study
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    g_l3 = gimp_all_df[gimp_all_df["Model"] == "LSTM 3"].iloc[::-1]
    axes[0].barh(g_l3["Group"], g_l3["RMSE_Increase_VND"], color="#2ca02c", edgecolor="#222222", alpha=0.9)
    axes[0].axvline(0.0, color="#333333", lw=0.8)
    axes[0].set_title("Group Permutation Importance trên LSTM 3 (5 seeds)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mức tăng Test RMSE khi hoán vị cả nhóm biến (VND/kg)")
    axes[0].grid(axis="x", alpha=0.25)

    ab_plot = ablation_df.iloc[::-1]
    y_pos = np.arange(len(ab_plot))
    axes[1].barh(y_pos, ab_plot["Test_RMSE_Mean"], xerr=ab_plot["Test_RMSE_Std"], capsize=3, color="#1f77b4", edgecolor="#222222", alpha=0.88)
    axes[1].axvline(naive_metrics["RMSE (VND)"], color="#d62728", ls=":", lw=1.3, label=f"Naive RW ({naive_metrics['RMSE (VND)']:.1f})")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(ab_plot["Ablation_Config"], fontsize=8.5)
    axes[1].set_xlim(1450, max(ab_plot["Test_RMSE_Mean"]) + 90)
    axes[1].set_title("Ablation Study: Đóng góp thực tế của từng nhóm biến (seq=10, 5 seeds)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Test RMSE (VND/kg) - Thấp hơn là tốt hơn")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "fig4_ablation_and_group_importance.png")
    plt.savefig(p4, dpi=180)
    plt.close(fig)

    print(f"Đã lưu biểu đồ vào: {p1}, {p2}, {p3}, {p4}")
    print(f"Hoàn tất toàn bộ thực nghiệm sau {time.time() - t_start:.1f} giây.")


if __name__ == "__main__":
    main()

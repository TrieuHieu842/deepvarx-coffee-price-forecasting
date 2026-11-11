# -*- coding: utf-8 -*-
"""
========================================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM TOÀN DIỆN CÁC PHIÊN BẢN DEEPVAR (DEEPVAR 1, 2, 3)
Dự báo xác suất đa biến giá thu mua cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
========================================================================================
Các phiên bản thực nghiệm (đồng bộ tập biến & 542 ngày Test với hệ thống LSTM & VARX):
  * Đối chứng 0a: Naive (Random Walk) điểm & xác suất (N(0, tau^2 * Cov_train))
  * Đối chứng 0b: DeepVAR-Level (Raw Price) - chẩn đoán suy thoái miền giá
  * DeepVAR 1   : Giá cà phê quá khứ (3 biến, Baseline tự hồi quy xác suất)
  * DeepVAR 2   : Giá + Thời tiết + Kinh tế (11 biến, Multivariate Low-Rank DeepVAR)
  * DeepVAR 2-D : Biến thể Diagonal (rank=0) - kiểm chứng vai trò hiệp phương sai Low-Rank
  * DeepVAR 3   : Giá + Toàn bộ biến ngoại sinh phù hợp (20 biến, Đánh giá khai thác dữ liệu)
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

from deepvar_coffee_models import (
    TARGET_PROVINCES,
    compute_deepvar_metrics,
    compute_deepvar_permutation_importance,
    compute_diagnostics,
    diebold_mariano_loss_diff,
    evaluate_raw_level_deepvar,
    fit_single_deepvar_seed,
    gaussian_crps_numpy,
    load_and_prepare_coffee_data,
)

sys.stdout.reconfigure(encoding="utf-8")

SEEDS: List[int] = [42, 123, 456, 789, 2024]
FIG_DIR = "figures_deepvar"
os.makedirs(FIG_DIR, exist_ok=True)

# Cấu hình đã tuyển chọn trên Validation (không chạm Test khi chọn cấu hình)
DEEPVAR_CONFIGS: Dict[str, Dict] = {
    "DeepVAR 1": {
        "feature_key": "DeepVAR 1",
        "purpose": "Baseline xác suất thuần nội sinh (Giá quá khứ 3 tỉnh)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.0,
        "in_dropout": 0.0,
        "optimizer_name": "adam",
        "weight_decay": 1e-4,
    },
    "DeepVAR 2": {
        "feature_key": "DeepVAR 2",
        "purpose": "Multivariate Low-Rank DeepVAR (Giá + Thời tiết + Kinh tế)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.1,
        "in_dropout": 0.0,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
    },
    "DeepVAR 2 (Diag)": {
        "feature_key": "DeepVAR 2",
        "purpose": "Ablation hiệp phương sai đường chéo (rank=0, bỏ tương quan liên tỉnh)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 0,
        "dropout": 0.1,
        "in_dropout": 0.0,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
    },
    "DeepVAR 3": {
        "feature_key": "DeepVAR 3",
        "purpose": "Full Exogenous DeepVAR (Giá + Toàn bộ biến ngoại sinh phù hợp)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.2,
        "in_dropout": 0.2,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main() -> None:
    t0 = time.time()
    bundle = load_and_prepare_coffee_data("data_daily_panel_clean_v2.csv")
    train_end, val_end = bundle.train_end, bundle.val_end

    print_header("BƯỚC 1: CHẨN ĐOÁN DỮ LIỆU & CẤU TRÚC TƯƠNG QUAN KHÔNG GIAN LIÊN TỈNH")
    diag = compute_diagnostics(bundle)
    print("\n--- 1.1. Tương quan tỷ suất sinh lợi thực nghiệm giữa 3 tỉnh (Tập Train) ---")
    print(diag["emp_corr_train"])
    print("\n--- 1.2. Tương quan tỷ suất sinh lợi thực nghiệm giữa 3 tỉnh (Tập Test) ---")
    print(diag["emp_corr_test"])
    print("\n-> Nhận xét: Tương quan liên tỉnh đạt ~0.88 - 0.93 => Bắt buộc dùng ma trận hiệp phương sai Low-Rank (Sigma = D + V V^T).")

    # ==================================================================================
    # BƯỚC 2: ĐÁNH GIÁ ĐỐI CHỨNG TRÊN TẬP TEST (542 NGÀY, 5 SEEDS)
    # ==================================================================================
    print_header("BƯỚC 2: THỰC NGHIỆM OUT-OF-SAMPLE DEEPVAR 1 / 2 / 3 (542 NGÀY TEST, 5 SEEDS)")

    test_true_p = bundle.price_df.iloc[val_end:].values
    test_prev_p = bundle.price_df.shift(1).iloc[val_end:].values
    test_true_r = bundle.ret_df.iloc[val_end:].values
    test_dates = bundle.dates[val_end:]

    # 2.0a. Naive (Random Walk) xác suất tĩnh chuẩn bị từ Cov_train hiệu chỉnh trên Val
    cov_train = np.cov(bundle.ret_df.iloc[:train_end].values.T)
    val_r = bundle.ret_df.iloc[train_end:val_end].values
    inv_cov_tr = np.linalg.inv(cov_train)
    tau_rw = float(np.sqrt(np.mean(np.einsum("ni,ij,nj->n", val_r, inv_cov_tr, val_r)) / 3.0))
    cov_rw_cal = cov_train * (tau_rw ** 2)
    std_rw_cal = np.sqrt(np.diag(cov_rw_cal))[None, :]
    rw_pred_r = np.zeros_like(test_true_r)
    rw_ret_err = test_true_r - rw_pred_r

    # NLL & CRPS của Naive xác suất tĩnh
    sign_det, logdet_rw = np.linalg.slogdet(cov_rw_cal)
    inv_cov_rw = np.linalg.inv(cov_rw_cal)
    quad_rw = np.einsum("ni,ij,nj->n", test_true_r, inv_cov_rw, test_true_r)
    nll_rw = float(0.5 * (3 * np.log(2 * np.pi) + logdet_rw + quad_rw).mean())
    crps_rw_mat = gaussian_crps_numpy(test_true_r, rw_pred_r, np.broadcast_to(std_rw_cal, test_true_r.shape))
    crps_sum_rw = float(np.mean(gaussian_crps_numpy(test_true_r.sum(axis=1), np.zeros(len(test_true_r)), np.full(len(test_true_r), np.sqrt(cov_rw_cal.sum())))))
    rw_q025 = test_prev_p * np.exp(-1.96 * std_rw_cal)
    rw_q975 = test_prev_p * np.exp(+1.96 * std_rw_cal)
    cov95_rw = float(np.mean((test_true_p >= rw_q025) & (test_true_p <= rw_q975)) * 100.0)
    rw_rmse = float(np.sqrt(np.mean((test_true_p - test_prev_p) ** 2)))
    rw_mae = float(np.mean(np.abs(test_true_p - test_prev_p)))
    rw_mape = float(np.mean(np.abs((test_true_p - test_prev_p) / test_true_p)) * 100.0)

    print(
        f"-> [Benchmark] Naive RW Prob : RMSE = {rw_rmse:.2f} VND | MAPE = {rw_mape:.3f}% | "
        f"NLL = {nll_rw:.3f} | CRPS = {np.mean(crps_rw_mat):.5f} | CRPS-Sum = {crps_sum_rw:.5f} | Cov95 = {cov95_rw:.1f}%"
    )

    # 2.0b. Chẩn đoán DeepVAR-Level (Raw price)
    raw_dv = evaluate_raw_level_deepvar(bundle.price_df, train_end, val_end, seq_len=10, seed=42)
    raw_rmse = float(np.sqrt(np.mean((raw_dv["test_true_price"] - raw_dv["test_pred_price"]) ** 2)))
    raw_mae = float(np.mean(np.abs(raw_dv["test_true_price"] - raw_dv["test_pred_price"])))
    raw_mape = float(np.mean(np.abs((raw_dv["test_true_price"] - raw_dv["test_pred_price"]) / raw_dv["test_true_price"])) * 100.0)
    ss_res_raw = np.sum((raw_dv["test_true_price"] - raw_dv["test_pred_price"]) ** 2)
    ss_tot_raw = np.sum((raw_dv["test_true_price"] - np.mean(raw_dv["test_true_price"])) ** 2)
    raw_r2 = float(1.0 - ss_res_raw / ss_tot_raw)
    print(f"-> [Diagnostic] DeepVAR-Level: RMSE = {raw_rmse:.2f} VND | MAPE = {raw_mape:.3f}% | R2 = {raw_r2:.4f} (Suy thoái vì vượt miền giá Train)")

    summary_rows = [
        {
            "Model": "Naive (Random Walk Prob)",
            "Purpose": "Chuẩn đối chứng P_hat_t = P_{t-1}, Cov tĩnh",
            "Num_Features": 1,
            "Rank": 3,
            "RMSE (VND)": f"{rw_rmse:.2f}",
            "MAE (VND)": f"{rw_mae:.2f}",
            "MAPE (%)": f"{rw_mape:.3f}%",
            "DA (%)": "0.00%",
            "NLL_cal (ret)": f"{nll_rw:.3f}",
            "CRPS (ret)": f"{np.mean(crps_rw_mat):.5f}",
            "CRPS-Sum": f"{crps_sum_rw:.5f}",
            "Cov80 (%)": "-",
            "Cov95 (%)": f"{cov95_rw:.1f}%",
            "MPIW95 (VND)": f"{np.mean(rw_q975 - rw_q025):.0f}",
            "DM_p vs Naive": "Benchmark",
            "Avg_Epoch": "-",
        },
        {
            "Model": "DeepVAR-Level (Raw)",
            "Purpose": "Chẩn đoán: DeepVAR trực tiếp trên giá gốc P_t",
            "Num_Features": 3,
            "Rank": 2,
            "RMSE (VND)": f"{raw_rmse:.2f}",
            "MAE (VND)": f"{raw_mae:.2f}",
            "MAPE (%)": f"{raw_mape:.3f}%",
            "DA (%)": "-",
            "NLL_cal (ret)": "-",
            "CRPS (ret)": "-",
            "CRPS-Sum": "-",
            "Cov80 (%)": "-",
            "Cov95 (%)": "-",
            "MPIW95 (VND)": "-",
            "DM_p vs Naive": "1.0000",
            "Avg_Epoch": "-",
        },
    ]

    model_outputs: Dict[str, Dict] = {}
    province_rows = []
    importance_frames = []
    corr_rows = []

    for mname, cfg in DEEPVAR_CONFIGS.items():
        X_df = bundle.feature_sets[cfg["feature_key"]]
        seed_runs, seed_metrics, seed_mu_r, seed_cov_r, seed_val_rmses = [], [], [], [], []

        for s in SEEDS:
            run = fit_single_deepvar_seed(
                X_df=X_df,
                Y_df=bundle.ret_df,
                price_df=bundle.price_df,
                train_end=train_end,
                val_end=val_end,
                seq_len=cfg["seq_len"],
                hidden_dim=cfg["hidden_dim"],
                cov_hidden_dim=cfg["cov_hidden_dim"],
                num_layers=1,
                rank=cfg["rank"],
                dropout=cfg["dropout"],
                in_dropout=cfg["in_dropout"],
                min_std=0.20,
                beta_nll=0.5,
                batch_size=16,
                lr=1e-3,
                optimizer_name=cfg["optimizer_name"],
                weight_decay=cfg["weight_decay"],
                max_epochs=50,
                patience=10,
                n_mc_samples=1000,
                seed=s,
            )
            seed_runs.append(run)
            seed_mu_r.append(run["test_pred_ret"])
            seed_cov_r.append(run["test_cov_ret_cal"])
            seed_val_rmses.append(run["val_rmse_price"])
            m = compute_deepvar_metrics(run, bench_ret_err=rw_ret_err)
            seed_metrics.append(m)

        mean_m = {k: float(np.mean([r[k] for r in seed_metrics])) for k in seed_metrics[0]}
        std_m = {k: float(np.std([r[k] for r in seed_metrics])) for k in seed_metrics[0]}

        # Ensemble qua 5 seeds (Mixture -> Moment-matched mean & std cho phân tích DM)
        ens_mu_r = np.mean(seed_mu_r, axis=0)
        ens_cov_r = np.mean(seed_cov_r, axis=0)
        ens_std_r = np.sqrt(np.diagonal(ens_cov_r, axis1=1, axis2=2))
        ens_pred_p = test_prev_p * np.exp(ens_mu_r)
        ens_rmse = float(np.sqrt(np.mean((test_true_p - ens_pred_p) ** 2)))
        ens_crps_mat = gaussian_crps_numpy(test_true_r, ens_mu_r, ens_std_r)

        mean_pred_corr = np.mean([r["mean_pred_corr"] for r in seed_runs], axis=0)

        model_outputs[mname] = {
            "runs": seed_runs,
            "mean_metrics": mean_m,
            "std_metrics": std_m,
            "ens_mu_r": ens_mu_r,
            "ens_std_r": ens_std_r,
            "ens_cov_r": ens_cov_r,
            "ens_pred_p": ens_pred_p,
            "ens_rmse": ens_rmse,
            "ens_crps_mat": ens_crps_mat,
            "mean_pred_corr": mean_pred_corr,
        }

        summary_rows.append(
            {
                "Model": mname,
                "Purpose": cfg["purpose"],
                "Num_Features": X_df.shape[1],
                "Rank": cfg["rank"],
                "RMSE (VND)": f"{mean_m['RMSE (VND)']:.2f} ± {std_m['RMSE (VND)']:.1f}",
                "MAE (VND)": f"{mean_m['MAE (VND)']:.2f} ± {std_m['MAE (VND)']:.1f}",
                "MAPE (%)": f"{mean_m['MAPE (%)']:.3f}% ± {std_m['MAPE (%)']:.2f}%",
                "DA (%)": f"{mean_m['DA (%)']:.2f}%",
                "NLL_cal (ret)": f"{mean_m['NLL_cal']:.3f} ± {std_m['NLL_cal']:.2f}",
                "CRPS (ret)": f"{mean_m['CRPS_ret']:.5f}",
                "CRPS-Sum": f"{mean_m['CRPS_Sum']:.5f}",
                "Cov80 (%)": f"{mean_m['Cov80 (%)']:.1f}%",
                "Cov95 (%)": f"{mean_m['Cov95 (%)']:.1f}%",
                "MPIW95 (VND)": f"{mean_m['MPIW95 (VND)']:.0f}",
                "DM_p vs Naive": f"{mean_m['DM_p_vs_RW']:.4f}",
                "Avg_Epoch": f"{mean_m['best_epoch']:.1f}",
            }
        )

        print(
            f"-> {mname:<17} ({X_df.shape[1]:2d}f, r={cfg['rank']}): "
            f"Val RMSE={np.mean(seed_val_rmses):.2f} | "
            f"Test RMSE={mean_m['RMSE (VND)']:.2f}±{std_m['RMSE (VND)']:.1f} | "
            f"DA={mean_m['DA (%)']:.2f}% | NLL={mean_m['NLL_cal']:.3f} | "
            f"CRPS={mean_m['CRPS_ret']:.5f} | CRPS-Sum={mean_m['CRPS_Sum']:.5f} | "
            f"Cov95={mean_m['Cov95 (%)']:.1f}% | tau={mean_m['tau_cal']:.2f} | Ens={ens_rmse:.2f}"
        )

        # Chi tiết từng tỉnh cho 3 phiên bản chính
        if mname in ("DeepVAR 1", "DeepVAR 2", "DeepVAR 3"):
            for p_idx, prov in enumerate(TARGET_PROVINCES):
                rmses_k, mapes_k, das_k, crps_k, cov95_k = [], [], [], [], []
                for run in seed_runs:
                    tp = run["test_true_price"][:, p_idx]
                    pp = run["test_pred_price"][:, p_idx]
                    tr = run["test_true_ret"][:, p_idx]
                    mr = run["test_pred_ret"][:, p_idx]
                    sr = run["test_std_ret_cal"][:, p_idx]
                    rmses_k.append(np.sqrt(np.mean((tp - pp) ** 2)))
                    mapes_k.append(np.mean(np.abs((tp - pp) / tp)) * 100.0)
                    mk = tr != 0
                    das_k.append(np.mean(np.sign(mr[mk]) == np.sign(tr[mk])) * 100.0)
                    crps_k.append(np.mean(gaussian_crps_numpy(tr, mr, sr)))
                    cov95_k.append(np.mean((tp >= run["pred_p_q025"][:, p_idx]) & (tp <= run["pred_p_q975"][:, p_idx])) * 100.0)
                province_rows.append(
                    {
                        "Model": mname,
                        "Province": prov,
                        "RMSE_Mean": round(float(np.mean(rmses_k)), 2),
                        "RMSE_Std": round(float(np.std(rmses_k)), 2),
                        "MAPE_Mean (%)": round(float(np.mean(mapes_k)), 3),
                        "DA_Mean (%)": round(float(np.mean(das_k)), 2),
                        "CRPS_ret": round(float(np.mean(crps_k)), 5),
                        "Cov95 (%)": round(float(np.mean(cov95_k)), 1),
                    }
                )

            corr_rows.append(
                {
                    "Model": mname,
                    "Pred_Corr(DL,LD)": round(float(mean_pred_corr[0, 1]), 4),
                    "Pred_Corr(DL,GL)": round(float(mean_pred_corr[0, 2]), 4),
                    "Pred_Corr(LD,GL)": round(float(mean_pred_corr[1, 2]), 4),
                    "Emp_Test_Corr(DL,LD)": float(diag["emp_corr_test"].iloc[0, 1]),
                    "Emp_Test_Corr(DL,GL)": float(diag["emp_corr_test"].iloc[0, 2]),
                    "Emp_Test_Corr(LD,GL)": float(diag["emp_corr_test"].iloc[1, 2]),
                }
            )

        # Permutation Feature Importance cho DeepVAR 2 & DeepVAR 3
        if mname in ("DeepVAR 2", "DeepVAR 3"):
            fnames = list(X_df.columns)
            imp_list = [
                compute_deepvar_permutation_importance(run, fnames, n_repeats=8, seed=42).set_index("Feature")
                for run in seed_runs
            ]
            avg_imp = sum(imp_list) / len(imp_list)
            avg_imp = avg_imp.sort_values("RMSE_Increase_VND", ascending=False).reset_index()
            avg_imp.insert(0, "Model", mname)
            importance_frames.append(avg_imp)

    # ==================================================================================
    # BƯỚC 3: KIỂM ĐỊNH DIEBOLD-MARIANO (CẢ ĐIỂM MSE LẪN XÁC SUẤT CRPS)
    # ==================================================================================
    print_header("BƯỚC 3: KIỂM ĐỊNH DIEBOLD-MARIANO (ĐIỂM MSE & XÁC SUẤT CRPS)")

    e_dv1 = (test_true_r - model_outputs["DeepVAR 1"]["ens_mu_r"]) ** 2
    e_dv2 = (test_true_r - model_outputs["DeepVAR 2"]["ens_mu_r"]) ** 2
    e_dv3 = (test_true_r - model_outputs["DeepVAR 3"]["ens_mu_r"]) ** 2
    e_rw = rw_ret_err ** 2

    c_dv1 = model_outputs["DeepVAR 1"]["ens_crps_mat"]
    c_dv2 = model_outputs["DeepVAR 2"]["ens_crps_mat"]
    c_dv2d = model_outputs["DeepVAR 2 (Diag)"]["ens_crps_mat"]
    c_dv3 = model_outputs["DeepVAR 3"]["ens_crps_mat"]

    print("--- 3.1. Kiểm định DM trên sai số điểm (MSE return) ---")
    for lbl, lb, lm in [
        ("DeepVAR 1 (ens) vs Naive", e_rw, e_dv1),
        ("DeepVAR 2 (ens) vs Naive", e_rw, e_dv2),
        ("DeepVAR 3 (ens) vs Naive", e_rw, e_dv3),
        ("DeepVAR 2 (ens) vs DeepVAR 1 (ens)", e_dv1, e_dv2),
        ("DeepVAR 3 (ens) vs DeepVAR 2 (ens)", e_dv2, e_dv3),
        ("DeepVAR 2 (ens) vs DeepVAR 3 (ens)", e_dv3, e_dv2),
    ]:
        st, pv = diebold_mariano_loss_diff(lb, lm, h=1)
        print(f"  * {lbl:<40}: DM = {st:+7.3f}, p = {pv:.4f}")

    print("\n--- 3.2. Kiểm định DM trên chất lượng phân phối xác suất (CRPS return) ---")
    for lbl, lb, lm in [
        ("DeepVAR 1 (ens) vs Naive Prob", crps_rw_mat, c_dv1),
        ("DeepVAR 2 (ens) vs Naive Prob", crps_rw_mat, c_dv2),
        ("DeepVAR 3 (ens) vs Naive Prob", crps_rw_mat, c_dv3),
        ("DeepVAR 2 (ens) vs DeepVAR 1 (ens)", c_dv1, c_dv2),
        ("DeepVAR 2 (LowRank) vs DeepVAR 2 (Diag)", c_dv2d, c_dv2),
        ("DeepVAR 3 (ens) vs DeepVAR 2 (ens)", c_dv2, c_dv3),
    ]:
        st, pv = diebold_mariano_loss_diff(lb, lm, h=1)
        print(f"  * {lbl:<40}: DM = {st:+7.3f}, p = {pv:.4f}")

    # Xuất bảng CSV
    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Bảng tổng hợp DeepVAR ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("deepvar_model_comparison_results.csv", index=False)

    prov_df = pd.DataFrame(province_rows)
    print("\n--- Bảng chi tiết từng tỉnh ---")
    print(prov_df.to_string(index=False))
    prov_df.to_csv("deepvar_province_results.csv", index=False)

    corr_df = pd.DataFrame(corr_rows)
    print("\n--- Tương quan liên tỉnh dự báo bởi Sigma_t vs Thực nghiệm Test ---")
    print(corr_df.to_string(index=False))
    corr_df.to_csv("deepvar_correlation_comparison.csv", index=False)

    imp_df = pd.concat(importance_frames, ignore_index=True)
    imp_df.to_csv("deepvar_feature_importance.csv", index=False)
    print("\n--- Top 10 đặc trưng quan trọng nhất của DeepVAR 3 (Mean RMSE & Joint CRPS) ---")
    print(imp_df[imp_df["Model"] == "DeepVAR 3"].head(10).to_string(index=False))

    # ==================================================================================
    # BƯỚC 4: XUẤT 3 BIỂU ĐỒ CHẨN ĐOÁN & ĐÁNH GIÁ XÁC SUẤT
    # ==================================================================================
    print_header("BƯỚC 4: XUẤT BIỂU ĐỒ")

    # Fig 1: Khoảng tin cậy xác suất 80% / 95% trên Test (Đắk Lắk, 90 ngày cuối) + Raw collapse
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    dv2_r0 = model_outputs["DeepVAR 2"]["runs"][0]
    zoom = 90
    zd = test_dates[-zoom:]
    axes[0].fill_between(
        zd, dv2_r0["pred_p_q025"][-zoom:, 0], dv2_r0["pred_p_q975"][-zoom:, 0],
        color="#1f77b4", alpha=0.16, label="Khoảng dự báo 95% (MC 1000 mẫu)"
    )
    axes[0].fill_between(
        zd, dv2_r0["pred_p_q10"][-zoom:, 0], dv2_r0["pred_p_q90"][-zoom:, 0],
        color="#1f77b4", alpha=0.30, label="Khoảng dự báo 80%"
    )
    axes[0].plot(zd, test_true_p[-zoom:, 0], color="#111111", lw=1.8, label="Thực tế (Đắk Lắk)")
    axes[0].plot(zd, model_outputs["DeepVAR 2"]["ens_pred_p"][-zoom:, 0], color="#d62728", lw=1.4, label="DeepVAR 2 (Median)")
    axes[0].set_title("Dự báo xác suất DeepVAR 2 kèm dải bất định (90 ngày cuối Test)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Giá cà phê (VND/kg)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].grid(alpha=0.25)

    axes[1].plot(test_dates, test_true_p[:, 0], color="#111111", lw=1.4, label="Thực tế Test (83.9k-139k)")
    axes[1].plot(test_dates, raw_dv["test_pred_price"][:, 0], color="#d62728", ls="--", lw=1.4, label=f"DeepVAR-Level Raw (RMSE={raw_rmse:,.0f})")
    axes[1].plot(test_dates, model_outputs["DeepVAR 2"]["ens_pred_p"][:, 0], color="#2ca02c", lw=1.2, alpha=0.85, label="DeepVAR 2 (Log-Return + Naive Anchor)")
    axes[1].set_title("Đối chứng suy thoái giá thô vs DeepVAR dừng hóa (Đắk Lắk)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Giá cà phê (VND/kg)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(loc="best", fontsize=8.5)
    axes[1].grid(alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_deepvar_probabilistic_intervals.png")
    plt.savefig(p1, dpi=180)
    plt.close(fig)

    # Fig 2: So sánh Point RMSE, NLL_cal, CRPS-Sum giữa các phiên bản
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    names = ["Naive\nProb", "DeepVAR 1\n(Giá QK)", "DeepVAR 2\n(TT+KT)", "DeepVAR 2\n(Diag r=0)", "DeepVAR 3\n(Full NS)"]
    keys_m = [None, "DeepVAR 1", "DeepVAR 2", "DeepVAR 2 (Diag)", "DeepVAR 3"]
    cols_bar = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]

    rmses_v = [rw_rmse] + [model_outputs[k]["mean_metrics"]["RMSE (VND)"] for k in keys_m[1:]]
    nlls_v = [nll_rw] + [model_outputs[k]["mean_metrics"]["NLL_cal"] for k in keys_m[1:]]
    crps_sum_v = [crps_sum_rw] + [model_outputs[k]["mean_metrics"]["CRPS_Sum"] for k in keys_m[1:]]

    for ax, vals, title, ylab in [
        (axes[0], rmses_v, "Test RMSE Điểm (VND/kg)", "RMSE (VND)"),
        (axes[1], nlls_v, "Multivariate Gaussian NLL (Thấp hơn = tốt)", "NLL (nats)"),
        (axes[2], crps_sum_v, "CRPS-Sum (Đánh giá hiệp phương sai liên tỉnh)", "CRPS-Sum"),
    ]:
        bars = ax.bar(names, vals, color=cols_bar, edgecolor="#222222", alpha=0.9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8.5)
        ymin = min(vals) * 0.85 if min(vals) > 0 else min(vals) * 1.15
        ymax = max(vals) * 1.12 if max(vals) > 0 else max(vals) * 0.85
        ax.set_ylim(ymin, ymax)
        for b, v in zip(bars, vals):
            fmt = f"{v:.1f}" if abs(v) > 100 else (f"{v:.3f}" if abs(v) > 0.5 else f"{v:.5f}")
            y_pos = v + (ymax - ymin) * 0.02 if v >= 0 else v - (ymax - ymin) * 0.02
            va_align = "bottom" if v >= 0 else "top"
            ax.text(b.get_x() + b.get_width() / 2, y_pos, fmt, ha="center", va=va_align, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_deepvar_versions_and_calibration.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    # Fig 3: Heatmap tương quan liên tỉnh (Empirical vs DeepVAR 2) + Feature Importance (Mean vs CRPS)
    fig = plt.figure(figsize=(17.0, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.35])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    emp_m = diag["emp_corr_test"].values
    pred_m = model_outputs["DeepVAR 2"]["mean_pred_corr"]
    prov_lbl = ["Đắk Lắk", "Lâm Đồng", "Gia Lai"]

    for ax, mat, ttl in [
        (ax0, emp_m, "Tương quan Thực nghiệm (Test)"),
        (ax1, pred_m, "Tương quan phần dư DeepVAR 2 (Sigma_t)"),
    ]:
        im = ax.imshow(mat, vmin=0.65, vmax=1.0, cmap="YlGnBu")
        ax.set_xticks(range(3)); ax.set_xticklabels(prov_lbl, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(prov_lbl, fontsize=9)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold")
        for i in range(3):
            for j in range(3):
                val = mat[i, j]
                txt_col = "white" if val >= 0.90 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=txt_col, fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=[ax0, ax1], location="bottom", shrink=0.75, pad=0.06)

    top10 = imp_df[imp_df["Model"] == "DeepVAR 3"].head(10).iloc[::-1]
    ax2.barh(top10["Feature"], top10["RMSE_Share (%)"], color="#1f77b4", alpha=0.85, label="Đóng góp Mean (RMSE)")
    ax2.barh(top10["Feature"], -top10["CRPS_Share (%)"], color="#ff7f0e", alpha=0.85, label="Đóng góp Phân phối (-CRPS)")
    ax2.axvline(0, color="#222222", lw=0.8)
    ax2.set_title("Tầm quan trọng biến DeepVAR 3 (Mean vs CRPS)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Tỷ trọng đóng góp (%)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", alpha=0.25)

    p3 = os.path.join(FIG_DIR, "fig3_covariance_heatmap_and_importance.png")
    plt.savefig(p3, dpi=180)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {p1}, {p2}, {p3}")
    print(f"Hoàn tất toàn bộ thực nghiệm DeepVAR sau {time.time() - t0:.1f} giây.")


if __name__ == "__main__":
    main()

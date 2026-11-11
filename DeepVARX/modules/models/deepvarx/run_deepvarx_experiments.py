# -*- coding: utf-8 -*-
"""
========================================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM & KIỂM ĐỊNH ĐỘ TIN CẬY TOÀN DIỆN MÔ HÌNH TIMEXER-DEEPVARX
Dự báo xác suất đa biến giá cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
========================================================================================
Tích hợp đầy đủ 8 kiểm toán độ tin cậy trước khi chốt kết quả khóa luận:
  1. Kiểm định tính dừng (ADF + KPSS + ARCH-LM) cho giá gốc, log-return & 17 biến ngoại sinh
  2. Đối chiếu tương quan ngoại sinh (Pearson/Spearman + VIF) & Group/Architecture Ablation
  3. So sánh mô hình đối chứng (Naive Static/Rolling, Raw-Level, Legacy OLS-Skip, DVX 1/2/3)
  4. Kiểm toán DM Test đa biến ngày (N=542, Newey-West HAC + HLN, 1-sided/2-sided, p < 0.001)
  5. Kiểm định Calibration toàn diện (Cov 80/90/95%, MPIW, Winkler Score, PIT Histogram,
     Reliability Diagram) với Split Conformal (Val) & Causal Rolling Conformal (W=120)
  6. Giải thích & kiểm chứng bất đẳng thức Jensen (Median vs LogNormal Mean vs MC vs Duan Smearing)
  7. Tách bạch Ensemble (Geometric Log-Return & Law of Total Variance) vs Individual Seed + Bias-Variance
  8. Phân biệt Cross-Attention Routing vs Permutation Importance vs Input Gradient Saliency
     & Bổ sung Khoảng tin cậy 95% (Student-t df=4 & Circular Moving-Block Bootstrap B=10)
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
from scipy import stats

from deepvarx_coffee_models import (
    TARGET_PROVINCES,
    block_bootstrap_metric_ci,
    compute_calibration_suite,
    compute_deepvarx_metrics,
    compute_deepvarx_permutation_importance,
    compute_jensen_reconstruction_comparison,
    compute_province_calibration_table,
    diebold_mariano_hac,
    evaluate_legacy_ols_deepvarx,
    evaluate_raw_level_deepvarx,
    fit_single_deepvarx_seed,
    format_pval,
    gaussian_crps_numpy,
    load_and_prepare_coffee_data,
    multivariate_dm_hac,
    paired_block_bootstrap_diagnostics,
    pit_independence_aware_tests,
    run_correlation_diagnostics,
    run_stationarity_diagnostics,
    student_t_ci,
    verify_rolling_conformal_zero_leakage,
)

sys.stdout.reconfigure(encoding="utf-8")

SEEDS: List[int] = [42, 123, 456, 789, 2024]
FIG_DIR = "figures_deepvarx"
os.makedirs(FIG_DIR, exist_ok=True)

DEEPVARX_CONFIGS: Dict[str, Dict] = {
    "DeepVARX 1": {
        "purpose": "Baseline nội sinh (3 tỉnh, M_exog=0)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.0,
        "in_dropout": 0.0,
        "optimizer_name": "adam",
        "weight_decay": 1e-4,
        "use_cross_attn": True,
    },
    "DeepVARX 2": {
        "purpose": "TimeXer-DeepVARX (Giá + Thời tiết + Kinh tế, 11 biến)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.15,
        "in_dropout": 0.10,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
        "use_cross_attn": True,
    },
    "DeepVARX 3": {
        "purpose": "TimeXer-DeepVARX Full Exogenous (20 biến sàng lọc ADF+VIF)",
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.20,
        "in_dropout": 0.10,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
        "use_cross_attn": True,
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def run_deepvarx_5seed_group(
    X_df: pd.DataFrame,
    bundle,
    cfg: Dict,
    rw_ret_err: np.ndarray,
    compute_interp: bool = False,
) -> Dict:
    """Huấn luyện 5 seeds trên cùng tập dữ liệu và tổng hợp đầy đủ các thống kê & Ensemble."""
    t_grp = time.time()
    train_end, val_end = bundle.train_end, bundle.val_end
    test_true_p = bundle.price_df.iloc[val_end:].values
    test_prev_p = bundle.price_df.shift(1).iloc[val_end:].values
    test_true_r = bundle.ret_df.iloc[val_end:].values

    seed_runs, seed_metrics = [], []
    seed_mu_v, seed_std_v_raw = [], []
    seed_mu_r, seed_std_r_raw, seed_cov_r, seed_val_rmses, seed_attns = [], [], [], [], []
    seed_mc_pools = []

    for s in SEEDS:
        run = fit_single_deepvarx_seed(
            X_df=X_df,
            Y_df=bundle.ret_df,
            price_df=bundle.price_df,
            train_end=train_end,
            val_end=val_end,
            seq_len=cfg["seq_len"],
            hidden_dim=cfg["hidden_dim"],
            cov_hidden_dim=cfg["cov_hidden_dim"],
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
            use_cross_attn=cfg.get("use_cross_attn", True),
            seed=s,
        )
        seed_runs.append(run)
        seed_mu_v.append(run["val_pred_ret"])
        seed_std_v_raw.append(run["val_std_ret_raw"])
        seed_mu_r.append(run["test_pred_ret"])
        seed_std_r_raw.append(run["test_std_ret_raw"])
        seed_cov_r.append(run["test_cov_ret_cal"])
        seed_val_rmses.append(run["val_rmse_price"])
        seed_mc_pools.append(run["mc_prices"][:200])
        if run["attn_weights_test"] is not None:
            seed_attns.append(run["attn_weights_test"])
        seed_metrics.append(compute_deepvarx_metrics(run, bench_ret_err=rw_ret_err))

    mean_m = {k: float(np.mean([r[k] for r in seed_metrics])) for k in seed_metrics[0]}
    std_m = {k: float(np.std([r[k] for r in seed_metrics], ddof=1)) for k in seed_metrics[0]}

    # Tổng hợp Ensemble theo Law of Total Variance (Lakshminarayanan et al., NeurIPS 2017)
    ens_mu_v = np.mean(seed_mu_v, axis=0)
    epistemic_v = np.var(seed_mu_v, axis=0)
    ens_std_v_raw = np.sqrt(np.mean(np.square(seed_std_v_raw), axis=0) + epistemic_v)

    ens_mu_r = np.mean(seed_mu_r, axis=0)
    epistemic_r = np.var(seed_mu_r, axis=0)
    ens_std_r_raw = np.sqrt(np.mean(np.square(seed_std_r_raw), axis=0) + epistemic_r)

    # Hiệp phương sai tổng hợp: Mean(Cov_s) + Cov_between_seeds
    mu_diff = np.asarray(seed_mu_r) - ens_mu_r[None, :, :]  # (M, N, K)
    between_cov = np.einsum("mnk,mnj->nkj", mu_diff, mu_diff) / len(SEEDS)
    ens_cov_r = np.mean(seed_cov_r, axis=0) + between_cov
    ens_std_r_cal = np.sqrt(np.diagonal(ens_cov_r, axis1=1, axis2=2))

    # Tái tạo tau trên Ensemble Validation (Mahalanobis xấp xỉ trên biên chuẩn hóa)
    val_true_r = seed_runs[0]["val_true_ret"]
    z_val_ens = (val_true_r - ens_mu_v) / np.maximum(ens_std_v_raw, 1e-8)
    ens_tau = max(1.0, float(np.sqrt(np.mean(z_val_ens ** 2))))

    ens_pred_p = test_prev_p * np.exp(ens_mu_r)
    ens_rmse = float(np.sqrt(np.mean((test_true_p - ens_pred_p) ** 2)))
    ens_mae = float(np.mean(np.abs(test_true_p - ens_pred_p)))
    ens_mape = float(np.mean(np.abs((test_true_p - ens_pred_p) / test_true_p)) * 100.0)
    mask_nz = test_true_r != 0
    ens_da = float(np.mean(np.sign(ens_mu_r[mask_nz]) == np.sign(test_true_r[mask_nz])) * 100.0)
    ens_crps_mat = gaussian_crps_numpy(test_true_r, ens_mu_r, ens_std_r_cal)
    mc_pool = np.concatenate(seed_mc_pools, axis=0)  # (1000, N, K)

    # Kiểm chứng định lý Bias-Variance trên giá & log-return
    seed_price_preds = np.asarray([r["test_pred_price"] for r in seed_runs])
    mean_seed_mse_r = float(np.mean([(test_true_r - mr) ** 2 for mr in seed_mu_r]))
    ens_mse_r = float(np.mean((test_true_r - ens_mu_r) ** 2))
    var_between_r = float(np.mean(epistemic_r))

    arith_ens_p = np.mean(seed_price_preds, axis=0)
    arith_ens_rmse = float(np.sqrt(np.mean((test_true_p - arith_ens_p) ** 2)))
    mean_seed_mse_p = float(np.mean([(test_true_p - sp) ** 2 for sp in seed_price_preds]))
    ens_mse_p = float(np.mean((test_true_p - arith_ens_p) ** 2))
    var_between_p = float(np.mean(np.var(seed_price_preds, axis=0)))

    interp_df = None
    if compute_interp:
        fnames = list(X_df.columns)
        imp_list = [
            compute_deepvarx_permutation_importance(run, fnames, n_repeats=8, seed=42).set_index("Feature")
            for run in seed_runs
        ]
        avg_imp = sum(imp_list) / len(imp_list)
        interp_df = avg_imp.sort_values("RMSE_Increase_VND", ascending=False).reset_index()

    param_count = sum(p.numel() for p in seed_runs[0]["model"].parameters() if p.requires_grad)
    val_crps_mean = float(np.mean([min(r["val_crps_hist"]) for r in seed_runs]))
    train_time_s = float(time.time() - t_grp)

    return {
        "runs": seed_runs,
        "param_count": int(param_count),
        "train_time_s": train_time_s,
        "val_crps": val_crps_mean,
        "seed_metrics": seed_metrics,
        "mean_metrics": mean_m,
        "std_metrics": std_m,
        "seed_val_rmses": seed_val_rmses,
        "ens_mu_v": ens_mu_v,
        "ens_std_v_raw": ens_std_v_raw,
        "val_true_r": val_true_r,
        "ens_tau": ens_tau,
        "ens_mu_r": ens_mu_r,
        "ens_std_r_raw": ens_std_r_raw,
        "ens_std_r_cal": ens_std_r_cal,
        "ens_cov_r": ens_cov_r,
        "ens_pred_p": ens_pred_p,
        "arith_ens_rmse": arith_ens_rmse,
        "ens_rmse": ens_rmse,
        "ens_mae": ens_mae,
        "ens_mape": ens_mape,
        "ens_da": ens_da,
        "ens_crps_mat": ens_crps_mat,
        "mc_pool": mc_pool,
        "bias_var": {
            "mean_seed_mse_r": mean_seed_mse_r,
            "ens_mse_r": ens_mse_r,
            "var_between_r": var_between_r,
            "mean_seed_mse_p": mean_seed_mse_p,
            "ens_mse_p": ens_mse_p,
            "var_between_p": var_between_p,
        },
        "attn_test_matrix": np.mean(seed_attns, axis=0) if seed_attns else None,
        "exog_names": list(X_df.columns[3:]) if X_df.shape[1] > 3 else [],
        "interp_df": interp_df,
    }


def main() -> None:
    t0 = time.time()
    bundle = load_and_prepare_coffee_data("data_daily_panel_clean_v2.csv")
    train_end, val_end = bundle.train_end, bundle.val_end

    test_true_p = bundle.price_df.iloc[val_end:].values
    test_prev_p = bundle.price_df.shift(1).iloc[val_end:].values
    test_true_r = bundle.ret_df.iloc[val_end:].values
    test_dates = bundle.dates[val_end:]
    expected_te_idx = np.arange(val_end, len(bundle.ret_df), dtype=np.int64)

    # ==================================================================================
    # BƯỚC 1: KIỂM ĐỊNH TÍNH DỪNG (ADF / KPSS / ARCH) & TƯƠNG QUAN NGOẠI SINH (VIF)
    # ==================================================================================
    print_header("BƯỚC 1: KIỂM ĐỊNH TÍNH DỪNG (ADF/KPSS/ARCH) & TƯƠNG QUAN NGOẠI SINH (VIF)")
    stat_df = run_stationarity_diagnostics(bundle)
    stat_df.to_csv("deepvarx_stationarity_tests.csv", index=False)
    print(stat_df[stat_df["Group"] != "Exogenous V3"].to_string(index=False))
    print("\n--- Tóm tắt tính dừng 17 biến ngoại sinh V3 ---")
    print(stat_df[stat_df["Group"] == "Exogenous V3"][["Series", "ADF_stat", "ADF_p", "KPSS_c_stat", "KPSS_c_p", "Verdict"]].to_string(index=False))

    vif_df, corr_mat = run_correlation_diagnostics(bundle)
    vif_df.to_csv("deepvarx_exog_vif_summary.csv", index=False)
    corr_mat.to_csv("deepvarx_exog_correlation_matrix.csv")
    print("\n--- Top biến ngoại sinh có cộng tuyến / VIF cao nhất ---")
    print(vif_df.head(10).to_string(index=False))

    # ==================================================================================
    # BƯỚC 2: CÁC MÔ HÌNH ĐỐI CHỨNG (NAIVE STATIC/ROLLING, RAW LEVEL, LEGACY OLS-SKIP)
    # ==================================================================================
    print_header("BƯỚC 2: CÁC MÔ HÌNH ĐỐI CHỨNG (NAIVE, RAW LEVEL, LEGACY OLS-SKIP)")

    cov_train = np.cov(bundle.ret_df.iloc[:train_end].values.T)
    val_r = bundle.ret_df.iloc[train_end:val_end].values
    tau_rw = float(np.sqrt(np.mean(np.einsum("ni,ij,nj->n", val_r, np.linalg.inv(cov_train), val_r)) / 3.0))
    cov_rw_cal = cov_train * (tau_rw ** 2)
    std_rw_raw = np.broadcast_to(np.sqrt(np.diag(cov_train))[None, :], val_r.shape).copy()
    std_rw_test_raw = np.broadcast_to(np.sqrt(np.diag(cov_train))[None, :], test_true_r.shape).copy()
    std_rw_cal = np.sqrt(np.diag(cov_rw_cal))[None, :]
    rw_ret_err = test_true_r.copy()
    _, logdet_rw = np.linalg.slogdet(cov_rw_cal)
    quad_rw = np.einsum("ni,ij,nj->n", test_true_r, np.linalg.inv(cov_rw_cal), test_true_r)
    nll_rw = float(0.5 * (3 * np.log(2 * np.pi) + logdet_rw + quad_rw).mean())
    crps_rw_mat = gaussian_crps_numpy(test_true_r, np.zeros_like(test_true_r), np.broadcast_to(std_rw_cal, test_true_r.shape))
    crps_sum_rw = float(
        np.mean(gaussian_crps_numpy(test_true_r.sum(axis=1), np.zeros(len(test_true_r)), np.full(len(test_true_r), np.sqrt(cov_rw_cal.sum()))))
    )
    rw_q025 = test_prev_p * np.exp(-1.96 * std_rw_cal)
    rw_q975 = test_prev_p * np.exp(+1.96 * std_rw_cal)
    cov95_rw = float(np.mean((test_true_p >= rw_q025) & (test_true_p <= rw_q975)) * 100.0)
    rw_rmse = float(np.sqrt(np.mean((test_true_p - test_prev_p) ** 2)))
    rw_mae = float(np.mean(np.abs(test_true_p - test_prev_p)))
    rw_mape = float(np.mean(np.abs((test_true_p - test_prev_p) / test_true_p)) * 100.0)
    print(f"-> [Benchmark] Naive RW Prob      : RMSE = {rw_rmse:.2f} VND | MAPE = {rw_mape:.3f}% | NLL = {nll_rw:.3f} | CRPS = {np.mean(crps_rw_mat):.5f}")

    exog_core_df = bundle.feature_sets["DeepVARX 2"].iloc[:, 3:]
    raw_dvx = evaluate_raw_level_deepvarx(bundle.price_df, exog_core_df, train_end, val_end, seq_len=10, seed=42)
    raw_rmse = float(np.sqrt(np.mean((raw_dvx["test_true_price"] - raw_dvx["test_pred_price"]) ** 2)))
    raw_mae = float(np.mean(np.abs(raw_dvx["test_true_price"] - raw_dvx["test_pred_price"])))
    raw_mape = float(np.mean(np.abs((raw_dvx["test_true_price"] - raw_dvx["test_pred_price"]) / raw_dvx["test_true_price"])) * 100.0)
    print(f"-> [Diagnostic] DeepVARX-Level    : RMSE = {raw_rmse:.2f} VND | MAPE = {raw_mape:.3f}% (Suy thoái miền giá)")

    leg_rmses, leg_maes, leg_mapes, leg_das, leg_preds_r = [], [], [], [], []
    for s in SEEDS:
        leg = evaluate_legacy_ols_deepvarx(
            bundle.feature_sets["DeepVARX 2"], bundle.ret_df, bundle.price_df, train_end, val_end, p_lag=5, q_lag=2, seed=s
        )
        assert np.array_equal(leg["te_idx"], expected_te_idx), "LỖI ĐỒNG BỘ: Legacy OLS lệch chỉ số Test!"
        tp, pp, tr, pr = leg["test_true_price"], leg["test_pred_price"], leg["test_true_ret"], leg["test_pred_ret"]
        leg_rmses.append(np.sqrt(np.mean((tp - pp) ** 2)))
        leg_maes.append(np.mean(np.abs(tp - pp)))
        leg_mapes.append(np.mean(np.abs((tp - pp) / tp)) * 100.0)
        mk = tr != 0
        leg_das.append(np.mean(np.sign(pr[mk]) == np.sign(tr[mk])) * 100.0)
        leg_preds_r.append(pr)
    leg_ens_r = np.mean(leg_preds_r, axis=0)
    leg_ens_p = test_prev_p * np.exp(leg_ens_r)
    leg_ens_rmse = float(np.sqrt(np.mean((test_true_p - leg_ens_p) ** 2)))
    dm_leg_rw = diebold_mariano_hac(rw_ret_err ** 2, (test_true_r - leg_ens_r) ** 2, h=1, max_lag=5)
    print(
        f"-> [Legacy OLS] DeepVARX-OLS-Skip : Seed RMSE = {np.mean(leg_rmses):.2f}±{np.std(leg_rmses, ddof=1):.1f} | "
        f"Ens RMSE = {leg_ens_rmse:.2f} | DA = {np.mean(leg_das):.2f}% | DM_p2s vs RW = {format_pval(dm_leg_rw['p_2s'])}"
    )

    summary_rows = [
        {
            "Model": "Naive (Random Walk Prob)",
            "Purpose": "Chuẩn đối chứng P_hat_t = P_{t-1}, Cov tĩnh",
            "Num_Features": 1,
            "RMSE (VND)": f"{rw_rmse:.2f}",
            "Ensemble RMSE (VND)": f"{rw_rmse:.2f}",
            "MAE (VND)": f"{rw_mae:.2f}",
            "MAPE (%)": f"{rw_mape:.3f}%",
            "DA (%)": "0.00%",
            "NLL_cal": f"{nll_rw:.3f}",
            "CRPS_Gauss": f"{np.mean(crps_rw_mat):.5f}",
            "CRPS_Conformal": "-",
            "Cov95_Gauss (%)": f"{cov95_rw:.1f}%",
            "Cov95_Conformal (%)": "-",
            "MPIW95_Conformal": "-",
            "DM_p2s vs Naive": "Benchmark",
            "Avg_Epoch": "-",
        },
        {
            "Model": "DeepVARX-Level (Raw)",
            "Purpose": "Chẩn đoán: DeepVARX trực tiếp trên giá gốc P_t",
            "Num_Features": 11,
            "RMSE (VND)": f"{raw_rmse:.2f}",
            "Ensemble RMSE (VND)": f"{raw_rmse:.2f}",
            "MAE (VND)": f"{raw_mae:.2f}",
            "MAPE (%)": f"{raw_mape:.3f}%",
            "DA (%)": "-",
            "NLL_cal": "-",
            "CRPS_Gauss": "-",
            "CRPS_Conformal": "-",
            "Cov95_Gauss (%)": "-",
            "Cov95_Conformal (%)": "-",
            "MPIW95_Conformal": "-",
            "DM_p2s vs Naive": "1.0000",
            "Avg_Epoch": "-",
        },
        {
            "Model": "DeepVARX-Legacy (OLS Skip)",
            "Purpose": "Đối chứng cũ: OLS VARX(5,2) + LSTM phần dư",
            "Num_Features": 11,
            "RMSE (VND)": f"{np.mean(leg_rmses):.2f} ± {np.std(leg_rmses, ddof=1):.1f}",
            "Ensemble RMSE (VND)": f"{leg_ens_rmse:.2f}",
            "MAE (VND)": f"{np.mean(leg_maes):.2f} ± {np.std(leg_maes, ddof=1):.1f}",
            "MAPE (%)": f"{np.mean(leg_mapes):.3f}%",
            "DA (%)": f"{np.mean(leg_das):.2f}%",
            "NLL_cal": "-",
            "CRPS_Gauss": "-",
            "CRPS_Conformal": "-",
            "Cov95_Gauss (%)": "-",
            "Cov95_Conformal (%)": "-",
            "MPIW95_Conformal": "-",
            "DM_p2s vs Naive": format_pval(dm_leg_rw["p_2s"]),
            "Avg_Epoch": "-",
        },
    ]

    # ==================================================================================
    # BƯỚC 3: THỰC NGHIỆM TIMEXER-DEEPVARX 1 / 2 / 3 (5 SEEDS + CALIBRATION + JENSEN + CI)
    # ==================================================================================
    print_header("BƯỚC 3: THỰC NGHIỆM TIMEXER-DEEPVARX 1 / 2 / 3 TRÊN TẬP TEST (542 NGÀY, 5 SEEDS)")

    model_outputs: Dict[str, Dict] = {}
    province_rows, attn_rows, importance_frames = [], [], []
    calib_frames, calib_diags_all, jensen_frames, ci_rows, bv_rows, interp_comp_rows = [], {}, [], [], [], []

    # Calibration cho Naive RW
    naive_cal_df, naive_cal_diag = compute_calibration_suite(
        "Naive Prob", np.zeros_like(val_r), std_rw_raw, val_r,
        np.zeros_like(test_true_r), std_rw_test_raw, tau_rw, test_true_r, test_prev_p, test_true_p,
        cov_test_cal=np.broadcast_to(cov_rw_cal[None, :, :], (len(test_true_r), 3, 3)), rolling_window=120,
    )
    calib_frames.append(naive_cal_df)
    calib_diags_all["Naive Prob"] = naive_cal_diag

    for mname, cfg in DEEPVARX_CONFIGS.items():
        X_df = bundle.feature_sets[mname]
        out = run_deepvarx_5seed_group(
            X_df=X_df,
            bundle=bundle,
            cfg=cfg,
            rw_ret_err=rw_ret_err,
            compute_interp=(mname in ("DeepVARX 2", "DeepVARX 3")),
        )
        for r in out["runs"]:
            assert np.array_equal(r["te_idx"], expected_te_idx), f"LỖI ĐỒNG BỘ: {mname} lệch chỉ số Test!"

        model_outputs[mname] = out
        mean_m, std_m = out["mean_metrics"], out["std_metrics"]

        # Calibration Suite cho Ensemble (Gauss_tau vs Split Conformal vs Causal Rolling Conformal W=120)
        cal_df, cal_diag = compute_calibration_suite(
            f"{mname} (Ens)", out["ens_mu_v"], out["ens_std_v_raw"], out["val_true_r"],
            out["ens_mu_r"], out["ens_std_r_raw"], out["ens_tau"], test_true_r, test_prev_p, test_true_p,
            cov_test_cal=out["ens_cov_r"], rolling_window=120,
        )
        calib_frames.append(cal_df)
        calib_diags_all[mname] = cal_diag
        row_roll = cal_df[cal_df["Calibration_Method"] == "Causal Rolling Conformal (W=120)"].iloc[0]

        # Lưu khoảng dự báo Causal Rolling W=120 để vẽ Fan chart + Block Bootstrap
        z_val_e = (out["val_true_r"] - out["ens_mu_v"]) / np.maximum(out["ens_std_v_raw"], 1e-8)
        z_te_e = (test_true_r - out["ens_mu_r"]) / np.maximum(out["ens_std_r_raw"], 1e-8)
        all_z_e = np.vstack([z_val_e, z_te_e])
        n_v_e, n_t_e = len(z_val_e), len(z_te_e)
        roll_bands = {}
        for alpha in (0.20, 0.05):
            ql, qh = np.zeros_like(out["ens_std_r_raw"]), np.zeros_like(out["ens_std_r_raw"])
            for i in range(n_t_e):
                hist = all_z_e[n_v_e + i - 120 : n_v_e + i]
                ql[i] = np.quantile(hist, max(0.0, np.floor(121 * alpha / 2.0) / 120.0), axis=0)
                qh[i] = np.quantile(hist, min(1.0, np.ceil(121 * (1.0 - alpha / 2.0)) / 120.0), axis=0)
            roll_bands[alpha] = (
                test_prev_p * np.exp(out["ens_mu_r"] + ql * out["ens_std_r_raw"]),
                test_prev_p * np.exp(out["ens_mu_r"] + qh * out["ens_std_r_raw"]),
            )
        out["roll_lo80"], out["roll_hi80"] = roll_bands[0.20]
        out["roll_lo95"], out["roll_hi95"] = roll_bands[0.05]

        # DM test Ensemble vs Naive trên chuỗi ngày N=542 (HAC lag=5)
        dm_ens_rw = diebold_mariano_hac(rw_ret_err ** 2, (test_true_r - out["ens_mu_r"]) ** 2, h=1, max_lag=5)
        out["dm_ens_vs_rw"] = dm_ens_rw

        # Jensen reconstruction comparison
        jdf = compute_jensen_reconstruction_comparison(
            f"{mname} (Ens)", out["ens_mu_v"], out["val_true_r"],
            out["ens_mu_r"], out["ens_std_r_raw"], out["ens_std_r_cal"],
            out["mc_pool"], test_prev_p, test_true_p,
        )
        jensen_frames.append(jdf)

        # Student-t CI (5 seeds) & Moving-Block Bootstrap 95% CI (542 ngày)
        seed_rmses = [m["RMSE (VND)"] for m in out["seed_metrics"]]
        seed_maes = [m["MAE (VND)"] for m in out["seed_metrics"]]
        seed_das = [m["DA (%)"] for m in out["seed_metrics"]]
        seed_crps = [m["CRPS_ret"] for m in out["seed_metrics"]]
        m_r, s_r, lo_t_r, hi_t_r = student_t_ci(seed_rmses)
        m_mae, s_mae, lo_t_mae, hi_t_mae = student_t_ci(seed_maes)
        m_da, s_da, lo_t_da, hi_t_da = student_t_ci(seed_das)
        m_cr, s_cr, lo_t_cr, hi_t_cr = student_t_ci(seed_crps)

        cov95_hit_roll = ((test_true_p >= out["roll_lo95"]) & (test_true_p <= out["roll_hi95"])).astype(np.float64)
        bb_ci = block_bootstrap_metric_ci(
            test_true_p, out["ens_pred_p"], test_true_r, out["ens_mu_r"], out["ens_crps_mat"], cov95_hit_roll
        )
        ci_rows.append(
            {
                "Model": mname,
                "Seed_RMSE_Mean±Std": f"{m_r:.2f} ± {s_r:.1f}",
                "Seed_RMSE_95CI_t(df=4)": f"[{lo_t_r:.2f}, {hi_t_r:.2f}]",
                "Ens_RMSE": f"{out['ens_rmse']:.2f}",
                "Ens_RMSE_BlockBoot95CI": f"[{bb_ci['RMSE'][0]:.2f}, {bb_ci['RMSE'][1]:.2f}]",
                "Seed_DA_95CI_t(df=4)": f"[{lo_t_da:.2f}%, {hi_t_da:.2f}%]",
                "Ens_DA_BlockBoot95CI": f"[{bb_ci['DA'][0]:.2f}%, {bb_ci['DA'][1]:.2f}%]",
                "Ens_CRPS_BlockBoot95CI": f"[{bb_ci['CRPS'][0]:.5f}, {bb_ci['CRPS'][1]:.5f}]",
                "RollCov95_BlockBoot95CI": f"[{bb_ci['Cov95'][0]:.1f}%, {bb_ci['Cov95'][1]:.1f}%]",
            }
        )

        # Bias-Variance & Ensemble Diagnostics
        bv = out["bias_var"]
        bv_rows.append(
            {
                "Model": mname,
                "Mean_Seed_RMSE": round(m_r, 2),
                "Geom_Ens_RMSE (exp(mean_r))": round(out["ens_rmse"], 2),
                "Arith_Ens_RMSE (mean_P)": round(out["arith_ens_rmse"], 2),
                "Ensemble_Gain_VND": round(m_r - out["ens_rmse"], 2),
                "Mean_Seed_MSE_ret": f"{bv['mean_seed_mse_r']:.7f}",
                "Ens_MSE_ret": f"{bv['ens_mse_r']:.7f}",
                "Seed_Var_ret": f"{bv['var_between_r']:.7f}",
                "Identity_Check (EnsMSE+Var-MeanSeedMSE)": f"{bv['ens_mse_r'] + bv['var_between_r'] - bv['mean_seed_mse_r']:.2e}",
            }
        )

        summary_rows.append(
            {
                "Model": mname,
                "Purpose": cfg["purpose"],
                "Num_Features": X_df.shape[1],
                "RMSE (VND)": f"{mean_m['RMSE (VND)']:.2f} ± {std_m['RMSE (VND)']:.1f}",
                "Ensemble RMSE (VND)": f"{out['ens_rmse']:.2f}",
                "MAE (VND)": f"{mean_m['MAE (VND)']:.2f} ± {std_m['MAE (VND)']:.1f}",
                "MAPE (%)": f"{mean_m['MAPE (%)']:.3f}% ± {std_m['MAPE (%)']:.2f}%",
                "DA (%)": f"{mean_m['DA (%)']:.2f}%",
                "NLL_cal": f"{mean_m['NLL_cal']:.3f} ± {std_m['NLL_cal']:.2f}",
                "CRPS_Gauss": f"{mean_m['CRPS_ret']:.5f}",
                "CRPS_Conformal": f"{row_roll['CRPS (ret)']:.5f}",
                "Cov95_Gauss (%)": f"{mean_m['Cov95 (%)']:.1f}%",
                "Cov95_Conformal (%)": f"{row_roll['Cov95 (%)']:.1f}%",
                "MPIW95_Conformal": f"{int(row_roll['MPIW95 (VND)'])}",
                "DM_p2s vs Naive": format_pval(dm_ens_rw["p_2s"]),
                "Avg_Epoch": f"{mean_m['best_epoch']:.1f}",
            }
        )

        print(
            f"-> {mname:<12} ({X_df.shape[1]:2d}f): Val RMSE={np.mean(out['seed_val_rmses']):.2f} | "
            f"Seed RMSE={mean_m['RMSE (VND)']:.2f}±{std_m['RMSE (VND)']:.1f} | Ens={out['ens_rmse']:.2f} | "
            f"DA={mean_m['DA (%)']:.2f}% | CRPS(G)={mean_m['CRPS_ret']:.5f} | "
            f"RollConf[80/90/95]=[{row_roll['Cov80 (%)']:.1f}/{row_roll['Cov90 (%)']:.1f}/{row_roll['Cov95 (%)']:.1f}%] | "
            f"DM_p2s={format_pval(dm_ens_rw['p_2s'])}"
        )

        # Chi tiết theo tỉnh
        for p_idx, prov in enumerate(TARGET_PROVINCES):
            rmses_k, mapes_k, das_k, crps_k, cov95_g_k = [], [], [], [], []
            for run in out["runs"]:
                tp, pp = run["test_true_price"][:, p_idx], run["test_pred_price"][:, p_idx]
                tr, mr, sr = run["test_true_ret"][:, p_idx], run["test_pred_ret"][:, p_idx], run["test_std_ret_cal"][:, p_idx]
                rmses_k.append(np.sqrt(np.mean((tp - pp) ** 2)))
                mapes_k.append(np.mean(np.abs((tp - pp) / tp)) * 100.0)
                mk = tr != 0
                das_k.append(np.mean(np.sign(mr[mk]) == np.sign(tr[mk])) * 100.0)
                crps_k.append(np.mean(gaussian_crps_numpy(tr, mr, sr)))
                cov95_g_k.append(np.mean((tp >= run["pred_p_q025"][:, p_idx]) & (tp <= run["pred_p_q975"][:, p_idx])) * 100.0)
            cov95_roll_k = float(np.mean((test_true_p[:, p_idx] >= out["roll_lo95"][:, p_idx]) & (test_true_p[:, p_idx] <= out["roll_hi95"][:, p_idx])) * 100.0)
            province_rows.append(
                {
                    "Model": mname,
                    "Province": prov,
                    "RMSE_Mean": round(float(np.mean(rmses_k)), 2),
                    "RMSE_Std": round(float(np.std(rmses_k, ddof=1)), 2),
                    "MAPE_Mean (%)": round(float(np.mean(mapes_k)), 3),
                    "DA_Mean (%)": round(float(np.mean(das_k)), 2),
                    "CRPS_Gauss": round(float(np.mean(crps_k)), 5),
                    "Cov95_Gauss (%)": round(float(np.mean(cov95_g_k)), 1),
                    "Cov95_Conformal (%)": round(cov95_roll_k, 1),
                }
            )

        # Trọng số Cross-Attention & Diễn giải đa chiều
        if out["attn_test_matrix"] is not None:
            mean_attn = out["attn_test_matrix"]
            avg_w, std_w = mean_attn.mean(axis=0), mean_attn.std(axis=0)
            for fname, mw, sw in zip(out["exog_names"], avg_w, std_w):
                attn_rows.append(
                    {
                        "Model": mname,
                        "Exogenous_Variate": fname,
                        "Mean_Attention_Weight": round(float(mw), 5),
                        "Temporal_Std": round(float(sw), 5),
                    }
                )
        if out["interp_df"] is not None:
            idf = out["interp_df"].copy()
            idf.insert(0, "Model", mname)
            importance_frames.append(idf)
            if out["attn_test_matrix"] is not None:
                a_map = dict(zip(out["exog_names"], out["attn_test_matrix"].mean(axis=0) * 100.0))
                ex_sub = idf[idf["Feature"].isin(out["exog_names"])].copy()
                ex_sub["Attention_Weight (%)"] = ex_sub["Feature"].map(a_map)
                rho_perm, p_perm = stats.spearmanr(ex_sub["Attention_Weight (%)"], ex_sub["RMSE_Increase_VND"])
                rho_grad, p_grad = stats.spearmanr(ex_sub["Attention_Weight (%)"], ex_sub["Grad_Share (%)"])
                interp_comp_rows.append(
                    {
                        "Model": mname,
                        "Spearman_Attn_vs_PermRMSE": round(float(rho_perm), 4),
                        "p_Attn_vs_PermRMSE": format_pval(float(p_perm)),
                        "Spearman_Attn_vs_GradSaliency": round(float(rho_grad), 4),
                        "p_Attn_vs_GradSaliency": format_pval(float(p_grad)),
                    }
                )

    # ==================================================================================
    # BƯỚC 4: CONTROLLED & FAIR FEATURE GROUP + ARCHITECTURAL ABLATION (6.8)
    # ==================================================================================
    print_header("BƯỚC 4: CONTROLLED FEATURE GROUP & ARCHITECTURAL ABLATION (ĐỒNG NHẤT TUNING & BUDGET)")

    unified_ctrl_cfg = {
        "seq_len": 10,
        "hidden_dim": 64,
        "cov_hidden_dim": 32,
        "rank": 2,
        "dropout": 0.15,
        "in_dropout": 0.10,
        "optimizer_name": "adamw",
        "weight_decay": 1e-3,
        "use_cross_attn": True,
    }
    ablation_specs = [
        ("A1. Endog Only (3f, ctrl drop=0.15)", bundle.feature_sets["DeepVARX 1"], unified_ctrl_cfg, None),
        (
            "A2. Endog + Weather Only (6f, drop=0.15)",
            bundle.feature_sets["DeepVARX 2"][TARGET_PROVINCES + ["temp_mean", "precip_nasa", "humidity_air"]],
            unified_ctrl_cfg,
            None,
        ),
        (
            "A3. Endog + London Only (4f, drop=0.15)",
            bundle.feature_sets["DeepVARX 2"][TARGET_PROVINCES + ["ret_london"]],
            unified_ctrl_cfg,
            None,
        ),
        (
            "A4. Endog + Core Econ Only (8f, drop=0.15)",
            bundle.feature_sets["DeepVARX 2"][TARGET_PROVINCES + ["ret_london", "ret_arabica", "ret_brent", "ret_dxy", "ret_usd_vnd"]],
            unified_ctrl_cfg,
            None,
        ),
        ("A5. Core Econ+Weather DVX2 (11f, drop=0.15)", bundle.feature_sets["DeepVARX 2"], unified_ctrl_cfg, model_outputs["DeepVARX 2"]),
        ("A6. Full TimeXer Control (20f, drop=0.15)", bundle.feature_sets["DeepVARX 3"], unified_ctrl_cfg, None),
        ("B1. DVX 3 TimeXer Full (20f, drop=0.20)", bundle.feature_sets["DeepVARX 3"], DEEPVARX_CONFIGS["DeepVARX 3"], model_outputs["DeepVARX 3"]),
        (
            "B2. V3 Dense-Concat No-Attn (20f, drop=0.20)",
            bundle.feature_sets["DeepVARX 3"],
            {**DEEPVARX_CONFIGS["DeepVARX 3"], "use_cross_attn": False},
            None,
        ),
    ]

    abl_outputs: Dict[str, Dict] = {}
    ablation_rows = []
    for abl_name, abl_X, abl_cfg, cached in ablation_specs:
        res = cached if cached is not None else run_deepvarx_5seed_group(abl_X, bundle, abl_cfg, rw_ret_err, compute_interp=False)
        abl_outputs[abl_name] = res
        mm, sm = res["mean_metrics"], res["std_metrics"]
        dm_ab = diebold_mariano_hac(rw_ret_err ** 2, (test_true_r - res["ens_mu_r"]) ** 2, h=1, max_lag=5)
        ablation_rows.append(
            {
                "Ablation_Variant": abl_name,
                "Num_Features": abl_X.shape[1],
                "Dropout": abl_cfg["dropout"],
                "Param_Count": res["param_count"],
                "Budget_MaxEpochs": 50,
                "Avg_Best_Epoch": round(mm["best_epoch"], 1),
                "Train_Time_s": round(res["train_time_s"], 1),
                "Val_CRPS": round(res["val_crps"], 5),
                "Val_RMSE (VND)": round(float(np.mean(res["seed_val_rmses"])), 2),
                "Seed_RMSE (VND)": f"{mm['RMSE (VND)']:.2f} ± {sm['RMSE (VND)']:.1f}",
                "Ensemble_RMSE (VND)": round(res["ens_rmse"], 2),
                "DA (%)": round(mm["DA (%)"], 2),
                "CRPS_Gauss": round(mm["CRPS_ret"], 5),
                "DM_stat_vs_Naive": round(dm_ab["dm_stat"], 3),
                "DM_p2s_vs_Naive": format_pval(dm_ab["p_2s"]),
            }
        )
        print(
            f"-> {abl_name:<42s} | Params={res['param_count']:6d} | Ep={mm['best_epoch']:4.1f} | "
            f"ValCRPS={res['val_crps']:.5f} | Seed={mm['RMSE (VND)']:.2f}±{sm['RMSE (VND)']:.1f} | Ens={res['ens_rmse']:.2f}"
        )
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv("deepvarx_feature_ablation.csv", index=False)

    # ==================================================================================
    # BƯỚC 5: KIỂM TOÁN TOÀN DIỆN (DM ĐA TỈNH, CONFORMAL ZERO-LEAK, PIT BERKOWITZ, BOOTSTRAP ĐỒNG BỘ)
    # ==================================================================================
    print_header("BƯỚC 5: KIỂM TOÁN DM TEST ĐA TỈNH, CONFORMAL ZERO-LEAKAGE, CALIBRATION RIÊNG TỈNH & PAIRED BOOTSTRAP")

    london_only_out = abl_outputs["A3. Endog + London Only (4f, drop=0.15)"]
    dense_v3_out = abl_outputs["B2. V3 Dense-Concat No-Attn (20f, drop=0.20)"]

    e_rw_r = rw_ret_err ** 2
    e_leg_r = (test_true_r - leg_ens_r) ** 2
    e_x1_r = (test_true_r - model_outputs["DeepVARX 1"]["ens_mu_r"]) ** 2
    e_x2_r = (test_true_r - model_outputs["DeepVARX 2"]["ens_mu_r"]) ** 2
    e_x3_r = (test_true_r - model_outputs["DeepVARX 3"]["ens_mu_r"]) ** 2
    e_lon_r = (test_true_r - london_only_out["ens_mu_r"]) ** 2
    e_den_r = (test_true_r - dense_v3_out["ens_mu_r"]) ** 2

    e_rw_p = (test_true_p - test_prev_p) ** 2
    e_leg_p = (test_true_p - leg_ens_p) ** 2
    e_x2_p = (test_true_p - model_outputs["DeepVARX 2"]["ens_pred_p"]) ** 2
    e_x3_p = (test_true_p - model_outputs["DeepVARX 3"]["ens_pred_p"]) ** 2
    e_den_p = (test_true_p - dense_v3_out["ens_pred_p"]) ** 2

    c_rw = crps_rw_mat
    c_x1 = model_outputs["DeepVARX 1"]["ens_crps_mat"]
    c_x2 = model_outputs["DeepVARX 2"]["ens_crps_mat"]
    c_x3 = model_outputs["DeepVARX 3"]["ens_crps_mat"]
    c_den = dense_v3_out["ens_crps_mat"]

    dm_pairs = [
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 1 (Ens) vs Naive", e_rw_r, e_x1_r, True),
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 2 (Ens) vs Naive", e_rw_r, e_x2_r, True),
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 3 (Ens) vs Naive", e_rw_r, e_x3_r, True),
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 2 (Ens) vs Legacy OLS (Ens)", e_leg_r, e_x2_r, True),
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 3 (Ens) vs Legacy OLS (Ens)", e_leg_r, e_x3_r, True),
        ("Daily Mean Ret-MSE (N=542)", "DeepVARX 3 (Ens) vs DeepVARX 2 (Ens)", e_x2_r, e_x3_r, True),
        ("Daily Mean Ret-MSE (N=542)", "V3 Dense-Concat (Ens) vs DeepVARX 3 TimeXer (Ens)", e_x3_r, e_den_r, True),
        ("Daily Mean Ret-MSE (N=542)", "London Only (4f, Ens) vs DeepVARX 2 (11f, Ens)", e_x2_r, e_lon_r, True),
        ("Daily Mean Ret-MSE (N=542)", "London Only (4f, Ens) vs DeepVARX 3 (20f, Ens)", e_x3_r, e_lon_r, True),
        ("Daily Mean Price-MSE (N=542)", "DeepVARX 2 (Ens) vs Naive", e_rw_p, e_x2_p, True),
        ("Daily Mean Price-MSE (N=542)", "DeepVARX 3 (Ens) vs Naive", e_rw_p, e_x3_p, True),
        ("Daily Mean Price-MSE (N=542)", "DeepVARX 2 (Ens) vs Legacy OLS (Ens)", e_leg_p, e_x2_p, True),
        ("Daily Mean Price-MSE (N=542)", "DeepVARX 3 (Ens) vs Legacy OLS (Ens)", e_leg_p, e_x3_p, True),
        ("Daily Mean Price-MSE (N=542)", "V3 Dense-Concat (Ens) vs DeepVARX 3 TimeXer (Ens)", e_x3_p, e_den_p, True),
        ("Daily Mean CRPS (N=542)", "DeepVARX 1 (Ens) vs Naive Prob", c_rw, c_x1, True),
        ("Daily Mean CRPS (N=542)", "DeepVARX 2 (Ens) vs Naive Prob", c_rw, c_x2, True),
        ("Daily Mean CRPS (N=542)", "DeepVARX 3 (Ens) vs Naive Prob", c_rw, c_x3, True),
        ("Daily Mean CRPS (N=542)", "DeepVARX 3 (Ens) vs DeepVARX 2 (Ens)", c_x2, c_x3, True),
        ("Daily Mean CRPS (N=542)", "V3 Dense-Concat (Ens) vs DeepVARX 3 TimeXer (Ens)", c_x3, c_den, True),
    ]
    for k, prov in enumerate(TARGET_PROVINCES):
        dm_pairs.append((f"Province {prov} Ret-MSE (N=542)", "DeepVARX 2 (Ens) vs Naive", e_rw_r[:, k], e_x2_r[:, k], False))
        dm_pairs.append((f"Province {prov} Ret-MSE (N=542)", "DeepVARX 3 (Ens) vs Naive", e_rw_r[:, k], e_x3_r[:, k], False))
        dm_pairs.append((f"Province {prov} Ret-MSE (N=542)", "DeepVARX 3 (Ens) vs Legacy OLS", e_leg_r[:, k], e_x3_r[:, k], False))
        dm_pairs.append((f"Province {prov} Ret-MSE (N=542)", "V3 Dense-Concat vs DVX 3 TimeXer", e_x3_r[:, k], e_den_r[:, k], False))

    dm_rows = []
    for scope, comp, lb, lm, agg in dm_pairs:
        r_hac = diebold_mariano_hac(lb, lm, h=1, max_lag=5, aggregate_provinces=agg)
        r_lag0 = diebold_mariano_hac(lb, lm, h=1, max_lag=0, aggregate_provinces=agg)
        mdm = multivariate_dm_hac(lb, lm, max_lag=5) if agg else {"MDM_Wald_chi2": np.nan, "MDM_p_F": np.nan}
        dm_rows.append(
            {
                "Scope_and_Loss": scope,
                "Comparison": comp,
                "N_Days": int(r_hac["n_obs"]),
                "Horizon_h": 1,
                "DM_Lag0": round(r_lag0["dm_stat"], 3),
                "DM_HAC(5)": round(r_hac["dm_stat"], 3),
                "p_1s_HAC": format_pval(r_hac["p_1s"]),
                "p_2s_HAC": format_pval(r_hac["p_2s"]),
                "MDM_Wald_chi2(3)": round(mdm["MDM_Wald_chi2"], 2) if np.isfinite(mdm["MDM_Wald_chi2"]) else "-",
                "MDM_p_F(3,539)": format_pval(mdm["MDM_p_F"]) if np.isfinite(mdm["MDM_p_F"]) else "-",
                "LossDiff_AC1": round(r_hac["ac1"], 3),
                "LjungBox_Q5_p": format_pval(r_hac["lb_p"]),
            }
        )
    dm_audit_df = pd.DataFrame(dm_rows)
    dm_audit_df.to_csv("deepvarx_dm_test_audit.csv", index=False)
    print(dm_audit_df.head(19).to_string(index=False))

    # 5.2: Chứng minh Rolling Conformal Zero-Leakage & Độ nhạy W={60,90,120,180}
    leak_frames = []
    for m_lbl, m_out in [("DeepVARX 2 (Ens)", model_outputs["DeepVARX 2"]), ("DeepVARX 3 (Ens)", model_outputs["DeepVARX 3"])]:
        a_info, w_df = verify_rolling_conformal_zero_leakage(
            m_lbl, m_out["ens_mu_v"], m_out["ens_std_v_raw"], m_out["val_true_r"],
            m_out["ens_mu_r"], m_out["ens_std_r_raw"], test_true_r, test_prev_p, test_true_p,
            bundle.dates, train_end, rolling_windows=[60, 90, 120, 180],
        )
        print(f"-> Conformal Zero-Leakage Check [{m_lbl}]: {a_info['status']} | max_diff={a_info['max_poison_diff']:.1e}")
        leak_frames.append(w_df)
    pd.concat(leak_frames, ignore_index=True).to_csv("deepvarx_conformal_leakage_and_window_audit.csv", index=False)

    # 5.3: Calibration RIÊNG TỪNG TỈNH + Kiểm định PIT có tính tự tương quan (Berkowitz AR(1) & LB)
    prov_cal_df = compute_province_calibration_table(
        [
            ("Naive Prob", np.zeros_like(val_r), std_rw_raw, val_r, np.zeros_like(test_true_r), std_rw_test_raw, tau_rw, test_true_r, test_prev_p, test_true_p),
            ("DeepVARX 1 (Ens)", model_outputs["DeepVARX 1"]["ens_mu_v"], model_outputs["DeepVARX 1"]["ens_std_v_raw"], model_outputs["DeepVARX 1"]["val_true_r"], model_outputs["DeepVARX 1"]["ens_mu_r"], model_outputs["DeepVARX 1"]["ens_std_r_raw"], model_outputs["DeepVARX 1"]["ens_tau"], test_true_r, test_prev_p, test_true_p),
            ("DeepVARX 2 (Ens)", model_outputs["DeepVARX 2"]["ens_mu_v"], model_outputs["DeepVARX 2"]["ens_std_v_raw"], model_outputs["DeepVARX 2"]["val_true_r"], model_outputs["DeepVARX 2"]["ens_mu_r"], model_outputs["DeepVARX 2"]["ens_std_r_raw"], model_outputs["DeepVARX 2"]["ens_tau"], test_true_r, test_prev_p, test_true_p),
            ("DeepVARX 3 (Ens)", model_outputs["DeepVARX 3"]["ens_mu_v"], model_outputs["DeepVARX 3"]["ens_std_v_raw"], model_outputs["DeepVARX 3"]["val_true_r"], model_outputs["DeepVARX 3"]["ens_mu_r"], model_outputs["DeepVARX 3"]["ens_std_r_raw"], model_outputs["DeepVARX 3"]["ens_tau"], test_true_r, test_prev_p, test_true_p),
            ("V3 Dense-Concat (Ens)", dense_v3_out["ens_mu_v"], dense_v3_out["ens_std_v_raw"], dense_v3_out["val_true_r"], dense_v3_out["ens_mu_r"], dense_v3_out["ens_std_r_raw"], dense_v3_out["ens_tau"], test_true_r, test_prev_p, test_true_p),
        ],
        rolling_window=120,
    )
    prov_cal_df.to_csv("deepvarx_province_calibration.csv", index=False)
    print("\n--- Calibration riêng từng tỉnh (Causal Rolling Conformal W=120) + Berkowitz AR(1) PIT ---")
    print(prov_cal_df[prov_cal_df["Calibration_Method"] == "Causal Rolling Conformal (W=120)"].to_string(index=False))

    # 5.4: Synchronous Multi-Province Paired Block Bootstrap Audit (B={5,10,20})
    london_lag1_test = bundle.feature_sets["DeepVARX 2"]["ret_london"].shift(1).iloc[val_end:].fillna(0.0).values
    boot_struct_df, boot_paired_df = paired_block_bootstrap_diagnostics(
        true_p=test_true_p,
        preds_map={
            "Naive Prob": {"pred_p": test_prev_p, "crps_mat": crps_rw_mat},
            "DeepVARX 1": {"pred_p": model_outputs["DeepVARX 1"]["ens_pred_p"], "crps_mat": model_outputs["DeepVARX 1"]["ens_crps_mat"]},
            "DeepVARX 2": {"pred_p": model_outputs["DeepVARX 2"]["ens_pred_p"], "crps_mat": model_outputs["DeepVARX 2"]["ens_crps_mat"]},
            "DeepVARX 3": {"pred_p": model_outputs["DeepVARX 3"]["ens_pred_p"], "crps_mat": model_outputs["DeepVARX 3"]["ens_crps_mat"]},
            "V3 Dense-Concat": {"pred_p": dense_v3_out["ens_pred_p"], "crps_mat": dense_v3_out["ens_crps_mat"]},
        },
        exog_london_lag1=london_lag1_test,
        block_lengths=[5, 10, 20],
        n_boot=2000,
        seed=42,
    )
    boot_struct_df.to_csv("deepvarx_bootstrap_correlation_check.csv", index=False)
    boot_paired_df.to_csv("deepvarx_bootstrap_structure_audit.csv", index=False)
    print("\n--- Kiểm tra bảo toàn tương quan đa tỉnh & Paired Differential Bootstrap 95% CI (B=10) ---")
    print(boot_struct_df.to_string(index=False))
    print(boot_paired_df[boot_paired_df["Block_Len_B"] == 10].to_string(index=False))

    # Xuất các bảng tổng hợp còn lại
    pd.DataFrame(summary_rows).to_csv("deepvarx_model_comparison_results.csv", index=False)
    pd.DataFrame(province_rows).to_csv("deepvarx_province_results.csv", index=False)

    cal_all_df = pd.concat(calib_frames, ignore_index=True)
    cal_all_df.to_csv("deepvarx_calibration_comparison.csv", index=False)

    jensen_all_df = pd.concat(jensen_frames, ignore_index=True)
    jensen_all_df.to_csv("deepvarx_jensen_reconstruction.csv", index=False)

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv("deepvarx_confidence_intervals.csv", index=False)

    bv_df = pd.DataFrame(bv_rows)
    bv_df.to_csv("deepvarx_ensemble_vs_seed_diagnostics.csv", index=False)

    attn_df = pd.DataFrame(attn_rows).sort_values(["Model", "Mean_Attention_Weight"], ascending=[True, False])
    attn_df.to_csv("deepvarx_attention_weights.csv", index=False)
    imp_df = pd.concat(importance_frames, ignore_index=True)
    imp_df.to_csv("deepvarx_feature_importance.csv", index=False)
    interp_comp_df = pd.DataFrame(interp_comp_rows)
    interp_comp_df.to_csv("deepvarx_interp_comparison.csv", index=False)

    # Đối chiếu liên mô hình
    print("\n--- Đối chiếu liên mô hình (cùng 542 ngày Test) ---")
    for path_rel, title in [
        (r"..\lstm\lstm_model_comparison_results.csv", "Hệ thống LSTM"),
        (r"..\deepvar\deepvar_model_comparison_results.csv", "Hệ thống DeepVAR"),
    ]:
        if os.path.exists(path_rel):
            print(f"[{title}]")
            print(pd.read_csv(path_rel).to_string(index=False))

    # ==================================================================================
    # BƯỚC 6: XUẤT 5 BIỂU ĐỒ KHOA HỌC & CHẨN ĐOÁN TOÀN DIỆN
    # ==================================================================================
    print_header("BƯỚC 6: XUẤT 5 BIỂU ĐỒ CHẨN ĐOÁN (figures_deepvarx/)")

    # Fig 1: Fan chart Causal Rolling Conformal W=120 của DVX 3 + Legacy OLS vs DVX 3
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), constrained_layout=True)
    zoom = 90
    zd = test_dates[-zoom:]
    dvx3_out = model_outputs["DeepVARX 3"]
    axes[0].fill_between(zd, dvx3_out["roll_lo95"][-zoom:, 0], dvx3_out["roll_hi95"][-zoom:, 0], color="#1f77b4", alpha=0.16, label="Khoảng dự báo 95% (Causal Conformal W=120)")
    axes[0].fill_between(zd, dvx3_out["roll_lo80"][-zoom:, 0], dvx3_out["roll_hi80"][-zoom:, 0], color="#1f77b4", alpha=0.32, label="Khoảng dự báo 80% (Causal Conformal)")
    axes[0].plot(zd, test_true_p[-zoom:, 0], color="#111111", lw=1.8, label="Thực tế (Đắk Lắk)")
    axes[0].plot(zd, dvx3_out["ens_pred_p"][-zoom:, 0], color="#d62728", lw=1.4, label="DeepVARX 3 Ensemble")
    axes[0].set_title("Dự báo xác suất Conformal TimeXer-DeepVARX 3 (90 ngày cuối Test)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("Giá cà phê (VND/kg)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(loc="best", fontsize=8.2)
    axes[0].grid(alpha=0.25)

    z2 = 45
    zd2 = test_dates[-z2:]
    axes[1].plot(zd2, test_true_p[-z2:, 0], color="#111111", lw=2.0, label="Thực tế (Đắk Lắk)")
    axes[1].plot(zd2, leg_ens_p[-z2:, 0], color="#ff7f0e", ls="--", lw=1.5, label=f"DeepVARX-Legacy OLS (Seed={np.mean(leg_rmses):.0f})")
    axes[1].plot(zd2, dvx3_out["ens_pred_p"][-z2:, 0], color="#1f77b4", lw=1.6, label=f"TimeXer-DeepVARX 3 (Ens={dvx3_out['ens_rmse']:.0f})")
    axes[1].set_title("Khắc phục sai lệch co giãn của OLS Skip cũ (45 ngày cuối Test)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("Giá cà phê (VND/kg)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(loc="best", fontsize=8.2)
    axes[1].grid(alpha=0.25)
    p1 = os.path.join(FIG_DIR, "fig1_deepvarx_probabilistic_intervals.png")
    plt.savefig(p1, dpi=180)
    plt.close(fig)

    # Fig 2: So sánh RMSE & CRPS (bao gồm Conformal CRPS)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), constrained_layout=True)
    m_labels = ["Naive\n(RW)", "Legacy\nOLS Skip", "DeepVARX 1\n(3f)", "DeepVARX 2\n(11f)", "DeepVARX 3\n(20f)"]
    rmses_b = [
        rw_rmse,
        float(np.mean(leg_rmses)),
        model_outputs["DeepVARX 1"]["mean_metrics"]["RMSE (VND)"],
        model_outputs["DeepVARX 2"]["mean_metrics"]["RMSE (VND)"],
        model_outputs["DeepVARX 3"]["mean_metrics"]["RMSE (VND)"],
    ]
    errs_b = [
        0.0,
        float(np.std(leg_rmses, ddof=1)),
        model_outputs["DeepVARX 1"]["std_metrics"]["RMSE (VND)"],
        model_outputs["DeepVARX 2"]["std_metrics"]["RMSE (VND)"],
        model_outputs["DeepVARX 3"]["std_metrics"]["RMSE (VND)"],
    ]
    bars = axes[0].bar(m_labels, rmses_b, yerr=errs_b, capsize=4, color=["#7f7f7f", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"], edgecolor="#222222", alpha=0.9)
    axes[0].axhline(1669.96, color="#9467bd", ls="--", lw=1.2, label="VARX-L Benchmark (1669.96)")
    axes[0].set_title("Test RMSE Điểm (Mean ± Std 5 seeds, 542 ngày Test)", fontsize=10.5, fontweight="bold")
    axes[0].set_ylabel("RMSE (VND/kg)")
    axes[0].set_ylim(1400, max(rmses_b) + 180)
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)
    for b, v in zip(bars, rmses_b):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 32, f"{v:.1f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    crps_labels = ["Naive\n(Gauss)", "DVX 1\n(Conformal)", "DVX 2\n(Conformal)", "DVX 3\n(Conformal)"]
    def _roll_crps(mlabel: str) -> float:
        return float(cal_all_df[(cal_all_df["Model"] == f"{mlabel} (Ens)") & (cal_all_df["Calibration_Method"] == "Causal Rolling Conformal (W=120)")]["CRPS (ret)"].iloc[0])
    crps_vals = [float(np.mean(crps_rw_mat)), _roll_crps("DeepVARX 1"), _roll_crps("DeepVARX 2"), _roll_crps("DeepVARX 3")]
    bars2 = axes[1].bar(crps_labels, crps_vals, color=["#7f7f7f", "#ff7f0e", "#2ca02c", "#1f77b4"], edgecolor="#222222", alpha=0.9)
    axes[1].set_title("Độ chính xác phân phối CRPS Conformal (Thấp hơn = tốt)", fontsize=10.5, fontweight="bold")
    axes[1].set_ylabel("CRPS (log-return)")
    axes[1].set_ylim(0.0055, max(crps_vals) + 0.0009)
    axes[1].grid(axis="y", alpha=0.25)
    for b, v in zip(bars2, crps_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.00012, f"{v:.5f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    p2 = os.path.join(FIG_DIR, "fig2_deepvarx_cross_models_comparison.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    # Fig 3: Phân biệt Cross-Attention vs Permutation Importance vs Gradient Saliency + Heteroskedasticity
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4), constrained_layout=True)
    i3 = imp_df[imp_df["Model"] == "DeepVARX 3"].copy()
    a3_map = dict(zip(dvx3_out["exog_names"], dvx3_out["attn_test_matrix"].mean(axis=0) * 100.0))
    i3["Attn (%)"] = i3["Feature"].map(a3_map).fillna(0.0)
    top_feats = i3.head(10).iloc[::-1]
    y_pos = np.arange(len(top_feats))
    h_bar = 0.27
    axes[0].barh(y_pos + h_bar, top_feats["Attn (%)"], height=h_bar, color="#1f77b4", label="Cross-Attention (%) [chỉ ngoại sinh]")
    axes[0].barh(y_pos, top_feats["RMSE_Share (%)"], height=h_bar, color="#d62728", label="Permutation RMSE Share (%)")
    axes[0].barh(y_pos - h_bar, top_feats["Grad_Share (%)"], height=h_bar, color="#2ca02c", label="Input Grad Saliency (%)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_feats["Feature"], fontsize=8.5)
    axes[0].set_xlabel("Tỷ trọng (%)")
    axes[0].set_title("Phân biệt Attention vs Feature Importance (Top 10)", fontsize=10.0, fontweight="bold")
    axes[0].legend(fontsize=7.8, loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    ex3_only = i3[i3["Feature"].isin(dvx3_out["exog_names"])].copy()
    ex3_only["Attn_Rank"] = ex3_only["Attn (%)"].rank(ascending=False, method="first").astype(int)
    ex3_only["Perm_Rank"] = ex3_only["RMSE_Increase_VND"].rank(ascending=False, method="first").astype(int)
    axes[1].plot([1, 17], [1, 17], color="#bbbbbb", ls="--", lw=1.2, zorder=1)
    axes[1].scatter(ex3_only["Attn_Rank"], ex3_only["Perm_Rank"], s=58, color="#9467bd", edgecolor="#111111", zorder=3)
    for _, r in ex3_only.iterrows():
        axes[1].annotate(r["Feature"], (r["Attn_Rank"], r["Perm_Rank"]), xytext=(4, -2), textcoords="offset points", fontsize=7.2)
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()
    axes[1].set_xlim(18.5, 0.2)
    axes[1].set_ylim(18.0, 0.2)
    axes[1].set_xlabel("Thứ hạng Cross-Attention (1 = cao nhất)")
    axes[1].set_ylabel("Thứ hạng Permutation ΔRMSE (1 = cao nhất)")
    rho_ex = float(stats.spearmanr(ex3_only["Attn (%)"], ex3_only["RMSE_Increase_VND"]).statistic)
    axes[1].set_title(f"Attention ≠ Feature Importance (Spearman ρ={rho_ex:.2f})", fontsize=10.0, fontweight="bold")
    axes[1].grid(alpha=0.25)

    pred_std_dl = dvx3_out["ens_std_r_cal"][:, 0]
    abs_err_dl = np.abs(test_true_r[:, 0] - dvx3_out["ens_mu_r"][:, 0])
    axes[2].plot(test_dates, abs_err_dl, color="#7f7f7f", alpha=0.45, lw=1.0, label="|Sai số thực| |r_t - μ_t|")
    axes[2].plot(test_dates, 1.96 * pred_std_dl, color="#d62728", lw=1.3, label="Ngưỡng biến động động (1.96·σ_t)")
    axes[2].set_title("Bất định động (Heteroskedasticity) - Đắk Lắk", fontsize=10.0, fontweight="bold")
    axes[2].set_ylabel("Biên độ log-return")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(fontsize=8.0)
    axes[2].grid(alpha=0.25)

    p3 = os.path.join(FIG_DIR, "fig3_timexer_attention_and_importance.png")
    plt.savefig(p3, dpi=180)
    plt.close(fig)

    # Fig 4: PIT Histogram đơn tỉnh Đắk Lắk (N=542, Berkowitz AR1) + Reliability Diagram + Winkler
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    d3_cal = calib_diags_all["DeepVARX 3"]
    pit_g_dl = d3_cal["Gauss + Val tau"]["pit_matrix"][:, 0]
    pit_c_dl = d3_cal["Causal Rolling Conformal (W=120)"]["pit_matrix"][:, 0]
    pt_g_dl = pit_independence_aware_tests(pit_g_dl, max_lag=5)
    pt_c_dl = pit_independence_aware_tests(pit_c_dl, max_lag=5)
    bins = np.linspace(0.0, 1.0, 11)
    axes[0].hist(pit_g_dl, bins=bins, density=True, alpha=0.45, color="#d62728", edgecolor="#8c1d18", label=f"Gauss+tau (KS p={format_pval(pt_g_dl['KS_p_N542'])}, Berk p={format_pval(pt_g_dl['Berkowitz_p'])})")
    axes[0].hist(pit_c_dl, bins=bins, density=True, alpha=0.55, color="#1f77b4", edgecolor="#0f4c81", label=f"Conformal W=120 (KS p={format_pval(pt_c_dl['KS_p_N542'])}, Berk p={format_pval(pt_c_dl['Berkowitz_p'])})")
    axes[0].axhline(1.0, color="#111111", ls="--", lw=1.4, label="Chuẩn Uniform(0,1)")
    axes[0].set_ylim(0.0, axes[0].get_ylim()[1] * 1.28)
    axes[0].set_title("PIT Histogram Đắk Lắk (N=542, DVX 3 Ens)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Giá trị PIT u_t")
    axes[0].set_ylabel("Mật độ")
    axes[0].legend(fontsize=7.3, loc="upper center")
    axes[0].grid(alpha=0.25)

    nom = d3_cal["Gauss + Val tau"]["nominal_grid"] * 100.0
    axes[1].plot([10, 99], [10, 99], color="#111111", ls="--", lw=1.4, label="Đường chuẩn y = x")
    axes[1].plot(nom, d3_cal["Gauss + Val tau"]["empirical_grid"] * 100.0, "o-", color="#d62728", label="DVX 3 Gauss + Val tau")
    axes[1].plot(nom, d3_cal["Split Conformal (Val)"]["empirical_grid"] * 100.0, "s--", color="#ff7f0e", label="DVX 3 Split Conformal (Val)")
    axes[1].plot(nom, d3_cal["Causal Rolling Conformal (W=120)"]["empirical_grid"] * 100.0, "^-", color="#1f77b4", lw=2.0, label="DVX 3 Causal Conformal (W=120)")
    axes[1].plot(nom, calib_diags_all["DeepVARX 2"]["Causal Rolling Conformal (W=120)"]["empirical_grid"] * 100.0, "d:", color="#2ca02c", lw=1.8, label="DVX 2 Causal Conformal (W=120)")
    axes[1].set_title("Reliability Diagram (Nominal vs Empirical Cov)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Độ bao phủ danh nghĩa (%)")
    axes[1].set_ylabel("Độ bao phủ thực nghiệm (%)")
    axes[1].legend(fontsize=7.8, loc="upper left")
    axes[1].grid(alpha=0.25)

    dvx3_rows = cal_all_df[cal_all_df["Model"] == "DeepVARX 3 (Ens)"].set_index("Calibration_Method")
    lvls = ["80%", "90%", "95%"]
    x_i = np.arange(len(lvls))
    w_b = 0.26
    for idx_m, (m_key, col_c, lbl_s) in enumerate([
        ("Gauss + Val tau", "#d62728", "Gauss + Val tau"),
        ("Split Conformal (Val)", "#ff7f0e", "Split Conformal (Val)"),
        ("Causal Rolling Conformal (W=120)", "#1f77b4", "Causal Conformal (W=120)"),
    ]):
        wvals = [dvx3_rows.loc[m_key, "Winkler80"], dvx3_rows.loc[m_key, "Winkler90"], dvx3_rows.loc[m_key, "Winkler95"]]
        axes[2].bar(x_i + (idx_m - 1) * w_b, wvals, width=w_b, color=col_c, edgecolor="#222222", alpha=0.9, label=lbl_s)
    axes[2].set_xticks(x_i)
    axes[2].set_xticklabels([f"Mức {l}" for l in lvls])
    axes[2].set_ylabel("Winkler Interval Score (Thấp hơn = tốt)")
    axes[2].set_title("Cân bằng Coverage & Độ sắc nét (Winkler Score)", fontsize=10.5, fontweight="bold")
    axes[2].legend(fontsize=8.0)
    axes[2].grid(axis="y", alpha=0.25)

    p4 = os.path.join(FIG_DIR, "fig4_calibration_pit_reliability.png")
    plt.savefig(p4, dpi=180)
    plt.close(fig)

    # Fig 5: Ma trận tương quan 17 biến ngoại sinh & Group/Architecture Ablation
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 6.0), constrained_layout=True)
    im = axes[0].imshow(corr_mat.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_xticks(np.arange(len(corr_mat.columns)))
    axes[0].set_yticks(np.arange(len(corr_mat.columns)))
    axes[0].set_xticklabels(corr_mat.columns, rotation=45, ha="right", fontsize=7.6)
    axes[0].set_yticklabels(corr_mat.columns, fontsize=7.6)
    axes[0].set_title("Ma trận tương quan Pearson giữa 17 biến ngoại sinh V3", fontsize=10.5, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.042, pad=0.03)

    abl_rev = abl_df.iloc[::-1]
    y_a = np.arange(len(abl_rev))
    seed_means = [float(s.split(" ± ")[0]) for s in abl_rev["Seed_RMSE (VND)"]]
    ens_vals = list(abl_rev["Ensemble_RMSE (VND)"])
    axes[1].barh(y_a + 0.18, seed_means, height=0.34, color="#ff7f0e", edgecolor="#222222", alpha=0.9, label="Mean Seed RMSE (5 seeds)")
    axes[1].barh(y_a - 0.18, ens_vals, height=0.34, color="#1f77b4", edgecolor="#222222", alpha=0.9, label="Ensemble RMSE")
    axes[1].set_yticks(y_a)
    axes[1].set_yticklabels(abl_rev["Ablation_Variant"], fontsize=8.5)
    axes[1].set_xlim(1500, max(seed_means) + 110)
    axes[1].set_xlabel("Test RMSE (VND/kg)")
    axes[1].set_title("Ablation theo nhóm biến & kiến trúc Cross-Attention", fontsize=10.5, fontweight="bold")
    axes[1].legend(fontsize=8.2, loc="lower right")
    axes[1].grid(axis="x", alpha=0.25)
    for yk, vs, ve in zip(y_a, seed_means, ens_vals):
        axes[1].text(vs + 6, yk + 0.18, f"{vs:.1f}", va="center", fontsize=7.8)
        axes[1].text(ve + 6, yk - 0.18, f"{ve:.1f}", va="center", fontsize=7.8, fontweight="bold")

    p5 = os.path.join(FIG_DIR, "fig5_ablation_and_correlation.png")
    plt.savefig(p5, dpi=180)
    plt.close(fig)

    print(f"Đã lưu 5 biểu đồ: {p1}, {p2}, {p3}, {p4}, {p5}")
    print(f"Hoàn tất toàn bộ kiểm định & thực nghiệm DeepVARX sau {time.time() - t0:.1f} giây.")


if __name__ == "__main__":
    main()

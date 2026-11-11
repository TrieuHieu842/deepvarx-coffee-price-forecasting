# -*- coding: utf-8 -*-
"""
========================================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM TOÀN DIỆN MÔ HÌNH VARX VÀ VARX-L
Dự báo đa biến giá thu mua cà phê Robusta Việt Nam (Đắk Lắk, Lâm Đồng, Gia Lai)
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

from varx_coffee_models import (
    TARGET_PROVINCES,
    CoffeeVARX,
    compute_metrics,
    compute_varx_permutation_importance,
    diebold_mariano_from_loss_diff,
    load_and_prepare_coffee_data,
)

sys.stdout.reconfigure(encoding="utf-8")

FIG_DIR = "figures_varx"
os.makedirs(FIG_DIR, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print("=" * 88)
    print("CHƯƠNG TRÌNH THỰC NGHIỆM MÔ HÌNH VARX & VARX-L ĐA BIẾN")
    print("=" * 88)

    bundle = load_and_prepare_coffee_data("data_daily_panel_clean_v2.csv")
    Y = bundle.ret_df.values
    P = bundle.price_df.values
    dates = bundle.dates
    tr_end = bundle.train_end
    va_end = bundle.val_end
    T = len(Y)
    te_idx = np.arange(va_end, T)
    te_dates = dates[te_idx]
    n_test = len(te_idx)

    print(f"Tổng số ngày: {T} | Train: {tr_end} | Val: {va_end - tr_end} | Test: {n_test} ngày ({te_dates[0]:%d/%m/%Y} -> {te_dates[-1]:%d/%m/%Y})")

    # 1. Benchmark Naive Random Walk
    p_true = P[te_idx]
    p_prev = P[te_idx - 1]
    r_true = Y[te_idx]
    r_naive = np.zeros_like(r_true)
    naive_m = compute_metrics(p_true, p_prev, r_true, r_naive)

    results = []
    results.append({
        "Model": "Naive (Random Walk)",
        "Description": "P_hat_t = P_{t-1}",
        "Num_Features": 3,
        "Penalty": "none",
        "RMSE (VND)": naive_m["RMSE (VND)"],
        "MAE (VND)": naive_m["MAE (VND)"],
        "MAPE (%)": f"{naive_m['MAPE (%)']:.3f}%",
        "R2_price": round(naive_m["R2_price"], 4),
        "R2_ret": round(naive_m["R2_ret"], 4),
        "DA (%)": "0.00%",
        "DM_stat vs Naive": "Benchmark",
        "DM_p vs Naive": "Benchmark",
    })

    model_preds: Dict[str, np.ndarray] = {"Naive": p_prev}
    prov_rows = []

    for i, prov in enumerate(TARGET_PROVINCES):
        prov_m = compute_metrics(p_true[:, i], p_prev[:, i], r_true[:, i], r_naive[:, i])
        prov_rows.append({
            "Model": "Naive (Random Walk)",
            "Province": prov,
            "RMSE (VND)": prov_m["RMSE (VND)"],
            "MAE (VND)": prov_m["MAE (VND)"],
            "MAPE (%)": f"{prov_m['MAPE (%)']:.3f}%",
            "DA (%)": "0.00%",
            "DM_p vs Naive": "Benchmark",
        })

    # Cấu hình danh mục mô hình thực nghiệm
    VARX_CONFIGS = [
        ("VAR Baseline", "VAR thuần túy (p=5, không ngoại sinh)", "VAR (No X)", 5, 0, "none", 0.0),
        ("VARX 1", "Giá + Sàn London & Tỷ giá (5 biến, p=5, q=2)", "VARX 1", 5, 2, "none", 0.0),
        ("VARX 2", "11 biến cốt lõi: Giá + Kinh tế + Thời tiết (OLS)", "VARX 2", 5, 2, "none", 0.0),
        ("VARX-L 2 (Ridge)", "11 biến cốt lõi có phạt Ridge (alpha=0.1)", "VARX 2", 5, 2, "ridge", 0.1),
        ("VARX-L 2 (Lasso)", "11 biến cốt lõi có phạt Lasso (alpha=1e-4)", "VARX 2", 5, 2, "lasso", 1e-4),
        ("VARX 3", "Full Exogenous: Giá + Toàn bộ 20 biến (OLS)", "VARX 3", 5, 2, "none", 0.0),
    ]

    var_base_preds = None

    for name, desc, fkey, p, q, pen, alpha in VARX_CONFIGS:
        feat_df = bundle.feature_sets[fkey]
        n_features = feat_df.shape[1]
        exog_df = feat_df.drop(columns=TARGET_PROVINCES) if fkey != "VAR (No X)" else None
        X_full = exog_df.values if exog_df is not None and exog_df.shape[1] > 0 else None
        X_train = X_full[:tr_end] if X_full is not None else None

        mdl = CoffeeVARX(p=p, q=q, penalty=pen, alpha=alpha)
        mdl.fit(Y[:tr_end], X_train)
        pred_ret = mdl.predict_rolling(Y, X_full, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        model_preds[name] = pred_p

        if name == "VAR Baseline":
            var_base_preds = pred_p

        m = compute_metrics(p_true, pred_p, r_true, pred_ret)

        # Diebold-Mariano test vs Naive
        e_bench = (p_true - p_prev) ** 2
        e_model = (p_true - pred_p) ** 2
        d_panel = np.mean(e_bench - e_model, axis=1)
        stat, pval = diebold_mariano_from_loss_diff(d_panel, h=1, alternative="greater")

        results.append({
            "Model": name,
            "Description": desc,
            "Num_Features": n_features,
            "Penalty": pen,
            "RMSE (VND)": m["RMSE (VND)"],
            "MAE (VND)": m["MAE (VND)"],
            "MAPE (%)": f"{m['MAPE (%)']:.3f}%",
            "R2_price": round(m["R2_price"], 4),
            "R2_ret": round(m["R2_ret"], 4),
            "DA (%)": f"{m['DA (%)']:.2f}%",
            "DM_stat vs Naive": round(stat, 3),
            "DM_p vs Naive": round(pval, 4),
        })

        # Thống kê chi tiết từng tỉnh
        for i, prov in enumerate(TARGET_PROVINCES):
            prov_m = compute_metrics(p_true[:, i], pred_p[:, i], r_true[:, i], pred_ret[:, i])
            d_prov = (p_true[:, i] - p_prev[:, i]) ** 2 - (p_true[:, i] - pred_p[:, i]) ** 2
            _, p_prov = diebold_mariano_from_loss_diff(d_prov, h=1, alternative="greater")
            prov_rows.append({
                "Model": name,
                "Province": prov,
                "RMSE (VND)": prov_m["RMSE (VND)"],
                "MAE (VND)": prov_m["MAE (VND)"],
                "MAPE (%)": f"{prov_m['MAPE (%)']:.3f}%",
                "DA (%)": f"{prov_m['DA (%)']:.2f}%",
                "DM_p vs Naive": round(p_prov, 4),
            })

    res_df = pd.DataFrame(results)
    prov_df = pd.DataFrame(prov_rows)

    print("\n--- 2. BẢNG TỔNG HỢP SO SÁNH HIỆU SUẤT CÁC PHIÊN BẢN VARX TRÊN TEST (542 NGÀY) ---")
    print(res_df.to_string(index=False))

    # 3. Tính toán Feature Importance trên VARX 2 (11 biến cốt lõi)
    print("\n--- 3. TÍNH TOÁN PERMUTATION FEATURE IMPORTANCE TRÊN VARX 2 (8 BIẾN NGOẠI SINH) ---")
    feat2_exog = bundle.feature_sets["VARX 2"].drop(columns=TARGET_PROVINCES)
    varx2_mdl = CoffeeVARX(p=5, q=2, penalty="none").fit(Y[:tr_end], feat2_exog.values[:tr_end])
    base_pred_ret2 = varx2_mdl.predict_rolling(Y, feat2_exog.values, te_idx)

    imp_df = compute_varx_permutation_importance(
        varx2_mdl,
        Y,
        feat2_exog.values,
        feat2_exog.columns.tolist(),
        te_idx,
        base_pred_ret2,
        r_true,
        p_prev,
        p_true,
        n_repeats=10,
        seed=42,
    )
    print(imp_df.to_string(index=False))

    res_df.to_csv("varx_model_comparison_results.csv", index=False, encoding="utf-8-sig")
    prov_df.to_csv("varx_province_results.csv", index=False, encoding="utf-8-sig")
    imp_df.to_csv("varx_feature_importance.csv", index=False, encoding="utf-8-sig")
    print("\nĐã lưu: varx_model_comparison_results.csv, varx_province_results.csv, varx_feature_importance.csv")

    # 4. Xuất biểu đồ trực quan
    # Fig 1: Đường dự báo VARX 2 vs VARX-L vs Naive vs Thực tế trên 3 tỉnh
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    prov_names = ["Đắk Lắk", "Lâm Đồng", "Gia Lai"]

    for i in range(3):
        axes[i].plot(te_dates, p_true[:, i], label="Giá thực tế", color="#222222", lw=1.5)
        axes[i].plot(te_dates, model_preds["VARX 2"][:, i], label="VARX 2 (11 biến, OLS)", color="#1f77b4", ls="--", lw=1.2)
        axes[i].plot(te_dates, model_preds["VARX-L 2 (Ridge)"][:, i], label="VARX-L 2 (Ridge alpha=0.1)", color="#9467bd", ls="-.", lw=1.2)
        axes[i].plot(te_dates, model_preds["Naive"][:, i], label="Naive (P_{t-1})", color="#888888", ls=":", lw=1.0)
        axes[i].set_title(f"Giá cà phê Robusta tỉnh {prov_names[i]} (Tập Test: 10/2024 -> 10/2026)", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("VNĐ/kg", fontsize=9.5)
        axes[i].legend(loc="upper left", fontsize=8.5)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_varx_forecast.png")
    plt.savefig(p1, dpi=180)
    plt.close(fig)

    # Fig 2: So sánh RMSE & Directional Accuracy giữa các phiên bản VARX
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    m_names = res_df["Model"].tolist()
    rmse_vals = res_df["RMSE (VND)"].tolist()
    da_vals = [float(x.replace("%", "")) for x in res_df["DA (%)"].tolist()]

    bars1 = ax1.bar(m_names, rmse_vals, color="#3470a3", edgecolor="#222222", alpha=0.88)
    ax1.axhline(naive_m["RMSE (VND)"], color="#d62728", ls="--", lw=1.3, label=f"Naive ({naive_m['RMSE (VND)']:.1f})")
    ax1.set_title("So sánh RMSE (VND) giữa các phiên bản VARX", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (VNĐ/kg)", fontsize=10)
    ax1.set_xticks(range(len(m_names)))
    ax1.set_xticklabels(m_names, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylim(1550, 1980)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars1, rmse_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 7, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    bars2 = ax2.bar(m_names, da_vals, color="#2ca02c", edgecolor="#222222", alpha=0.88)
    ax2.axhline(50.0, color="#888888", ls=":", lw=1.2, label="Ngưỡng 50% (Tung đồng xu)")
    ax2.set_title("Tỷ lệ đoán đúng hướng DA (%) giữa các phiên bản VARX", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Đúng hướng DA (%)", fontsize=10)
    ax2.set_xticks(range(len(m_names)))
    ax2.set_xticklabels(m_names, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 85)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, da_vals):
        if v > 0:
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_varx_comparison.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    # Fig 3: Biểu đồ Permutation Importance các biến ngoại sinh trong VARX 2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    imp_sorted = imp_df.iloc[::-1]
    y_pos = np.arange(len(imp_sorted))
    ax.barh(y_pos, imp_sorted["Importance_Share (%)"], color="#ff7f0e", edgecolor="#222222", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp_sorted["Feature"], fontsize=9.5)
    ax.set_xlabel("Tỷ trọng đóng góp vào việc hạ RMSE (%)", fontsize=10)
    ax.set_title("Tầm quan trọng đặc trưng ngoại sinh trong VARX 2 (Permutation Importance)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(imp_sorted["Importance_Share (%)"]):
        ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "fig3_varx_feature_importance.png")
    plt.savefig(p3, dpi=180)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {p1}, {p2}, {p3}")
    print(f"Hoàn tất toàn bộ thực nghiệm VARX sau {time.time() - t0:.1f} giây.")


if __name__ == "__main__":
    main()


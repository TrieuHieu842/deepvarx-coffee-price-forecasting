# -*- coding: utf-8 -*-
"""
========================================================================================
CHƯƠNG TRÌNH THỰC NGHIỆM ĐỐI CHỨNG MÔ HÌNH VECTOR AUTOREGRESSION (VAR)
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

from var_coffee_models import (
    TARGET_PROVINCES,
    CoffeeVAR,
    compute_metrics,
    diebold_mariano_from_loss_diff,
    load_and_prepare_coffee_data,
    select_var_order_aic_bic,
)

sys.stdout.reconfigure(encoding="utf-8")

FIG_DIR = "figures_var"
os.makedirs(FIG_DIR, exist_ok=True)


def main() -> None:
    t0 = time.time()
    print("=" * 88)
    print("CHƯƠNG TRÌNH THỰC NGHIỆM MÔ HÌNH VECTOR AUTOREGRESSION (VAR)")
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

    # 1. Chọn bậc trễ AIC/BIC trên tập Train
    print("\n--- 1. Bảng Tiêu Chuẩn Thông Tin Lựa Chọn Bậc Trễ (Train) ---")
    aic_df = select_var_order_aic_bic(Y[:tr_end], max_p=10)
    print(aic_df.to_string(index=False))
    best_p_aic = int(aic_df.iloc[0]["p"])
    print(f"-> Bậc trễ tối ưu theo AIC: p = {best_p_aic}")

    # 2. Benchmark Naive Random Walk
    p_true = P[te_idx]
    p_prev = P[te_idx - 1]
    r_true = Y[te_idx]
    r_naive = np.zeros_like(r_true)
    naive_m = compute_metrics(p_true, p_prev, r_true, r_naive)

    results = []
    results.append({
        "Model": "Naive (Random Walk)",
        "Lag_p": 1,
        "Num_Features": 3,
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

    # Thống kê từng tỉnh cho Naive
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

    # 3. Chạy các phiên bản VAR với lag p in {1, 2, 3, 5, 7}
    lags_to_eval = [1, 2, 3, 5, 7]
    for p in lags_to_eval:
        var_mdl = CoffeeVAR(p=p).fit(Y[:tr_end])
        pred_ret = var_mdl.predict_rolling(Y, te_idx)
        pred_p = p_prev * np.exp(pred_ret)
        model_preds[f"VAR({p})"] = pred_p

        m = compute_metrics(p_true, pred_p, r_true, pred_ret)

        # Diebold-Mariano test trên loss chênh lệch tổng hợp 3 tỉnh
        e_bench = (p_true - p_prev) ** 2
        e_model = (p_true - pred_p) ** 2
        d_panel = np.mean(e_bench - e_model, axis=1)
        stat, pval = diebold_mariano_from_loss_diff(d_panel, h=1, alternative="greater")

        name = f"VAR({p})"
        results.append({
            "Model": name,
            "Lag_p": p,
            "Num_Features": 3,
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

    print("\n--- 2. BẢNG TỔNG HỢP SO SÁNH HIỆU SUẤT MÔ HÌNH VAR TRÊN TEST (542 NGÀY) ---")
    print(res_df.to_string(index=False))

    res_df.to_csv("var_model_comparison_results.csv", index=False, encoding="utf-8-sig")
    prov_df.to_csv("var_province_results.csv", index=False, encoding="utf-8-sig")
    print("\nĐã lưu: var_model_comparison_results.csv, var_province_results.csv")

    # 4. Vẽ biểu đồ trực quan
    # Fig 1: Đường dự báo thực tế VAR(5) vs Naive vs Thực tế trên 3 tỉnh
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    prov_names = ["Đắk Lắk", "Lâm Đồng", "Gia Lai"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i in range(3):
        axes[i].plot(te_dates, p_true[:, i], label="Giá thực tế", color="#222222", lw=1.5)
        axes[i].plot(te_dates, model_preds["VAR(5)"][:, i], label="Dự báo VAR(5)", color=colors[i], ls="--", lw=1.2)
        axes[i].plot(te_dates, model_preds["Naive"][:, i], label="Naive (P_{t-1})", color="#888888", ls=":", lw=1.0)
        axes[i].set_title(f"Giá cà phê Robusta tỉnh {prov_names[i]} (Tập Test: 10/2024 -> 10/2026)", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("VNĐ/kg", fontsize=9.5)
        axes[i].legend(loc="upper left", fontsize=8.5)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "fig1_var_forecast.png")
    plt.savefig(p1, dpi=180)
    plt.close(fig)

    # Fig 2: So sánh RMSE & Directional Accuracy giữa các bậc trễ p
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    var_plot = res_df[res_df["Model"].str.startswith("VAR")]
    x_lbls = var_plot["Model"].tolist()
    rmse_vals = var_plot["RMSE (VND)"].tolist()
    da_vals = [float(x.replace("%", "")) for x in var_plot["DA (%)"].tolist()]

    bars1 = ax1.bar(x_lbls, rmse_vals, color="#4a7bb0", edgecolor="#222222", alpha=0.85)
    ax1.axhline(naive_m["RMSE (VND)"], color="#d62728", ls="--", lw=1.3, label=f"Naive Benchmark ({naive_m['RMSE (VND)']:.1f})")
    ax1.set_title("So sánh RMSE (VND) giữa các bậc trễ VAR(p)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("RMSE (VNĐ/kg)", fontsize=10)
    ax1.set_ylim(1915, 1940)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for b, v in zip(bars1, rmse_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x_lbls, da_vals, color="#2ca02c", edgecolor="#222222", alpha=0.85)
    ax2.axhline(50.0, color="#888888", ls=":", lw=1.2, label="Ngưỡng 50% (Tung đồng xu)")
    ax2.set_title("Tỷ lệ đoán đúng hướng DA (%) giữa các bậc trễ VAR(p)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Đúng hướng DA (%)", fontsize=10)
    ax2.set_ylim(45, 55)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars2, da_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "fig2_var_lag_comparison.png")
    plt.savefig(p2, dpi=180)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {p1}, {p2}")
    print(f"Hoàn tất toàn bộ thực nghiệm VAR sau {time.time() - t0:.1f} giây.")


if __name__ == "__main__":
    main()


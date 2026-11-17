"""Report generation and export engine for Diebold-Mariano evaluation results."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from .comparison import ComparisonSuiteResult


def safe_to_csv(df: pd.DataFrame, target_path: Path) -> str:
    """Safely write DataFrame to CSV with fallback if file is locked."""
    try:
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        return str(target_path)
    except PermissionError:
        fallback = target_path.with_name(f"{target_path.stem}_latest{target_path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return str(fallback)


def export_evaluation_results(
    results: ComparisonSuiteResult,
    output_dir: Union[str, Path] = "statistical_evaluation/data/output",
    summary_filename: str = "dm_summary_results.csv",
    detailed_filename: str = "dm_detail_results.csv",
    markdown_filename: str = "DM_EVALUATION_REPORT.md",
    export_detailed: bool = True,
) -> Dict[str, str]:
    """Export summary CSV, detailed CSV, and Markdown report to the output directory.

    Returns:
        dict mapping file keys to output paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_paths = {}

    # 1. Export Summary CSV
    summary_path = out_dir / summary_filename
    if not results.summary_df.empty:
        preferred_cols = [
            "experiment_id",
            "data_frequency",
            "forecast_strategy",
            "actual_source",
            "province",
            "horizon",
            "model_a",
            "model_b",
            "model_a_family",
            "model_b_family",
            "model_a_config",
            "model_b_config",
            "model_a_type",
            "model_b_type",
            "loss_function",
            "n_observations",
            "mean_actual_price",
            "mean_pred_price_a",
            "mean_pred_price_b",
            "mean_loss_a",
            "mean_loss_b",
            "mean_loss_diff",
            "dm_stat",
            "dm_p_value",
            "hln_stat",
            "hln_p_value",
            "holm_p_value",
            "significance_level",
            "is_statistically_significant",
            "better_model_by_loss",
            "better_model_statistical",
            "final_interpretation",
            "hac_lag",
            "is_nested",
            "nested_restricted_model",
            "nested_unrestricted_model",
            "clark_west_stat",
            "clark_west_p_value",
            "nested_warning",
        ]
        cols_to_use = [c for c in preferred_cols if c in results.summary_df.columns]
        for c in results.summary_df.columns:
            if c not in cols_to_use:
                cols_to_use.append(c)

        export_df = results.summary_df[cols_to_use].copy()
        exported_paths["summary_csv"] = safe_to_csv(export_df, summary_path)

    # 2. Export Detailed CSV (save as both dm_detail_results.csv and dm_detailed_observations.csv)
    if export_detailed and not results.detailed_df.empty:
        detailed_path = out_dir / detailed_filename
        exported_paths["detailed_csv"] = safe_to_csv(results.detailed_df, detailed_path)

        alt_detailed_path = out_dir / "dm_detailed_observations.csv"
        exported_paths["detailed_csv_alt"] = safe_to_csv(results.detailed_df, alt_detailed_path)

    # 3. Export Markdown Report
    if markdown_filename and not results.summary_df.empty:
        md_path = out_dir / markdown_filename
        md_content = generate_markdown_report(results)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        exported_paths["markdown_report"] = str(md_path)

    return exported_paths


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table without requiring tabulate package."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in df.columns) + " |"
        data_lines.append(row_str)
    return "\n".join([header_line, separator_line] + data_lines)


def generate_markdown_report(results: ComparisonSuiteResult) -> str:
    """Generate a clean, structured Markdown evaluation report."""
    df = results.summary_df
    meta = results.metadata

    lines = [
        "# BÁO CÁO TỔNG HỢP KIỂM ĐỊNH THỐNG KÊ DIEBOLD–MARIANO",
        "",
        "## 1. Thông số thực nghiệm",
        f"- **Các mô hình so sánh:** {', '.join(meta.get('models_evaluated', []))}",
        f"- **Các tỉnh / khu vực:** {', '.join(meta.get('provinces_evaluated', []))}",
        f"- **Bước dự báo (Horizons):** {meta.get('horizons_evaluated', [])}",
        f"- **Hàm mất mát:** {', '.join(meta.get('loss_functions', []))}",
        f"- **Mức ý nghĩa thống kê (Alpha):** {meta.get('alpha', 0.05)}",
        f"- **Tổng số cặp kiểm định:** {meta.get('total_pairwise_tests', len(df))}",
        f"- **Tổng số quan sát chi tiết:** {meta.get('total_observations_evaluated', 0):,}",
        "",
        "## 2. Bảng kết quả kiểm định tổng hợp (Summary Results)",
        "",
    ]

    # Convert table to markdown
    display_cols = [
        "province",
        "horizon",
        "model_a",
        "model_b",
        "loss_function",
        "n_observations",
        "hln_stat",
        "hln_p_value",
        "holm_p_value",
        "is_statistically_significant",
        "better_model_by_loss",
        "better_model_statistical",
    ]
    present_cols = [c for c in display_cols if c in df.columns]
    table_df = df[present_cols].copy()

    # Format numeric columns
    for num_col in ["hln_stat"]:
        if num_col in table_df.columns:
            table_df[num_col] = table_df[num_col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "–")
    for p_col in ["hln_p_value", "holm_p_value"]:
        if p_col in table_df.columns:
            table_df[p_col] = table_df[p_col].map(
                lambda x: "< 0.0001" if (pd.notna(x) and x < 1e-4) else (f"{x:.4f}" if pd.notna(x) else "–")
            )

    lines.append(df_to_markdown_simple(table_df))
    lines.append("")
    lines.append("## 3. Tổng hợp các cặp có ưu thế thống kê vượt trội (Statistically Significant Differences)")
    lines.append("")

    sig_df = df[df["is_statistically_significant"] == True]
    if sig_df.empty:
        lines.append("Không có cặp mô hình nào đạt được sự khác biệt có ý nghĩa thống kê ở mức alpha đã chọn.")
    else:
        lines.append(f"Có **{len(sig_df)}** phép kiểm định xác nhận sự khác biệt có ý nghĩa thống kê:")
        lines.append("")
        for _, r in sig_df.iterrows():
            lines.append(
                f"- **{r['province']}** (h={r['horizon']}, {r['loss_function'].upper()}): "
                f"**{r['better_model_statistical']}** vượt trội hơn (HLN stat={r['hln_stat']:+.3f}, "
                f"p-value Holm={r['holm_p_value']:.4f})."
            )

    lines.append("")
    lines.append("---")
    lines.append("*Báo cáo được sinh tự động bởi Statistical Evaluation Module (Diebold-Mariano Suite).*")

    return "\n".join(lines)


def generate_model_discovery_report(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "statistical_evaluation/data/output",
    filename: str = "model_discovery_report.csv",
) -> str:
    """Generate model_discovery_report.csv detailing all discovered models and their properties."""
    import numpy as np

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    cols = {c.lower(): c for c in df.columns}
    m_col = cols.get("model", "model")
    p_col = cols.get("province", cols.get("region", "province"))
    t_col = cols.get("target_date", cols.get("date", "target_date"))
    y_col = cols.get("actual", cols.get("y_true", "actual"))
    yp_col = cols.get("predicted", cols.get("y_pred", "predicted"))
    h_col = cols.get("horizon", "horizon")

    rows = []
    for m in sorted(df[m_col].dropna().unique()):
        sub = df[df[m_col] == m]
        has_std = "pred_std" in sub.columns and sub["pred_std"].notna().any()

        m_low = str(m).lower()
        if "naive" in m_low:
            family = "Benchmark"
        elif "deepvarx" in m_low:
            family = "DeepVARX"
        elif "deepvar" in m_low:
            family = "DeepVAR"
        elif "varx" in m_low:
            family = "VARX"
        elif "var" in m_low:
            family = "VAR"
        elif "lstm" in m_low:
            family = "LSTM"
        else:
            family = "Other"

        y_t = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
        y_p = pd.to_numeric(sub[yp_col], errors="coerce").to_numpy(dtype=float)
        mask = ~(np.isnan(y_t) | np.isnan(y_p))
        y_t = y_t[mask]
        y_p = y_p[mask]

        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2))) if len(y_t) > 0 else np.nan
        mae = float(np.mean(np.abs(y_t - y_p))) if len(y_t) > 0 else np.nan

        provs = sorted(sub[p_col].dropna().unique().tolist()) if p_col in sub.columns else ["all"]
        d_min = str(sub[t_col].min()) if t_col in sub.columns else ""
        d_max = str(sub[t_col].max()) if t_col in sub.columns else ""
        h_val = int(sub[h_col].iloc[0]) if h_col in sub.columns else 1

        m_cfg = sub["model_config"].iloc[0] if "model_config" in sub.columns else ""
        exp_id = sub["experiment_id"].iloc[0] if "experiment_id" in sub.columns else "EXP_SESSION_2LAYER_20261113"
        freq = sub["data_frequency"].iloc[0] if "data_frequency" in sub.columns else "daily"
        strat = sub["forecast_strategy"].iloc[0] if "forecast_strategy" in sub.columns else "rolling_1step_ahead"
        src = sub["actual_source"].iloc[0] if "actual_source" in sub.columns else "GiaCaPhe_Clean_V2"

        mean_act = float(np.mean(y_t)) if len(y_t) > 0 else np.nan
        mean_pred = float(np.mean(y_p)) if len(y_p) > 0 else np.nan

        rows.append({
            "experiment_id": exp_id,
            "data_frequency": freq,
            "forecast_strategy": strat,
            "actual_source": src,
            "model_name": m,
            "family": family,
            "model_config": m_cfg,
            "forecast_type": "Probabilistic" if has_std else "Point",
            "has_pred_std": has_std,
            "n_observations": len(sub),
            "provinces": ", ".join(provs),
            "date_start": d_min,
            "date_end": d_max,
            "horizon": h_val,
            "mean_actual_price_vnd": round(mean_act, 2),
            "mean_predicted_price_vnd": round(mean_pred, 2),
            "test_rmse_vnd": round(rmse, 2),
            "test_mae_vnd": round(mae, 2),
            "evaluation_status": "Included in DM Test",
        })

    disc_df = pd.DataFrame(rows)
    disc_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return str(out_path)

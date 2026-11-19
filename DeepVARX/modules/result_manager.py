"""
QUẢN LÝ LƯU TRỮ VÀ KHÔI PHỤC KẾT QUẢ HUẤN LUYỆN MÔ HÌNH (MODEL RESULT MANAGER)
Đảm bảo kết quả dự báo, độ đo đánh giá và trọng số mô hình được lưu trữ bền vững trên đĩa (disk),
sẵn sàng phục vụ cho Trang 6 (Báo cáo thực nghiệm) và Trang 7 (Kiểm định thống kê DM).
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from modules.data_utils import metrics

logger = logging.getLogger("ResultManager")

RESULTS_DIR = Path("data/results")
FORECAST_DATASET_PATH = Path("statistical_evaluation/data/input/all_models_forecasts_test_542days.csv")

ORDER = ["VAR", "VARX", "LSTM", "DeepVAR", "DeepVARX"]

MODEL_DM_MAP = {
    "VAR": "VAR(5)",
    "VARX": "VARX 2",
    "LSTM": "LSTM 2",
    "DeepVAR": "DeepVAR 2",
    "DeepVARX": "DeepVARX 3",
}

PROV_SLUG_MAP = {
    "Đắk Lắk": "dak_lak",
    "Lâm Đồng": "lam_dong",
    "Gia Lai": "gia_lai",
    "Đắk Nông": "dak_nong",
    "dak_lak": "dak_lak",
    "lam_dong": "lam_dong",
    "gia_lai": "gia_lai",
    "dak_nong": "dak_nong",
}

SLUG_TO_PROV = {
    "dak_lak": "Đắk Lắk",
    "lam_dong": "Lâm Đồng",
    "gia_lai": "Gia Lai",
    "dak_nong": "Đắk Nông",
}


def ensure_results_dir() -> Path:
    """Đảm bảo thư mục data/results tồn tại."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_model_run(name: str, out: dict, alloc_info: dict | None = None, meta: dict | None = None) -> dict[str, str]:
    """
    Lưu toàn bộ kết quả của mô hình sau khi huấn luyện vào thư mục data/results/:
    1. predictions_{name}.csv: Chuỗi dự báo test chi tiết (date, province, y_true, y_pred, y_prev, error, abs_error, sq_error)
    2. by_province_{name}.csv: Bảng metrics theo từng tỉnh
    3. metrics_{name}.json: Metrics trung bình, naive metrics, thời gian huấn luyện
    4. history_{name}.csv: Lịch sử loss qua các epoch (nếu có)
    5. attention_{name}.pkl: Trọng số attention (nếu có)
    6. models_summary.csv: Bảng tổng hợp cập nhật của toàn bộ các mô hình
    7. all_predictions.csv: Tệp gộp dự báo của các mô hình đã huấn luyện
    8. Đồng bộ vào all_models_forecasts_test_542days.csv cho Trang 7 (DM Test).
    """
    ensure_results_dir()
    saved_files = {}

    pred = out.get("pred")
    by_prov = out.get("by_prov")
    avg_metrics = out.get("metrics")
    scope = alloc_info.get("scope", "Wide") if alloc_info else out.get("scope", "Wide")
    seconds = out.get("seconds", 0.0)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Lưu chuỗi dự báo chi tiết
    if isinstance(pred, pd.DataFrame) and not pred.empty:
        pred_df = pred.copy()
        if "y_true" in pred_df.columns and "y_pred" in pred_df.columns:
            pred_df["error"] = pred_df["y_true"] - pred_df["y_pred"]
            pred_df["abs_error"] = pred_df["error"].abs()
            pred_df["sq_error"] = pred_df["error"] ** 2

        pred_path = RESULTS_DIR / f"predictions_{name}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8")
        saved_files["predictions"] = str(pred_path)

    # 2. Lưu metrics theo từng tỉnh
    if isinstance(by_prov, pd.DataFrame) and not by_prov.empty:
        by_prov_path = RESULTS_DIR / f"by_province_{name}.csv"
        by_prov.to_csv(by_prov_path, index=False, encoding="utf-8")
        saved_files["by_province"] = str(by_prov_path)

    # 3. Lưu metrics trung bình và metadata vào JSON
    meta_info = {
        "model": name,
        "scope": scope,
        "metrics": avg_metrics or {},
        "naive_metrics": out.get("naive_metrics") or {},
        "seconds": float(seconds),
        "trained_at": now_str,
    }
    meta_json_path = RESULTS_DIR / f"metrics_{name}.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)
    saved_files["metrics_json"] = str(meta_json_path)

    # 4. Lưu lịch sử huấn luyện (epochs) nếu có
    history = out.get("history")
    if isinstance(history, pd.DataFrame) and not history.empty:
        hist_path = RESULTS_DIR / f"history_{name}.csv"
        history.to_csv(hist_path, index=False, encoding="utf-8")
        saved_files["history"] = str(hist_path)

    # 5. Lưu trọng số attention nếu có
    attn = out.get("attn")
    if attn and isinstance(attn, dict):
        attn_pkl_path = RESULTS_DIR / f"attention_{name}.pkl"
        with open(attn_pkl_path, "wb") as f:
            pickle.dump(attn, f)
        saved_files["attention_pkl"] = str(attn_pkl_path)

        if isinstance(attn.get("epoch"), pd.DataFrame):
            attn_epoch_path = RESULTS_DIR / f"attention_epoch_{name}.csv"
            attn["epoch"].to_csv(attn_epoch_path, encoding="utf-8")
            saved_files["attention_epoch"] = str(attn_epoch_path)

    # 6. Cập nhật bảng tổng hợp models_summary.csv
    summary_path = RESULTS_DIR / "models_summary.csv"
    if summary_path.exists():
        try:
            sum_df = pd.read_csv(summary_path)
        except Exception:
            sum_df = pd.DataFrame()
    else:
        sum_df = pd.DataFrame()

    new_row = {
        "Mô hình": name,
        "Phạm vi": scope,
        "RMSE": avg_metrics.get("RMSE", np.nan) if avg_metrics else np.nan,
        "MAE": avg_metrics.get("MAE", np.nan) if avg_metrics else np.nan,
        "MAPE (%)": avg_metrics.get("MAPE (%)", np.nan) if avg_metrics else np.nan,
        "sMAPE (%)": avg_metrics.get("sMAPE (%)", np.nan) if avg_metrics else np.nan,
        "R²": avg_metrics.get("R²", np.nan) if avg_metrics else np.nan,
        "Đúng hướng (%)": avg_metrics.get("Đúng hướng (%)", np.nan) if avg_metrics else np.nan,
        "Thời gian (s)": round(float(seconds), 2),
        "Ngày huấn luyện": now_str,
    }

    if not sum_df.empty and "Mô hình" in sum_df.columns:
        sum_df = sum_df[sum_df["Mô hình"] != name]
        sum_df = pd.concat([sum_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        sum_df = pd.DataFrame([new_row])

    # Đảm bảo thứ tự theo ORDER
    order_map = {m: i for i, m in enumerate(ORDER)}
    sum_df["_sort"] = sum_df["Mô hình"].map(lambda x: order_map.get(x, 99))
    sum_df = sum_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    sum_df.to_csv(summary_path, index=False, encoding="utf-8")
    saved_files["models_summary"] = str(summary_path)

    # 7. Cập nhật tệp gộp all_predictions.csv
    all_pred_path = RESULTS_DIR / "all_predictions.csv"
    if isinstance(pred, pd.DataFrame) and not pred.empty:
        p_sub = pred.copy()
        p_sub["model"] = name
        if all_pred_path.exists():
            try:
                all_df = pd.read_csv(all_pred_path)
                all_df = all_df[all_df["model"] != name]
                all_df = pd.concat([all_df, p_sub], ignore_index=True)
            except Exception:
                all_df = p_sub
        else:
            all_df = p_sub
        all_df.to_csv(all_pred_path, index=False, encoding="utf-8")
        saved_files["all_predictions"] = str(all_pred_path)

    # 8. Đồng bộ vào dataset kiểm định DM (Trang 7)
    sync_to_dm_forecast_dataset(name, pred)

    logger.info("Đã lưu kết quả huấn luyện mô hình %s vào %s", name, RESULTS_DIR)
    return saved_files


def sync_to_dm_forecast_dataset(name: str, pred: pd.DataFrame | None) -> bool:
    """Đồng bộ dự báo vừa huấn luyện vào file all_models_forecasts_test_542days.csv."""
    if pred is None or pred.empty or not FORECAST_DATASET_PATH.exists():
        return False

    canonical_name = MODEL_DM_MAP.get(name, name)
    try:
        fc_df = pd.read_csv(FORECAST_DATASET_PATH)
        pred_copy = pred.copy()
        pred_copy["date_str"] = pd.to_datetime(pred_copy["date"]).dt.strftime("%Y-%m-%d")
        pred_copy["prov_slug"] = pred_copy["province"].map(lambda p: PROV_SLUG_MAP.get(p, p))

        # Cập nhật các dòng khớp (model, province, target_date)
        match_mask = (fc_df["model"] == canonical_name)
        if match_mask.any():
            lookup = pred_copy.set_index(["prov_slug", "date_str"])
            for idx in fc_df[match_mask].index:
                row_prov = fc_df.at[idx, "province"]
                row_date = str(fc_df.at[idx, "target_date"])
                if (row_prov, row_date) in lookup.index:
                    matched = lookup.loc[(row_prov, row_date)]
                    if isinstance(matched, pd.DataFrame):
                        matched = matched.iloc[0]
                    y_pred_val = float(matched["y_pred"])
                    y_true_val = float(matched["y_true"])
                    y_prev_val = float(matched.get("y_prev", np.nan))

                    fc_df.at[idx, "predicted_price"] = y_pred_val
                    fc_df.at[idx, "predicted"] = y_pred_val
                    fc_df.at[idx, "actual_price"] = y_true_val
                    fc_df.at[idx, "actual"] = y_true_val
                    if not pd.isna(y_prev_val):
                        fc_df.at[idx, "prev_price"] = y_prev_val

            fc_df.to_csv(FORECAST_DATASET_PATH, index=False, encoding="utf-8")
            return True
    except Exception as e:
        logger.warning("Không thể đồng bộ dự báo vào DM dataset: %s", e)
    return False


def load_all_saved_results() -> dict[str, dict]:
    """
    Tải toàn bộ kết quả mô hình đã lưu trên đĩa:
    - Ưu tiên đọc từ data/results/ nếu có.
    - Nếu data/results/ chưa đủ 6 mô hình, tự động bổ sung từ all_models_forecasts_test_542days.csv.
    """
    ensure_results_dir()
    results = {}

    # 1. Đọc từ data/results/
    for name in ORDER:
        pred_path = RESULTS_DIR / f"predictions_{name}.csv"
        prov_path = RESULTS_DIR / f"by_province_{name}.csv"
        json_path = RESULTS_DIR / f"metrics_{name}.json"

        if pred_path.exists() and prov_path.exists():
            try:
                pred = pd.read_csv(pred_path)
                pred["date"] = pd.to_datetime(pred["date"])
                by_prov = pd.read_csv(prov_path)
                
                meta_dict = {}
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)

                # Nạp attention nếu có
                attn = None
                attn_pkl = RESULTS_DIR / f"attention_{name}.pkl"
                if attn_pkl.exists():
                    try:
                        with open(attn_pkl, "rb") as f:
                            attn = pickle.load(f)
                    except Exception:
                        attn = None

                # Nạp history nếu có
                history = None
                hist_path = RESULTS_DIR / f"history_{name}.csv"
                if hist_path.exists():
                    try:
                        history = pd.read_csv(hist_path)
                    except Exception:
                        history = None

                results[name] = {
                    "pred": pred,
                    "by_prov": by_prov,
                    "metrics": meta_dict.get("metrics") or {k: float(by_prov[k].mean()) for k in ["RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "R²", "Đúng hướng (%)"] if k in by_prov.columns},
                    "naive_metrics": meta_dict.get("naive_metrics") or {},
                    "scope": meta_dict.get("scope", "Wide"),
                    "seconds": meta_dict.get("seconds", 0.0),
                    "log": f"[saved-disk] Đã nạp kết quả lưu trên đĩa ({meta_dict.get('trained_at', 'trước đó')}).",
                    "attn": attn,
                    "history": history,
                }
            except Exception as e:
                logger.warning("Lỗi nạp mô hình %s từ data/results/: %s", name, e)

    # 2. Nếu thiếu mô hình, nạp bổ sung từ all_models_forecasts_test_542days.csv
    missing_models = [m for m in ORDER if m not in results]
    if missing_models and FORECAST_DATASET_PATH.exists():
        try:
            fc_df = pd.read_csv(FORECAST_DATASET_PATH)
            for m_key in missing_models:
                m_name = MODEL_DM_MAP.get(m_key, m_key)
                sub = fc_df[fc_df["model"] == m_name].copy()
                if sub.empty:
                    continue
                pred = pd.DataFrame({
                    "date": pd.to_datetime(sub["target_date"]),
                    "province": sub["province"].map(lambda p: SLUG_TO_PROV.get(p, p)),
                    "y_true": sub["actual_price"],
                    "y_pred": sub["predicted_price"],
                    "y_prev": sub["prev_price"],
                })
                by = pd.DataFrame([{"province": e, **metrics(g["y_true"], g["y_pred"], g.get("y_prev"))} for e, g in pred.groupby("province")])
                metric_cols = [c for c in ["RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "R²", "Đúng hướng (%)"] if c in by.columns]
                avg = {k: float(by[k].mean()) for k in metric_cols}
                results[m_key] = {
                    "pred": pred,
                    "by_prov": by,
                    "metrics": avg,
                    "naive_metrics": metrics(pred["y_true"], pred["y_prev"]),
                    "scope": "Wide" if "VAR" in m_key else "Global",
                    "seconds": 0.0,
                    "log": f"[pre-loaded] Kết quả dự báo tiêu chuẩn của {m_name}.",
                    "attn": None,
                    "history": None,
                }
        except Exception as e:
            logger.warning("Lỗi nạp bổ sung từ all_models_forecasts_test_542days.csv: %s", e)

    return results


def init_results_storage_if_empty() -> None:
    """Tự động khởi tạo lưu trữ kết quả ban đầu vào data/results/ nếu thư mục còn trống."""
    ensure_results_dir()
    summary_path = RESULTS_DIR / "models_summary.csv"
    if summary_path.exists() and summary_path.stat().st_size > 50:
        return

    logger.info("Khởi tạo bộ nhớ kết quả data/results/ từ dữ liệu tiêu chuẩn...")
    loaded = load_all_saved_results()
    for name, data in loaded.items():
        save_model_run(name, data, alloc_info={"scope": data.get("scope", "Wide")})

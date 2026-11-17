"""Universal Data Adapter for multi-format forecast data ingestion.

Standard Schema:
- model: str (Model identifier)
- province: str (Province or region)
- forecast_origin: str (YYYY-MM-DD origin date of forecast)
- target_date: str (YYYY-MM-DD target date of forecast)
- horizon: int (Forecast step h >= 1)
- actual: float (Ground truth value in VND/kg)
- predicted: float (Point forecast value in VND/kg)
- pred_std: Optional[float] (Predictive standard deviation for probabilistic models)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_SCHEMA = [
    "experiment_id",
    "data_frequency",
    "forecast_strategy",
    "actual_source",
    "model",
    "model_family",
    "model_config",
    "prediction_type",
    "province",
    "forecast_origin",
    "target_date",
    "horizon",
    "actual_price",
    "predicted_price",
    "actual",
    "predicted",
]


class UniversalForecastAdapter:
    """Universal parser and normalizer for forecast tables."""

    @staticmethod
    def identify_format(df: pd.DataFrame) -> str:
        """Identify whether a DataFrame is in long-format, wide-format, or summary metric format."""
        cols = [c.lower().strip() for c in df.columns]

        # Check if summary metrics table
        if len(df) < 50 and any(m in cols for m in ["rmse", "mae", "mape (%)", "da (%)"]):
            return "summary_metrics"

        # Check if long-format
        has_model = any(c in cols for c in ["model", "model_name", "forecaster", "method"])
        has_pred = any(c in cols for c in ["predicted", "pred", "y_pred", "forecast", "y_hat"])
        has_actual = any(c in cols for c in ["actual", "true", "y_true", "actual_value"])

        if has_model and has_pred and has_actual:
            return "long_format"

        # Check if wide-format: has date and multiple prediction columns
        has_date = any(c in cols for c in ["date", "target_date", "forecast_date"])
        pred_cols = [c for c in cols if any(k in c for k in ["pred_", "y_hat_", "forecast_"])]
        if has_date and (has_actual or "price_vnd_kg" in cols) and len(pred_cols) >= 1:
            return "wide_format"

        return "unknown"

    @classmethod
    def normalize_long_format(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map heterogeneous column names to standard long-format schema."""
        col_map = {}
        for c in df.columns:
            low = str(c).strip().lower()
            if low in ["model", "model_name", "forecaster", "method"]:
                col_map[c] = "model"
            elif low in ["model_family", "family"]:
                col_map[c] = "model_family"
            elif low in ["model_config", "config", "hyperparams"]:
                col_map[c] = "model_config"
            elif low in ["prediction_type", "pred_type"]:
                col_map[c] = "prediction_type"
            elif low in ["experiment_id", "run_id"]:
                col_map[c] = "experiment_id"
            elif low in ["data_frequency", "frequency", "freq"]:
                col_map[c] = "data_frequency"
            elif low in ["forecast_strategy", "strategy"]:
                col_map[c] = "forecast_strategy"
            elif low in ["actual_source", "source"]:
                col_map[c] = "actual_source"
            elif low in ["province", "region", "location", "entity"]:
                col_map[c] = "province"
            elif low in ["target_date", "date", "forecast_date", "time"]:
                col_map[c] = "target_date"
            elif low in ["forecast_origin", "origin", "origin_date", "input_end"]:
                col_map[c] = "forecast_origin"
            elif low in ["horizon", "h", "step"]:
                col_map[c] = "horizon"
            elif low in ["actual_price", "price_actual", "true_price"]:
                col_map[c] = "actual_price"
            elif low in ["predicted_price", "price_predicted", "pred_price"]:
                col_map[c] = "predicted_price"
            elif low in ["actual", "true", "y_true", "actual_value", "price_vnd_kg"]:
                col_map[c] = "actual"
            elif low in ["predicted", "pred", "y_pred", "forecast", "y_hat"]:
                col_map[c] = "predicted"
            elif low in ["pred_std", "std", "sigma", "scale"]:
                col_map[c] = "pred_std"
            elif low in ["y_prev", "prev_price", "prev"]:
                col_map[c] = "prev_price"

        df_out = df.rename(columns=col_map).copy()

        # Mirror actual_price and predicted_price
        if "actual_price" not in df_out.columns and "actual" in df_out.columns:
            df_out["actual_price"] = df_out["actual"]
        if "actual" not in df_out.columns and "actual_price" in df_out.columns:
            df_out["actual"] = df_out["actual_price"]

        if "predicted_price" not in df_out.columns and "predicted" in df_out.columns:
            df_out["predicted_price"] = df_out["predicted"]
        if "predicted" not in df_out.columns and "predicted_price" in df_out.columns:
            df_out["predicted"] = df_out["predicted_price"]

        # Default audit attributes if not present
        if "data_frequency" not in df_out.columns:
            df_out["data_frequency"] = "daily"
        if "forecast_strategy" not in df_out.columns:
            df_out["forecast_strategy"] = "rolling_1step_ahead"
        if "actual_source" not in df_out.columns:
            df_out["actual_source"] = "GiaCaPhe_Clean_V2"

        # Target date standardization
        if "target_date" in df_out.columns:
            df_out["target_date"] = pd.to_datetime(df_out["target_date"]).dt.strftime("%Y-%m-%d")

        # Horizon
        if "horizon" not in df_out.columns:
            df_out["horizon"] = 1
        else:
            df_out["horizon"] = df_out["horizon"].astype(int)

        # Forecast origin derivation if missing
        if "forecast_origin" not in df_out.columns and "target_date" in df_out.columns:
            # If y_prev is available or sequential, forecast_origin for h=1 is previous target date
            df_out["forecast_origin"] = (
                pd.to_datetime(df_out["target_date"]) - pd.to_timedelta(df_out["horizon"], unit="D")
            ).dt.strftime("%Y-%m-%d")

        # Province
        if "province" not in df_out.columns:
            df_out["province"] = "all"

        # Numeric conversions
        df_out["actual"] = pd.to_numeric(df_out["actual"], errors="coerce")
        df_out["predicted"] = pd.to_numeric(df_out["predicted"], errors="coerce")

        if "pred_std" in df_out.columns:
            df_out["pred_std"] = pd.to_numeric(df_out["pred_std"], errors="coerce")

        cols_order = [c for c in STANDARD_SCHEMA if c in df_out.columns]
        extra_cols = [c for c in df_out.columns if c not in cols_order]
        return df_out[cols_order + extra_cols]

    @classmethod
    def normalize_wide_format(
        cls,
        df: pd.DataFrame,
        model_columns: Optional[Dict[str, str]] = None,
        date_col: str = "date",
        actual_col: str = "actual",
        province_col: str = "province",
    ) -> pd.DataFrame:
        """Convert wide-format DataFrame (where each model has its own column) into long-format."""
        low_cols = {c.lower(): c for c in df.columns}
        d_col = low_cols.get(date_col.lower(), date_col)
        y_col = low_cols.get(actual_col.lower(), actual_col)
        p_col = low_cols.get(province_col.lower(), None)

        if d_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"Wide-format requires date ('{date_col}') and actual ('{actual_col}') columns.")

        if model_columns is None:
            # Identify columns that are not date, actual, province
            exclude = {d_col, y_col}
            if p_col:
                exclude.add(p_col)
            model_columns = {c: c.replace("pred_", "").replace("y_hat_", "") for c in df.columns if c not in exclude}

        records = []
        for orig_col, model_name in model_columns.items():
            if orig_col not in df.columns:
                continue
            for _, row in df.iterrows():
                target_dt = pd.to_datetime(row[d_col]).strftime("%Y-%m-%d")
                records.append({
                    "model": model_name,
                    "province": row[p_col] if p_col else "all",
                    "forecast_origin": (pd.to_datetime(target_dt) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "target_date": target_dt,
                    "horizon": 1,
                    "actual": float(row[y_col]),
                    "predicted": float(row[orig_col]),
                })

        return pd.DataFrame(records)

    @classmethod
    def load_and_standardize(cls, file_or_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load and normalize any file or DataFrame into standard schema."""
        if isinstance(file_or_path, pd.DataFrame):
            df = file_or_path
        else:
            p = Path(file_or_path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            df = pd.read_csv(p)

        fmt = cls.identify_format(df)
        if fmt == "summary_metrics":
            raise ValueError(
                f"File contains aggregated summary metrics, not observation-level predictions. "
                f"DM Test requires observation-level time-series forecasts."
            )
        elif fmt == "wide_format":
            return cls.normalize_wide_format(df)
        else:
            return cls.normalize_long_format(df)

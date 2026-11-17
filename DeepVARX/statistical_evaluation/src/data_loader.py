"""Data loading and standardization engine for model forecasts."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .data_validator import validate_forecast_dataframe, ValidationReport


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to uniform lowercase conventions."""
    mapping = {
        "target_date": "date",
        "forecast_date": "date",
        "time": "date",
        "region": "province",
        "location": "province",
        "entity": "province",
        "model_name": "model",
        "forecaster": "model",
        "method": "model",
        "actual": "y_true",
        "ground_truth": "y_true",
        "actual_value": "y_true",
        "predicted": "y_pred",
        "forecast": "y_pred",
        "pred": "y_pred",
        "y_hat": "y_pred",
        "h": "horizon",
        "step": "horizon",
    }
    new_cols = {}
    for col in df.columns:
        clean = str(col).strip().lower()
        new_cols[col] = mapping.get(clean, clean)
    return df.rename(columns=new_cols)


def load_forecast_csv(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a single forecast CSV file and standardize columns."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Forecast file not found: {p}")

    df = pd.read_csv(p, encoding=encoding)
    df = normalize_column_names(df)

    if "horizon" not in df.columns:
        df["horizon"] = 1

    return df


def load_from_directory(
    dir_path: Union[str, Path],
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and merge all forecast CSV files from a directory."""
    dir_p = Path(dir_path)
    if not dir_p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_p}")

    files = sorted(dir_p.glob(pattern))
    if not files:
        raise ValueError(f"No CSV files found matching '{pattern}' in {dir_p}")

    dfs = []
    for f in files:
        sub_df = load_forecast_csv(f)
        # If model column missing, infer from filename
        if "model" not in sub_df.columns:
            stem = f.stem.replace("predictions_", "").replace("pred_", "").replace("_test", "")
            sub_df["model"] = stem
        dfs.append(sub_df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined

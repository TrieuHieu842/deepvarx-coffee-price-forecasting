"""Statistical Evaluation Module for Coffee Price Forecasting."""
from .loss_functions import (
    squared_error_loss,
    absolute_error_loss,
    gaussian_crps_loss,
    compute_loss_differential,
    get_loss_function,
)
from .hac_estimator import estimate_hac_variance, select_hac_lag
from .hln_correction import apply_hln_correction, compute_hln_multiplier
from .holm_correction import holm_bonferroni_correction
from .dm_test import DMTestResult, diebold_mariano_test
from .data_validator import validate_forecast_dataframe, ValidationReport
from .data_alignment import align_model_pair
from .data_loader import load_forecast_csv, load_from_directory
from .comparison import run_model_comparisons, ComparisonSuiteResult
from .report_generator import export_evaluation_results

__all__ = [
    "squared_error_loss",
    "absolute_error_loss",
    "gaussian_crps_loss",
    "compute_loss_differential",
    "get_loss_function",
    "estimate_hac_variance",
    "select_hac_lag",
    "apply_hln_correction",
    "compute_hln_multiplier",
    "holm_bonferroni_correction",
    "DMTestResult",
    "diebold_mariano_test",
    "validate_forecast_dataframe",
    "ValidationReport",
    "align_model_pair",
    "load_forecast_csv",
    "load_from_directory",
    "run_model_comparisons",
    "ComparisonSuiteResult",
    "export_evaluation_results",
]

"""Configuration and default parameters for Statistical Evaluation Module."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationConfig:
    alpha: float = 0.05
    default_horizons: List[int] = field(default_factory=lambda: [1])
    default_loss_funcs: List[str] = field(default_factory=lambda: ["mse", "mae"])
    default_provinces: List[str] = field(default_factory=lambda: ["dak_lak", "lam_dong", "gia_lai"])
    hac_lag: Optional[int] = None  # None = auto selection using Newey-West formula
    use_hln_correction: bool = True
    use_holm_correction: bool = True
    min_observations: int = 10
    warn_nested_models: bool = True

"""Common data models used across the project."""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional

@dataclass
class GARCHResult:
    """Data class for GARCH model results"""
    model_type: str  # One of: 'garch', 'egarch', 'gjrgarch'
    distribution: str  # One of: 'normal', 'studentst'
    params: dict
    forecast_path: np.ndarray  # Shape: (252,) annualized volatility
    volatility_path: np.ndarray  # Historical volatilities

@dataclass
class ForecastWindow:
    """Data class for forecast windows"""
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: List[GARCHResult]
    ensemble_stats: Dict[str, float]
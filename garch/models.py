from dataclasses import dataclass
from datetime import datetime
import numpy as np
from typing import List, Dict

@dataclass
class GARCHResult:
    """Container for GARCH estimation results"""
    model_type: str
    distribution: str
    params: Dict[str, float]
    forecast_path: np.ndarray  # Store mean forecast path
    volatility_path: np.ndarray  # Store historical volatilities

@dataclass
class ForecastWindow:
    """Data class for forecast windows"""
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: List[GARCHResult]
    ensemble_stats: Dict[str, Dict[str, float]] 
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
from .estimator import GARCHEstimator, GARCHResult

logger = logging.getLogger(__name__)

@dataclass
class ForecastWindow:
    """Container for expanding window forecast results"""
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: List[GARCHResult]
    ensemble_stats: Dict[str, float]
    
class GARCHForecaster:
    """Manages expanding window GARCH forecasts"""
    
    def __init__(self,
                estimator: Optional[GARCHEstimator] = None,
                min_window: int = 1260,
                step_size: int = 21):
        """
        Initialize forecaster
        
        Parameters:
        -----------
        estimator : GARCHEstimator, optional
            Pre-configured GARCH estimator, or creates new one if None
        min_window : int
            Minimum window size for estimation (default 5 years)
        step_size : int
            Number of days to expand window (default 1 month)
        """
        self.estimator = estimator or GARCHEstimator()
        self.min_window = min_window
        self.step_size = step_size
        self.forecast_windows: List[ForecastWindow] = []
        
    def generate_expanding_windows(self, returns: np.ndarray, start_idx: int = None,
                                min_window: int = None) -> List[np.ndarray]:
        """Generate expanding windows for analysis"""
        if min_window is None:
            min_window = self.min_window
        
        if len(returns) < min_window:
            raise ValueError(f"Insufficient data: {len(returns)} < {min_window}")
        
        if start_idx is None:
            start_idx = min_window
        
        windows = []
        for i in range(start_idx, len(returns)):
            window = returns[max(0, i-min_window):i]
            if len(window) >= min_window:
                windows.append(window)
        
        return windows
    
    def get_forecast_series(self, stat_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time series of a particular ensemble statistic
        
        Parameters:
        -----------
        stat_name : str
            Name of statistic to extract (GEV, EVOEV, etc.)
            
        Returns:
        --------
        tuple
            (dates, values) for the requested statistic
        """
        if not self.forecast_windows:
            raise ValueError("No forecast windows available")
            
        dates = np.array([w.end_date for w in self.forecast_windows])
        values = np.array([w.ensemble_stats[stat_name] for w in self.forecast_windows])
        
        return dates, values
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast results to DataFrame"""
        if not self.forecast_windows:
            raise ValueError("No forecast windows available")
            
        data = []
        for window in self.forecast_windows:
            row = {
                'start_date': window.start_date,
                'end_date': window.end_date,
                **window.ensemble_stats
            }
            data.append(row)
            
        return pd.DataFrame(data)

# Example usage
if __name__ == '__main__':
    # Generate sample data
    np.random.seed(42)
    n_days = 2000
    dates = pd.date_range('2015-01-01', periods=n_days)
    returns = np.random.normal(0, 1, n_days) * 0.01
    
    # Initialize forecaster
    forecaster = GARCHForecaster()
    
    # Generate expanding window forecasts
    windows = forecaster.generate_expanding_windows(
        returns=returns,
        dates=dates,
        forecast_horizon=252
    )
    
    # Convert to DataFrame
    results_df = forecaster.to_dataframe()
    print("\nFirst few rows of results:")
    print(results_df.head())
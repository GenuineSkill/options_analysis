from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
from .estimator import GARCHEstimator, GARCHResult
import pickle
from pathlib import Path

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
    
    def __init__(self, estimator, min_window: int = 1260):
        """Initialize forecaster with GARCH estimator and minimum window size"""
        self.estimator = estimator
        self.min_window = min_window

    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a single window of returns data"""
        # Return the window array directly, not a dict
        return window

    def generate_expanding_windows(self, returns: np.ndarray, 
                                 checkpoint_dir: Optional[Path] = None,
                                 start_idx: Optional[int] = None,
                                 min_window: Optional[int] = None) -> List[np.ndarray]:
        """Generate expanding windows for analysis"""
        logger = logging.getLogger('garch.forecaster')
        
        try:
            if min_window is None:
                min_window = self.min_window
                
            if len(returns) < min_window:
                raise ValueError(f"Insufficient data: {len(returns)} < {min_window}")
                
            if start_idx is None:
                start_idx = min_window
                
            windows = []
            for i in range(min_window, len(returns) + 1):
                window = returns[max(0, i-min_window):i]
                if len(window) >= min_window:
                    if checkpoint_dir:
                        checkpoint_file = checkpoint_dir / f"window_{i}.npy"
                        if not checkpoint_file.exists():
                            np.save(checkpoint_file, window)
                    windows.append(window)
                    
            logger.info(f"Generated {len(windows)} windows from {len(returns)} observations")
            if not windows:
                logger.warning(f"No windows generated. Returns length: {len(returns)}, min_window: {min_window}")
            
            return windows
            
        except Exception as e:
            logger.error(f"Error generating windows: {str(e)}")
            raise
    
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
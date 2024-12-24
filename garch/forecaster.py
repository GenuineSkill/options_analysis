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
    
    def __init__(self, estimator: GARCHEstimator, min_window: int = 1260):
        """Initialize forecaster with estimator and minimum window size"""
        self.estimator = estimator
        self.min_window = min_window
        self.logger = logging.getLogger('garch.forecaster')
        
    def generate_expanding_windows(self, returns: np.ndarray, 
                                 start_idx: int = 0) -> List[np.ndarray]:
        """Generate expanding windows for estimation"""
        n_obs = len(returns)
        windows = []
        
        for end_idx in range(self.min_window + start_idx, n_obs + 1):
            window = returns[start_idx:end_idx]
            windows.append(window)
            
        self.logger.info(f"Generated {len(windows)} windows from {n_obs} observations")
        return windows
        
    def generate_forecasts(self, window: np.ndarray, 
                         garch_results: List[GARCHResult]) -> Dict[str, float]:
        """Generate forecasts for each GARCH model"""
        try:
            forecasts = {}
            
            for result in garch_results:
                model_name = f"{result.model_type}_{result.distribution}"
                # Handle both scalar and array forecasts
                if np.isscalar(result.forecasts_annualized):
                    forecasts[model_name] = np.array([result.forecasts_annualized])
                else:
                    # Ensure array is 1D
                    forecasts[model_name] = np.asarray(result.forecasts_annualized).ravel()
            
            # Calculate ensemble statistics from forecasts
            stats = self.calculate_ensemble_stats(forecasts)
            return stats
            
        except Exception as e:
            self.logger.error(f"Error generating forecasts: {str(e)}")
            raise
            
    def calculate_ensemble_stats(self, forecasts: Dict[str, np.ndarray],
                               implied_vols: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate ensemble statistics from forecasts"""
        try:
            # Safety check for empty forecasts
            if not forecasts or len(forecasts) == 0:
                return self._empty_stats()
            
            # Convert all forecasts to scalar values
            processed_forecasts = {}
            for name, forecast in forecasts.items():
                if forecast is not None:
                    # Convert to numpy array if needed
                    if not isinstance(forecast, np.ndarray):
                        forecast = np.array([forecast])
                    # Take mean if array
                    if len(forecast) > 0:
                        processed_forecasts[name] = float(np.nanmean(forecast))
            
            if not processed_forecasts:
                return self._empty_stats()
            
            # Calculate statistics from scalar values
            values = np.array(list(processed_forecasts.values()))
            stats = {
                'GEV': float(np.nanmean(values)),
                'EVOEV': float(np.nanstd(values)),
                'DEV': float(np.nanstd(values)),  # Using std for scalar values
                'KEV': 0.0,  # No time series for scalar values
                'SEVTS': 0.0,  # No term structure for scalar values
                'n_models': len(processed_forecasts)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble stats: {str(e)}")
            return self._empty_stats()
    
    def _empty_stats(self) -> Dict[str, float]:
        """Return empty statistics dictionary"""
        return {
            'GEV': np.nan,
            'EVOEV': np.nan,
            'DEV': np.nan,
            'KEV': np.nan,
            'SEVTS': np.nan,
            'n_models': 0
        }
    
    def process_window(self, window: np.ndarray) -> np.ndarray:
        """Process a single window of returns data"""
        # Return the window array directly, not a dict
        return window
    
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
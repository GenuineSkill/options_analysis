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
        
    def generate_expanding_windows(self,
                                returns: np.ndarray,
                                dates: np.ndarray,
                                forecast_horizon: int = 252) -> List[ForecastWindow]:
        """
        Generate forecasts using expanding windows
        
        Parameters:
        -----------
        returns : array-like
            Return series to analyze
        dates : array-like
            Dates corresponding to returns
        forecast_horizon : int
            Horizon for forecasts in business days
            
        Returns:
        --------
        List[ForecastWindow]
            Forecast results for each window
        """
        if len(returns) < self.min_window:
            raise ValueError(f"Insufficient data: {len(returns)} < {self.min_window}")
            
        # Convert dates to pandas datetime and create index
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        
        windows = []
        start_idx = 0
        
        while start_idx < len(returns):
            # Find end date that gives us the required number of business days
            window_start = dates[start_idx]
            target_bdays = self.min_window
            
            # Use business day range to find end date
            window_end_date = pd.bdate_range(start=window_start, periods=target_bdays)[-1]
            
            # Use get_indexer instead of deprecated get_loc
            end_idxs = dates.get_indexer([window_end_date], method='ffill')
            end_idx = end_idxs[0] + 1
            
            if end_idx > len(returns):
                break
                
            # Get window data
            window_returns = returns[start_idx:end_idx]
            window_end = dates[end_idx-1]
            
            # Verify window size
            window_bdays = len(pd.bdate_range(window_start, window_end))
            if window_bdays < self.min_window - 5:  # Allow some flexibility
                logger.warning(f"Insufficient business days in window: {window_bdays}")
                break
            
            try:
                # Estimate models for window
                garch_results = self.estimator.estimate_models(
                    window_returns,
                    forecast_horizon=forecast_horizon
                )
                
                if garch_results:  # Only add window if we got valid results
                    # Calculate ensemble statistics
                    stats = self.estimator.calculate_ensemble_stats(
                        garch_results,
                        tau=min(forecast_horizon, self.step_size)
                    )
                    
                    # Store results
                    window = ForecastWindow(
                        start_date=window_start,
                        end_date=window_end,
                        returns=window_returns,
                        garch_results=garch_results,
                        ensemble_stats=stats
                    )
                    windows.append(window)
                
            except Exception as e:
                logger.error(f"Error processing window {window_start}-{window_end}: {str(e)}")
                
            # Move to next window using business days
            next_start = pd.bdate_range(start=window_start, periods=self.step_size+1)[-1]
            
            # Use get_indexer instead of deprecated get_loc
            start_idxs = dates.get_indexer([next_start], method='ffill')
            start_idx = start_idxs[0]
            
            if start_idx < 0:  # If date not found
                start_idx += self.step_size
                if start_idx >= len(returns):
                    break
        
        self.forecast_windows = windows
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
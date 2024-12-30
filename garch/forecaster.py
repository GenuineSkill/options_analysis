from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging
from datetime import datetime
from .estimator import GARCHEstimator, GARCHResult
import pickle
from pathlib import Path
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

@dataclass
class ForecastWindow:
    """Container for expanding window forecast results"""
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: List[GARCHResult]
    ensemble_stats: Dict[str, Dict[str, float]]

class GARCHForecaster:
    """Manages expanding window forecasts and ensemble statistics"""
    
    def __init__(self, min_observations: int = 1260,
                 n_simulations: int = 1000,
                 random_seed: Optional[int] = None,
                 checkpoint_dir: Optional[Path] = None):
        """Initialize forecaster with GARCH estimator"""
        self.min_observations = min_observations
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.checkpoint_dir = checkpoint_dir
        
        self.estimator = GARCHEstimator(
            min_observations=min_observations,
            n_simulations=n_simulations,
            random_seed=random_seed,
            checkpoint_dir=checkpoint_dir
        )
        self.windows: List[ForecastWindow] = []
        
    def calculate_ensemble_stats(self, garch_results: List[GARCHResult],
                               horizons: Dict[str, int] = None) -> Dict[str, Dict[str, float]]:
        """Calculate ensemble statistics for each horizon"""
        if not garch_results:
            return {}
        
        if horizons is None:
            horizons = {
                '1M': 21, '2M': 42, '3M': 63,
                '6M': 126, '12M': 252
            }
        
        stats = {}
        n_models = len(garch_results)
        
        try:
            # Validate forecast paths
            forecast_paths = []
            for result in garch_results:
                if result.forecast_path is None or np.isnan(result.forecast_path).any():
                    self.logger.warning("Invalid forecast path detected")
                    continue
                forecast_paths.append(result.forecast_path.reshape(-1))
            
            if not forecast_paths:
                raise ValueError("No valid forecast paths available")
            
            # Stack forecasts
            all_forecasts = np.stack(forecast_paths)  # Shape: (n_models, 252)
            
            for horizon_name, tau in horizons.items():
                # Get forecasts up to horizon τ
                horizon_forecasts = all_forecasts[:, :tau]
                
                # Validate forecasts before calculations
                if np.isnan(horizon_forecasts).any():
                    self.logger.warning(f"NaN values in forecasts for horizon {horizon_name}")
                    continue
                    
                # Calculate statistics with error checking
                try:
                    daily_means = np.mean(horizon_forecasts, axis=0)
                    daily_stds = np.std(horizon_forecasts, axis=0)
                    
                    stats[horizon_name] = {
                        'GEV': float(np.mean(daily_means)),
                        'EVOEV': float(np.std(daily_means)),
                        'DEV': float(np.mean(daily_stds)),
                        'KEV': float(scipy_stats.skew(horizon_forecasts.flatten())),
                        'SEVTS': float(np.polyfit(np.arange(tau), daily_means, 1)[0]),
                        'n_models': len(forecast_paths)
                    }
                except Exception as e:
                    self.logger.error(f"Error calculating stats for horizon {horizon_name}: {str(e)}")
                    continue
                    
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating ensemble stats: {str(e)}")
            return {}

    def generate_expanding_windows(self,
                                 returns: np.ndarray,
                                 dates: pd.DatetimeIndex,
                                 min_observations: Optional[int] = None) -> List[ForecastWindow]:
        """Generate expanding window forecasts"""
        if min_observations is None:
            min_observations = self.estimator.min_observations
            
        self.windows = []
        n_windows = len(dates) - min_observations
        
        logger.info(f"Generating {n_windows} expanding windows...")
        
        for i in range(n_windows):
            window_end = i + min_observations
            window_returns = returns[i:window_end]
            
            # Estimate GARCH models
            garch_results = self.estimator.estimate_models(
                window_returns,
                date=dates[window_end]
            )
            
            # Calculate ensemble statistics
            ensemble_stats = self.calculate_ensemble_stats(garch_results)
            
            # Create window object
            window = ForecastWindow(
                start_date=dates[i],
                end_date=dates[window_end],
                returns=window_returns,
                garch_results=garch_results,
                ensemble_stats=ensemble_stats
            )
            
            self.windows.append(window)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i + 1}/{n_windows} windows")
                
        return self.windows

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        if not self.windows:
            raise ValueError("No windows available")
            
        records = []
        for window in self.windows:
            for horizon, stats in window.ensemble_stats.items():
                record = {
                    'date': window.end_date,
                    'horizon': horizon,
                    **stats
                }
                records.append(record)
                
        return pd.DataFrame(records)

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
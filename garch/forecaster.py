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
from .models import ForecastWindow, GARCHResult

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
        self.logger = logging.getLogger('garch.forecaster')
        
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
            # Stack forecasts and validate
            forecast_paths = []
            model_info = []  # Track which model produced which forecast
            for result in garch_results:
                if result.forecast_path is None or np.isnan(result.forecast_path).any():
                    self.logger.warning(f"Invalid forecast path detected for {result.model_type}-{result.distribution}")
                    continue
                forecast_paths.append(result.forecast_path)
                model_info.append(f"{result.model_type}-{result.distribution}")
            
            if not forecast_paths:
                raise ValueError("No valid forecast paths available")
            
            # Stack forecasts (already in percentage terms)
            all_forecasts = np.stack(forecast_paths)
            
            for horizon_name, tau in horizons.items():
                horizon_forecasts = all_forecasts[:, :tau]
                
                if np.isnan(horizon_forecasts).any():
                    self.logger.warning(f"NaN values in forecasts for horizon {horizon_name}")
                    continue
                
                # Audit extreme values
                daily_means = np.mean(horizon_forecasts, axis=0)
                mean_vol = np.mean(daily_means)
                
                # If mean volatility is extreme, log detailed diagnostics
                if mean_vol > 100 or mean_vol < 5:
                    self.logger.warning(f"\nExtreme volatility detected for {horizon_name}:")
                    self.logger.warning(f"Mean volatility: {mean_vol:.2f}%")
                    self.logger.warning("\nModel-by-day forecast matrix:")
                    
                    # Create a formatted table of forecasts
                    header = "Day |" + "|".join(f"{m:^15}" for m in model_info)
                    self.logger.warning("-" * len(header))
                    self.logger.warning(header)
                    self.logger.warning("-" * len(header))
                    
                    for day in range(tau):
                        row = f"{day+1:3d} |"
                        for model in range(len(model_info)):
                            row += f"{horizon_forecasts[model,day]:15.2f}|"
                        self.logger.warning(row)
                    self.logger.warning("-" * len(header))
                    
                    # Log model parameters
                    self.logger.warning("\nModel parameters:")
                    for i, result in enumerate(garch_results):
                        self.logger.warning(f"\n{model_info[i]}:")
                        for param, value in result.params.items():
                            self.logger.warning(f"  {param}: {value:.6f}")
                
                try:
                    daily_stds = np.std(horizon_forecasts, axis=0)
                    
                    stats[horizon_name] = {
                        'GEV': float(mean_vol),  # Simple mean as per Durham
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
                                 forecast_horizon: int = 252) -> List[ForecastWindow]:
        """Generate expanding window forecasts"""
        
        if len(returns) < self.min_observations:
            raise ValueError("Insufficient observations for estimation")
        
        n_windows = len(returns) - self.min_observations + 1
        
        # Add window size validation
        self.logger.warning(
            f"\nExpanding window setup:"
            f"\n  Total observations: {len(returns)}"
            f"\n  Min observations: {self.min_observations}"
            f"\n  Number of windows: {n_windows}"
            f"\n  Date range: {dates[0]} to {dates[-1]}"
            f"\n  First window: {dates[0]} to {dates[self.min_observations-1]}"
            f"\n  Last window: {dates[-self.min_observations]} to {dates[-1]}"
        )
        
        self.windows = []
        
        logger.info(f"Generating {n_windows} expanding windows...")
        
        for i in range(n_windows):
            window_end = i + self.min_observations
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

    def calculate_gev(self, forecast_paths: List[np.ndarray]) -> float:
        """
        Calculate Generalized Expected Volatility (GEV) from forecast paths.
        GEV combines mean volatility forecast with higher moments to capture uncertainty.
        """
        try:
            # Convert paths to numpy array and validate
            paths = np.array(forecast_paths)
            if len(paths) == 0:
                self.logger.warning("Empty forecast paths")
                return None
            
            # Remove any invalid values
            if np.any(np.isnan(paths)) or np.any(np.isinf(paths)):
                self.logger.warning("Invalid values in forecast paths, attempting to clean")
                paths = paths[~np.isnan(paths) & ~np.isinf(paths)]
                if len(paths) == 0:
                    return None
            
            # Clip extreme values (keep within 5-150% annualized vol)
            paths = np.clip(paths, 5, 150)
            
            # Calculate components with better numerical stability
            # 1. Mean volatility (base forecast)
            mean_vol = np.mean(paths)
            
            # 2. Normalized volatility of volatility (uncertainty)
            vol_of_vol = np.std(paths) / mean_vol  # Normalize by mean
            
            # 3. Normalized skewness (asymmetry)
            skew = scipy_stats.skew(paths.flatten())
            norm_skew = np.tanh(skew / 2)  # Bound between -1 and 1
            
            # 4. Normalized excess kurtosis (tail risk)
            kurt = scipy_stats.kurtosis(paths.flatten())
            norm_kurt = np.tanh(kurt / 6)  # Bound between -1 and 1
            
            # Combine components with controlled scaling
            gev = mean_vol * (
                1.0 +  # Base volatility
                0.2 * vol_of_vol +  # Add 0-20% for uncertainty
                0.1 * norm_skew +   # Add ±10% for asymmetry
                0.1 * norm_kurt     # Add ±10% for tail risk
            )
            
            # Validate final result
            if not (10 < gev < 100):
                self.logger.warning(
                    f"GEV outside expected range: {gev:.1f}% "
                    f"(mean={mean_vol:.1f}%, vov={vol_of_vol:.2f}, "
                    f"skew={norm_skew:.2f}, kurt={norm_kurt:.2f})"
                )
                return None
            
            # Log components for debugging
            self.logger.debug(
                f"GEV components: mean={mean_vol:.1f}%, "
                f"vov={vol_of_vol:.2f}, skew={norm_skew:.2f}, "
                f"kurt={norm_kurt:.2f}, final={gev:.1f}%"
            )
            
            return gev
            
        except Exception as e:
            self.logger.error(f"Error calculating GEV: {str(e)}")
            return None

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
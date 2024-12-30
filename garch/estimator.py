from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from arch import arch_model
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import warnings
import logging
from pathlib import Path
import pickle
from .checkpoint import CheckpointManager
from datetime import datetime
from .models import GARCHResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GARCHResult:
    """Container for GARCH estimation results"""
    model_type: str
    distribution: str
    params: Dict[str, float]
    forecast_path: np.ndarray  # Store mean forecast path
    volatility_path: np.ndarray  # Store historical volatilities

class GARCHEstimator:
    """Estimates and manages ensemble of GARCH models"""
    
    def __init__(self, min_observations: int = 1260,
                 n_simulations: int = 1000,
                 random_seed: Optional[int] = None,
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize estimator
        
        Args:
            min_observations: Minimum number of observations for estimation
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            checkpoint_dir: Directory for checkpointing results
        """
        self.min_observations = min_observations
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        # Load holiday calendar
        holiday_file = Path(__file__).parent.parent / "data_manager/data/market_holidays_1987_2027.csv"
        self.holiday_calendar = pd.read_csv(holiday_file)
        self.holiday_calendar['date'] = pd.to_datetime(self.holiday_calendar['date'])
        
        # Define model specifications with GJR-GARCH
        self.model_specs = [
            ('garch', 'normal', {'p': 1, 'o': 0, 'q': 1}),
            ('garch', 'studentst', {'p': 1, 'o': 0, 'q': 1}),
            ('egarch', 'normal', {'p': 1, 'q': 1}),
            ('egarch', 'studentst', {'p': 1, 'q': 1}),
            # GJR-GARCH implemented as GARCH with o=1
            ('gjrgarch', 'normal', {'p': 1, 'o': 1, 'q': 1}),
            ('gjrgarch', 'studentst', {'p': 1, 'o': 1, 'q': 1})
        ]
        
        self.checkpoint_manager = None
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
            
        self.logger = logging.getLogger('garch.estimator')

    def _get_next_n_trading_days(self, start_date: datetime, index_id: str, n: int = 252) -> pd.DatetimeIndex:
        """Get the next n trading days from start_date for given index"""
        holiday_col = f'{index_id}_holiday'
        
        # Get future dates from calendar
        future_dates = self.holiday_calendar[
            (self.holiday_calendar['date'] > start_date)
        ].copy()
        
        # Filter out holidays for this index
        trading_days = future_dates[future_dates[holiday_col] == 0]['date']
        
        # Take first n trading days
        return trading_days.head(n).index

    def _validate_forecasts(self, forecast_path: np.ndarray, 
                           model_type: str, 
                           distribution: str) -> bool:
        """Validate forecast values are reasonable"""
        try:
            # Check for NaN or inf
            if np.any(np.isnan(forecast_path)) or np.any(np.isinf(forecast_path)):
                self.logger.warning(f"NaN/Inf in forecasts for {model_type}-{distribution}")
                return False
                
            # Check reasonable range (8% to 100% annualized volatility)
            if np.any(forecast_path < 8) or np.any(forecast_path > 100):
                self.logger.warning(
                    f"Unreasonable volatility range for {model_type}-{distribution}: "
                    f"min={np.min(forecast_path):.1f}%, max={np.max(forecast_path):.1f}%"
                )
                return False
                
            # Check for extreme changes
            changes = np.diff(forecast_path)
            if np.any(np.abs(changes) > 30):
                self.logger.warning(
                    f"Extreme volatility changes for {model_type}-{distribution}: "
                    f"max change={np.max(np.abs(changes)):.1f}%"
                )
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating forecasts: {str(e)}")
            return False

    def _estimate_single_model(self, returns: np.ndarray,
                             model_spec: tuple,
                             date: datetime,
                             index_id: str) -> Optional[GARCHResult]:
        """
        Estimate single GARCH model and generate mean forecast path
        """
        try:
            model_type, distribution, _ = model_spec
            
            # Even tighter bounds and better starting values
            if model_type == 'egarch':
                bounds = {
                    'mu': (-0.0005, 0.0005),
                    'omega': (-1.0, -0.1),
                    'alpha[1]': (0.05, 0.15),
                    'beta[1]': (0.90, 0.98)
                }
                starting_points = [
                    {'mu': 0, 'omega': -0.5, 'alpha[1]': 0.10, 'beta[1]': 0.94},
                    {'mu': 0, 'omega': -0.3, 'alpha[1]': 0.08, 'beta[1]': 0.92}
                ]
            else:  # GARCH and GJR-GARCH
                bounds = {
                    'mu': (-0.0005, 0.0005),
                    'omega': (1e-6, 2e-5),
                    'alpha[1]': (0.03, 0.10),
                    'beta[1]': (0.87, 0.93)
                }
                starting_points = [
                    {'mu': 0, 'omega': 5e-6, 'alpha[1]': 0.05, 'beta[1]': 0.90},
                    {'mu': 0, 'omega': 1e-5, 'alpha[1]': 0.07, 'beta[1]': 0.89}
                ]
                
                if model_type == 'gjrgarch':
                    bounds['gamma[1]'] = (0.03, 0.10)
                    for p in starting_points:
                        p['gamma[1]'] = 0.05
            
            if distribution == 'studentst':
                bounds['nu'] = (8, 15)  # Much tighter bounds on degrees of freedom
                for p in starting_points:
                    p['nu'] = 10.0  # Start in middle of range
                
            # Try each starting point
            best_result = None
            best_llf = -np.inf
            
            for i, start_vals in enumerate(starting_points):
                try:
                    result = model.fit(
                        disp='off',
                        starting_values=start_vals,
                        bounds=bounds,
                        options={'maxiter': 1000}
                    )
                    
                    # Only accept if parameters are valid
                    if self._validate_garch_params(result.params):
                        if result.loglikelihood > best_llf:
                            best_result = result
                            best_llf = result.loglikelihood
                            
                except Exception as e:
                    self.logger.debug(f"Estimation attempt {i+1} failed: {str(e)}")
                    continue
                
            if best_result is None:
                self.logger.warning(
                    f"Failed to estimate {model_type}-{distribution} model "
                    f"with valid parameters"
                )
                return None
            
            # Ensure returns are in decimal form
            if np.std(returns) > 0.1:  # If std > 10%, assume percentage
                self.logger.warning("Converting returns from percentage to decimal")
                returns = returns / 100
                
            # Log data characteristics
            self.logger.info(
                f"\nData summary for {model_type}-{distribution}:"
                f"\n  Mean: {np.mean(returns):.6f}"
                f"\n  Std:  {np.std(returns):.6f}"
                f"\n  Skew: {scipy.stats.skew(returns):.6f}"
                f"\n  Kurt: {scipy.stats.kurtosis(returns):.6f}"
            )
            
            # Configure model
            model = arch_model(
                returns,
                vol=model_type.lower(),
                p=1, o=1 if model_type == 'gjrgarch' else 0, q=1,
                dist=distribution,
                rescale=False  # Important: don't let arch_model rescale
            )
            
            # Generate forecasts (in daily variance terms)
            forecasts = best_result.forecast(horizon=252)
            daily_vol = np.sqrt(forecasts.variance.values[-1, :])
            
            # Convert to annualized volatility only at the end
            annual_vol = daily_vol * np.sqrt(252) * 100
            
            return GARCHResult(
                model_type=model_type,
                distribution=distribution,
                params=best_result.params.to_dict(),
                forecast_path=annual_vol,
                volatility_path=best_result.conditional_volatility * np.sqrt(252) * 100
            )
            
        except Exception as e:
            self.logger.error(f"Error in {model_type}-{distribution}: {str(e)}")
            return None

    def _validate_garch_params(self, params: pd.Series) -> bool:
        """Validate GARCH parameters are reasonable"""
        try:
            # Check mean
            if abs(params.get('mu', 0)) > 0.001:
                self.logger.warning(f"Mean too large: {params['mu']:.6f}")
                return False
            
            # Check omega
            if not (1e-6 <= params.get('omega', 0) <= 1e-4):
                self.logger.warning(f"Omega outside bounds: {params['omega']:.6f}")
                return False
            
            # Check ARCH effect
            if not (0.01 <= params.get('alpha[1]', 0) <= 0.15):
                self.logger.warning(f"Alpha outside bounds: {params['alpha[1]']:.6f}")
                return False
            
            # Check GARCH persistence
            if not (0.80 <= params.get('beta[1]', 0) <= 0.95):
                self.logger.warning(f"Beta outside bounds: {params['beta[1]']:.6f}")
                return False
            
            # Check leverage effect if present
            if 'gamma[1]' in params and not (0 <= params['gamma[1]'] <= 0.15):
                self.logger.warning(f"Gamma outside bounds: {params['gamma[1]']:.6f}")
                return False
            
            # Check degrees of freedom if present
            if 'nu' in params and not (4.1 <= params['nu'] <= 30):
                self.logger.warning(f"Nu outside bounds: {params['nu']:.6f}")
                return False
            
            # Check total persistence
            persistence = params.get('alpha[1]', 0) + params.get('beta[1]', 0)
            if 'gamma[1]' in params:
                persistence += 0.5 * params['gamma[1]']
            
            if not (0.95 <= persistence <= 0.999):
                self.logger.warning(f"Invalid persistence: {persistence:.3f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return False

    def estimate_models(self, returns: np.ndarray,
                       date: Optional[pd.Timestamp] = None,
                       index_id: str = None) -> List[GARCHResult]:
        """
        Estimate all GARCH models with checkpointing
        """
        if len(returns) < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(returns)} < {self.min_observations}"
            )
            
        # Add detailed window size diagnostics
        window_years = len(returns) / 252  # Approximate years of data
        self.logger.warning(
            f"\nWindow diagnostics for {date}:"
            f"\n  Window size: {len(returns)} observations"
            f"\n  Approx years: {window_years:.1f}"
            f"\n  Start date: {date - pd.Timedelta(days=len(returns))} "
            f"\n  End date: {date}"
        )
        
        # Input validation and diagnostics
        returns = np.array(returns, dtype=np.float64)
        returns_stats = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'abs_max': np.max(np.abs(returns))
        }
        
        self.logger.info(
            f"Input returns stats for {date}:\n"
            f"  Mean: {returns_stats['mean']:.6f}\n"
            f"  Std:  {returns_stats['std']:.6f}\n"
            f"  Min:  {returns_stats['min']:.6f}\n"
            f"  Max:  {returns_stats['max']:.6f}\n"
            f"  |Max|: {returns_stats['abs_max']:.6f}"
        )
        
        # Pre-validate returns
        if returns_stats['abs_max'] > 1:
            self.logger.info("Pre-scaling returns (percentage to decimal)")
            returns = returns / 100
            
        # Verify scaling
        if np.max(np.abs(returns)) > 0.5:
            self.logger.error(f"Returns too large after scaling: max={np.max(np.abs(returns)):.2f}")
            return []
            
        if np.std(returns) < 0.0001 or np.std(returns) > 0.1:
            self.logger.error(f"Returns volatility outside reasonable range: std={np.std(returns):.6f}")
            return []
        
        try:
            # Try to load from checkpoint
            if self.checkpoint_manager and date:
                checkpoint = self.checkpoint_manager.load_checkpoint(date)
                if checkpoint and self._validate_checkpoint(checkpoint):
                    self.logger.debug(f"Using checkpoint for {date}")
                    return checkpoint
            
            # Estimate all models in parallel
            with ProcessPoolExecutor() as executor:
                futures = []
                for model_spec in self.model_specs:
                    future = executor.submit(
                        self._estimate_single_model,
                        returns.copy(),  # Pass a copy to each worker
                        model_spec,
                        date,
                        index_id or 'SPX'
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
            # Log estimation results
            if results:
                forecasts = [r.forecast_path for r in results]
                self.logger.info(
                    f"Model estimation summary for {date}:\n"
                    f"  Models estimated: {len(results)}\n"
                    f"  Mean forecast: {np.mean(forecasts):.1f}%\n"
                    f"  Std forecast: {np.std(forecasts):.1f}%\n"
                    f"  Range: [{np.min(forecasts):.1f}%, {np.max(forecasts):.1f}%]"
                )
            
            # Save checkpoint if we have valid results
            if self.checkpoint_manager and date and results:
                self.checkpoint_manager.save_checkpoint(date, results)
                
            return results
            
        except Exception as e:
            self.logger.error(f"Error estimating models: {str(e)}")
            raise

    def _validate_checkpoint(self, results: List[GARCHResult]) -> bool:
        """Validate checkpoint results"""
        try:
            if not results:
                return False
                
            for result in results:
                # Check if result has all required attributes
                if not all(hasattr(result, attr) for attr in 
                          ['model_type', 'distribution', 'params', 
                           'forecast_path', 'volatility_path']):
                    return False
                    
                # Check if forecasts are valid
                if (not isinstance(result.forecast_path, np.ndarray) or 
                    len(result.forecast_path) == 0):
                    return False
                    
                # Check if volatilities are valid
                if (not isinstance(result.volatility_path, np.ndarray) or 
                    len(result.volatility_path) == 0):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating checkpoint: {str(e)}")
            return False

# Example usage
if __name__ == '__main__':
    # Example data preparation
    returns = np.random.normal(0, 1, 2000) * 0.01  # 2000 days of returns
    
    # Initialize estimator
    estimator = GARCHEstimator(
        min_observations=1260,
        n_simulations=1000,
        random_seed=42
    )
    
    # Estimate models
    results = estimator.estimate_models(returns, forecast_horizon=252)
    
    # Calculate ensemble statistics for 1-month horizon
    stats = estimator.calculate_ensemble_stats(results, tau=21)
    print("Ensemble statistics (annualized volatility %):")
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}%")
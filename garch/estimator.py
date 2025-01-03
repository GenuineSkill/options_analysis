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
from .models import GARCHResult, ForecastWindow
from .holiday_handler import HolidayHandler
from data_manager.database import GARCHDatabase

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
    
    def __init__(self, min_observations: int = 3000,
                 n_simulations: int = 1000,
                 random_seed: int = 42,
                 checkpoint_dir: str = None):
        """
        Initialize estimator
        
        Args:
            min_observations: Minimum number of daily returns required (12 years)
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
            checkpoint_dir: Directory to store checkpoints
        """
        self.min_observations = min_observations
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize holiday handler
        self.holiday_handler = HolidayHandler()
        
        # Load holiday calendar from absolute path
        holiday_file = Path(r"C:\Users\w.sterling\volatility\options_analysis\data_manager\data\market_holidays_1987_2027.csv")
        self.holiday_calendar = pd.read_csv(holiday_file)
        self.holiday_calendar['date'] = pd.to_datetime(self.holiday_calendar['date'])
        
        # Define model specifications
        self.model_specs = [
            ('garch', 'normal', 1),
            ('garch', 'studentst', 1),
            ('egarch', 'normal', 1),
            ('egarch', 'studentst', 1),
            ('gjrgarch', 'normal', 1),
            ('gjrgarch', 'studentst', 1)
        ]
        
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

    def _validate_forecasts(self, forecasts: np.ndarray) -> bool:
        """Validate forecast values are reasonable"""
        try:
            # Check for NaN or inf
            if np.any(np.isnan(forecasts)) or np.any(np.isinf(forecasts)):
                self.logger.warning(f"Found NaN/inf in forecasts")
                return False
                
            # Check reasonable range (5% to 100% annualized)
            if np.any(forecasts < 5) or np.any(forecasts > 100):
                self.logger.warning(
                    f"Forecasts outside range [5%, 100%]: min={np.min(forecasts):.1f}%, "
                    f"max={np.max(forecasts):.1f}%"
                )
                return False
                
            # Check for explosive behavior
            changes = np.diff(forecasts)
            if np.any(np.abs(changes) > 10):  # Max 10% daily change
                self.logger.warning(
                    f"Large daily changes detected: max change = {np.max(np.abs(changes)):.1f}%"
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
        """Estimate a single GARCH model specification"""
        try:
            model_type, distribution, p = model_spec
            
            # Initialize random state
            random_state = np.random.RandomState(self.random_seed)
            
            # Scale returns to percentage form for estimation
            returns_std = np.std(returns)
            self.logger.info(f"Input returns std: {returns_std:.6f}")
            returns = returns * 100  # Convert decimal returns to percentage
            
            # Initialize model with correct parameters
            vol_model = 'garch' if model_type == 'gjrgarch' else model_type
            model = arch_model(
                returns,
                mean='constant',
                vol=vol_model,
                p=1,  # Fixed p=1 for all models
                o=1 if model_type == 'gjrgarch' else 0,  # Leverage term for GJR-GARCH
                q=1,  # Fixed q=1 for all models
                dist=distribution,
                rescale=True  # Enable rescaling for numerical stability
            )
            
            # Fit model
            result = model.fit(
                disp='off',
                show_warning=False,
                options={'maxiter': 1000},
                update_freq=0
            )
            
            # Generate forecasts based on model type
            if model_type == 'egarch':
                # Generate multiple paths for EGARCH using simulation
                variance_paths = np.zeros((252, self.n_simulations))
                
                for sim in range(self.n_simulations):
                    forecast = result.forecast(
                        horizon=252,
                        method='simulation',
                        simulations=1,
                        random_state=random_state,
                        reindex=False
                    )
                    # Extract variance path - shape is (T, horizon, 1)
                    var_path = forecast.variance.values
                    if len(var_path.shape) == 3:
                        var_path = var_path[-1, :, 0]  # Take last observation, first simulation
                    else:
                        var_path = var_path[-1, :]  # Take last observation
                    variance_paths[:, sim] = var_path
                    
                self.logger.info(f"EGARCH paths shape: {variance_paths.shape}")
            else:
                # Use analytic + noise for other models
                base_forecast = result.forecast(
                    horizon=252,
                    method='analytic',
                    reindex=False
                )
                base_variance = base_forecast.variance.values[-1]
                
                # Generate multiple paths with reduced noise
                variance_paths = np.zeros((252, self.n_simulations))
                for i in range(self.n_simulations):
                    noise = 1 + random_state.normal(0, 0.02, size=252)  # Reduced noise
                    variance_paths[:, i] = base_variance * noise
            
            self.logger.info(f"Generated variance paths shape: {variance_paths.shape}")
            
            # Compute mean path using median across simulations
            mean_var_path = np.median(variance_paths, axis=1)
            
            # Convert to annualized volatility percentage
            daily_vol = np.sqrt(mean_var_path)  # Convert variance to volatility
            annual_vol = daily_vol * np.sqrt(252)  # Annualize
            
            # Log forecast statistics
            self.logger.info(
                f"Raw forecast stats for {model_type}-{distribution}:\n"
                f"  Daily vol (mean): {np.mean(daily_vol):.6f}\n"
                f"  Annual vol (mean): {np.mean(annual_vol):.6f}"
            )
            
            # Log final forecast statistics
            self.logger.info(
                f"Final forecast stats for {model_type}-{distribution} (percentage):\n"
                f"  Mean: {np.mean(annual_vol):.1f}%\n"
                f"  Std:  {np.std(annual_vol):.1f}%\n"
                f"  Min:  {np.min(annual_vol):.1f}%\n"
                f"  Max:  {np.max(annual_vol):.1f}%"
            )
            
            # Validate forecast path
            if not self._validate_forecast_path(annual_vol):
                return None
            
            return GARCHResult(
                model_type=model_type,
                distribution=distribution,
                params=dict(result.params),
                forecast_path=annual_vol,
                volatility_path=np.sqrt(result.conditional_volatility) * np.sqrt(252)
            )
            
        except Exception as e:
            self.logger.error(f"Error in {model_type}-{distribution}: {str(e)}")
            return None

    def _validate_garch_params(self, params: dict, model_type: str) -> bool:
        """Validate GARCH parameters are reasonable"""
        try:
            # Basic parameter bounds - adjusted for typical ranges
            param_bounds = {
                'omega': (1e-7, 0.05),  # Increased upper bound significantly
                'alpha[1]': (0.01, 0.2),
                'beta[1]': (0.7, 0.99),
                'gamma[1]': (0.01, 0.2)  # For GJR-GARCH
            }
            
            # Check each parameter
            for param, (min_val, max_val) in param_bounds.items():
                if param in params:
                    value = params[param]
                    if not (min_val <= value <= max_val):
                        self.logger.warning(
                            f"Parameter {param}={value:.6f} outside bounds "
                            f"[{min_val}, {max_val}]"
                        )
                        return False
            
            # Check persistence
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            gamma = params.get('gamma[1]', 0)
            persistence = alpha + beta + (gamma/2 if model_type == 'gjrgarch' else 0)
            
            if not (0.95 <= persistence <= 0.999):
                self.logger.warning(f"Invalid persistence {persistence:.3f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {str(e)}")
            return False

    def _validate_forecast_path(self, path: np.ndarray) -> bool:
        """Validate entire forecast path"""
        try:
            # No zeros or negatives
            if np.any(path <= 0):
                self.logger.warning("Found non-positive values in forecast path")
                return False
            
            # Check reasonable range (5% to 100% annualized)
            if np.any(path < 5) or np.any(path > 100):
                self.logger.warning(
                    f"Forecast outside range [5%, 100%]: min={np.min(path):.1f}%, "
                    f"max={np.max(path):.1f}%"
                )
                return False
            
            # Check for monotonicity violations
            # Volatility shouldn't change by more than 50% between days
            changes = np.abs(np.diff(path) / path[:-1])
            if np.any(changes > 0.50):
                self.logger.warning(
                    f"Large daily changes detected: max change = {np.max(changes):.1f}%"
                )
                return False
            
            # Long-term mean reversion - relaxed constraint
            if abs(path[-1] - path[0]) / path[0] > 1.0:  # Allow up to 100% change
                self.logger.warning(
                    f"Excessive long-term drift: start={path[0]:.1f}%, end={path[-1]:.1f}%"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating forecast path: {str(e)}")
            return False

    def _prepare_returns(self, returns: np.ndarray, dates: pd.DatetimeIndex, index_id: str) -> np.ndarray:
        """Convert returns to daily log returns if needed and verify trading days"""
        # Verify dates are trading days
        holiday_col = f'{index_id}_holiday'
        holidays = self.holiday_calendar[self.holiday_calendar[holiday_col] == 1]['date']
        if any(date in holidays for date in dates):
            raise ValueError("Input returns contain holidays")
        
        # If returns look like percentages (std > 0.1)
        if np.std(returns) > 0.1:
            self.logger.info("Converting percentage returns to decimal")
            returns = returns / 100
            
        # Verify reasonable range for daily returns
        returns_std = np.std(returns)
        if not (0.005 <= returns_std <= 0.03):
            raise ValueError(f"Returns volatility outside daily range: std={returns_std:.6f}")
            
        # Convert to log returns if not already
        if np.min(returns) < -1:  # Simple returns have lower bound of -1
            raise ValueError("Returns appear to be incorrectly scaled")
            
        log_returns = np.log1p(returns)
        
        self.logger.info(
            f"Prepared log returns:\n"
            f"  Mean: {np.mean(log_returns):.6f}\n"
            f"  Std:  {np.std(log_returns):.6f}"
        )
        
        return log_returns

    def estimate_models(self, returns: np.ndarray, date: datetime, index_id: str) -> List[GARCHResult]:
        """Estimate all GARCH model specifications"""
        try:
            # Get actual dates accounting for trading days
            days_of_returns = len(returns)
            trading_days = pd.date_range(
                start=pd.Timestamp('1987-01-02'),  # First price date
                end=pd.Timestamp(date),
                freq='B'  # Business days
            )
            
            # Filter out holidays for this index
            holiday_col = f'{index_id}_holiday'
            holidays = self.holiday_calendar[self.holiday_calendar[holiday_col] == 1]['date']
            trading_days = [
                day for day in trading_days 
                if day not in holidays
            ]
            
            price_start_date = trading_days[0]
            return_start_date = trading_days[1]  # First return requires two prices
            
            # Log window diagnostics
            self.logger.warning(f"""
Window diagnostics for {date}:
  Returns: {days_of_returns} observations
  Price window: {price_start_date.strftime('%Y-%m-%d')} to {pd.Timestamp(date).strftime('%Y-%m-%d')}
  Return window: {return_start_date.strftime('%Y-%m-%d')} to {pd.Timestamp(date).strftime('%Y-%m-%d')}
  Approx years: {days_of_returns/252:.1f}
  Trading days per year: {len(trading_days)/((trading_days[-1] - trading_days[0]).days/365.25):.1f}
""")
            
            # Log input data statistics
            self.logger.info(f"Input returns stats for {date}:")
            stats = {
                'Mean': returns.mean(),
                'Std': returns.std(),
                'Min': returns.min(),
                'Max': returns.max(),
                '|Max|': max(abs(returns.min()), abs(returns.max())),
                'First date': return_start_date.strftime('%Y-%m-%d'),
                'Last date': pd.Timestamp(date).strftime('%Y-%m-%d')
            }
            for name, value in stats.items():
                if isinstance(value, str):
                    self.logger.info(f"  {name}: {value}")
                else:
                    self.logger.info(f"  {name}: {value:8.6f}")
            
            results = []
            for model_spec in self.model_specs:
                try:
                    result = self._estimate_single_model(
                        returns=returns,
                        model_spec=model_spec,
                        date=date,
                        index_id=index_id
                    )
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error estimating {model_spec}: {str(e)}")
                    continue
                
            if not results:
                raise ValueError("No valid models estimated")
            
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

    def _convert_to_annualized_vol(self, daily_variance: np.ndarray) -> np.ndarray:
        """Convert daily variance to annualized volatility percentage"""
        # Use actual number of trading days (~252) for annualization
        trading_days_per_year = len(self.holiday_calendar[
            self.holiday_calendar[f'SPX_holiday'] == 0
        ]) / ((self.holiday_calendar['date'].max() - 
               self.holiday_calendar['date'].min()).days / 365.25)
        
        # Convert variance to volatility
        daily_vol = np.sqrt(daily_variance)
        
        # Annualize using actual trading days and convert to percentage
        annual_vol_pct = daily_vol * np.sqrt(trading_days_per_year) * 100
        
        return annual_vol_pct

    def _validate_input_data(self, returns: np.ndarray, dates: pd.DatetimeIndex, index_id: str) -> bool:
        """Validate input data meets requirements"""
        try:
            # Check dates are within valid range
            if not all(self.holiday_handler.validate_date(date) for date in [dates[0], dates[-1]]):
                self.logger.error(
                    f"Dates outside valid range: {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}"
                )
                return False
            
            # Verify dates are trading days
            if not all(self.holiday_handler.is_trading_day(date, index_id) for date in dates):
                self.logger.error("Input returns contain holidays")
                return False
            
            # Check for sufficient observations
            if len(returns) < self.min_observations:
                self.logger.error(f"Insufficient observations: {len(returns)} < {self.min_observations}")
                return False
            
            # Check for missing values
            if np.any(np.isnan(returns)):
                self.logger.error("Input returns contain missing values")
                return False
            
            # Verify reasonable range for daily returns
            returns_std = np.std(returns)
            if not (0.005 <= returns_std <= 0.03):
                self.logger.error(f"Returns volatility outside daily range: std={returns_std:.6f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False

    def _get_starting_values(self, model_type: str, distribution: str) -> dict:
        """Get reasonable starting values for optimization"""
        base_values = {
            'mu': 0.0,
            'omega': 5e-6,
            'alpha[1]': 0.08,
            'beta[1]': 0.90
        }
        
        if model_type == 'gjrgarch':
            base_values['gamma[1]'] = 0.05
        
        if distribution == 'studentst':
            base_values['nu'] = 8.0
        
        return base_values

    def calculate_ensemble_stats(self, garch_results: List[GARCHResult], 
                               horizons: List[int] = [21, 42, 63, 126, 252]) -> Dict[str, Dict[str, float]]:
        """Calculate ensemble statistics from GARCH model forecasts for multiple horizons"""
        if not garch_results:
            raise ValueError("No valid GARCH results provided")
        
        J = len(garch_results)  # Number of models
        ensemble_stats = {}
        
        for tau in horizons:
            # Extract forecast paths up to tau for all models
            forecasts = np.array([result.forecast_path[:tau] for result in garch_results])  # Shape: (J, tau)
            
            # Calculate ensemble statistics for this horizon
            mean_forecast = np.mean(forecasts, axis=0)  # Average across models for each time point
            gev = np.mean(forecasts)  # Overall mean
            evoev = np.sqrt(np.mean((mean_forecast - gev) ** 2))
            dev = np.mean([np.std(forecasts[:, t], ddof=1) for t in range(tau)])
            kev = np.mean([stats.skew(forecasts[:, t]) for t in range(tau)])
            
            # Calculate SEVTS without dividing by tau
            total_changes = forecasts[:, -1] - forecasts[:, 0]  # End minus beginning for each model
            sevts = np.mean(total_changes)  # Average across models
            
            # Log individual model changes
            self.logger.info(
                f"\nSEVTS calculation for horizon T{tau}:\n"
                f"Individual model changes (end - start):\n"
                + "\n".join(f"  Model {j}: {change:.2f}%" 
                           for j, change in enumerate(total_changes))
                + f"\nFinal SEVTS: {sevts:.2f}%"
            )
            
            # Store results for this horizon
            horizon_key = f'T{tau}'
            ensemble_stats[horizon_key] = {
                'gev': float(gev),
                'evoev': float(evoev),
                'dev': float(dev),
                'kev': float(kev),
                'sevts': float(sevts)
            }
            
            # Log all statistics for this horizon
            self.logger.info(
                f"\nEnsemble statistics for {horizon_key} ({tau} days):\n"
                f"  GEV:   {gev:.1f}%  (Mean expected volatility)\n"
                f"  EVOEV: {evoev:.1f}%  (Std dev of mean forecast deviations)\n"
                f"  DEV:   {dev:.1f}%  (Mean cross-sectional std dev)\n"
                f"  KEV:   {kev:.2f}  (Mean cross-sectional skewness)\n"
                f"  SEVTS: {sevts:.2f}%  (Mean total change in volatility)"
            )
        
        return ensemble_stats

    def calculate_ensemble_stats_multi(self, garch_results: List[GARCHResult], 
                                     horizons: List[int] = None) -> Dict[int, Dict[str, float]]:
        """Calculate ensemble statistics for multiple forecast horizons
        
        Args:
            garch_results: List of GARCH model results
            horizons: List of forecast horizons in trading days. If None, uses standard
                     option expiries [21, 42, 63, 126, 252] (1M, 2M, 3M, 6M, 12M)
                     
        Returns:
            Dictionary mapping horizon -> ensemble statistics
        """
        if horizons is None:
            horizons = [21, 42, 63, 126, 252]  # Standard option expiries
        
        # Validate horizons
        max_horizon = max(horizons)
        if max_horizon > 252:
            raise ValueError(f"Maximum horizon {max_horizon} exceeds 252 trading days")
        
        # Calculate stats for each horizon
        horizon_stats = {}
        for tau in horizons:
            self.logger.info(f"\nCalculating ensemble statistics for {tau}-day horizon:")
            horizon_stats[tau] = self.calculate_ensemble_stats(garch_results, tau)
        
        return horizon_stats

# Example usage
if __name__ == '__main__':
    # Example data preparation
    returns = np.random.normal(0, 1, 2000) * 0.01  # 2000 days of returns
    
    # Initialize estimator
    estimator = GARCHEstimator(
        min_observations=3000,
        n_simulations=1000,
        random_seed=42
    )
    
    # Estimate models
    results = estimator.estimate_models(returns, forecast_horizon=252)
    
    # Calculate ensemble statistics for standard option expiries
    horizon_stats = estimator.calculate_ensemble_stats_multi(results)
    
    # Print results
    print("\nEnsemble statistics by horizon (annualized volatility %):")
    for tau, stats in horizon_stats.items():
        print(f"\n{tau}-day horizon:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}%")
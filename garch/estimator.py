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

    def _estimate_single_model(self, returns: np.ndarray,
                             model_spec: tuple,
                             date: datetime,
                             index_id: str) -> Optional[GARCHResult]:
        """
        Estimate single GARCH model and generate mean forecast path
        """
        try:
            model_type, distribution, garch_params = model_spec
            
            # Map gjrgarch to garch with o=1 for arch_model
            arch_model_type = 'garch' if model_type == 'gjrgarch' else model_type.lower()
            
            # Configure model
            model = arch_model(
                returns,
                vol=arch_model_type,
                dist=distribution,
                p=garch_params.get('p', 1),
                o=garch_params.get('o', 0),
                q=garch_params.get('q', 1),
                rescale=False
            )
            
            # Fit model
            result = model.fit(disp='off', show_warning=False)
            
            # Get next 252 trading days
            forecast_dates = self._get_next_n_trading_days(date, index_id)
            n_forecast_days = len(forecast_dates)
            
            # Generate and average forecast paths
            forecast_path = np.zeros(n_forecast_days)
            for _ in range(self.n_simulations):
                sim = result.forecast(
                    horizon=n_forecast_days,
                    method='simulation',
                    simulations=1,
                    reindex=False
                )
                forecast_path += np.sqrt(sim.variance.values.flatten()) * np.sqrt(252)
            
            forecast_path /= self.n_simulations
            
            return GARCHResult(
                model_type=model_type,
                distribution=distribution,
                params=result.params.to_dict(),
                forecast_path=forecast_path,
                volatility_path=result.conditional_volatility * np.sqrt(252)
            )
            
        except Exception as e:
            self.logger.warning(
                f"Failed to estimate {model_type}-{distribution}: {str(e)}"
            )
            return None

    def estimate_models(self, returns: np.ndarray,
                       date: Optional[pd.Timestamp] = None) -> List[GARCHResult]:
        """Estimate all GARCH models with checkpointing"""
        if len(returns) < self.min_observations:
            raise ValueError(
                f"Insufficient observations: {len(returns)} < {self.min_observations}"
            )
            
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
                        returns,
                        model_spec,
                        date,
                        'index'
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
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
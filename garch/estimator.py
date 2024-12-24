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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set logging level for checkpoint-related messages
checkpoint_logger = logging.getLogger('garch.estimator')
checkpoint_logger.setLevel(logging.INFO)  # Change to logging.DEBUG to see all checkpoint messages

# Suppress known warnings from arch package
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')

@dataclass
class GARCHResult:
    """Container for GARCH estimation results"""
    model_type: str
    distribution: str
    params: Dict[str, float]
    forecasts_annualized: np.ndarray  # Store annualized volatility forecasts
    volatility_annualized: np.ndarray  # Store annualized historical volatilities

class GARCHEstimator:
    """Estimates and manages ensemble of GARCH models"""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.logger = logging.getLogger('garch.estimator')
        self.checkpoint_manager = None
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
            
        # Define model specifications
        self.model_specs = [
            ('GARCH', 'normal'),
            ('GARCH', 'studentst'),
            ('EGARCH', 'normal'),
            ('EGARCH', 'studentst'),
            ('GJR-GARCH', 'normal'),
            ('GJR-GARCH', 'studentst')
        ]
        
        self.min_observations = 252  # Minimum one year of data
            
    def _annualize_variance(self, daily_variance: np.ndarray) -> np.ndarray:
        """Convert daily variance to annualized volatility"""
        return np.sqrt(252 * daily_variance) * 100  # Convert to percentage
        
    def _estimate_single_model(self, 
                             returns: np.ndarray,
                             model_type: str,
                             distribution: str,
                             forecast_horizon: int = 252,
                             n_simulations: int = 1000) -> Optional[GARCHResult]:
        """
        Estimate single GARCH model and generate forecasts
        """
        try:
            # Convert returns to percentage
            returns_pct = returns * 100
            
            # Configure model
            if model_type == 'GARCH':
                model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist=distribution, rescale=False)
            elif model_type == 'EGARCH':
                model = arch_model(returns_pct, vol='EGARCH', p=1, q=1, dist=distribution, rescale=False)
            elif model_type == 'GJR-GARCH':
                model = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, dist=distribution, rescale=False)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            # Fit model
            result = model.fit(disp='off', show_warning=False)
            
            # Generate forecasts - use simulation for all models to ensure consistency
            forecasts = result.forecast(horizon=forecast_horizon, method='simulation', 
                                     simulations=n_simulations)
            
            # Get mean variance across simulations
            forecast_variance = forecasts.variance.mean(axis=1).values[-forecast_horizon:]
            
            # Convert to annualized volatility (scalar, not array)
            forecast_volatility = np.sqrt(forecast_variance)
            forecast_volatility_annualized = float(forecast_volatility.mean() * np.sqrt(252))
            historical_volatility_annualized = float(np.sqrt(result.conditional_volatility[-1]) * np.sqrt(252))
            
            return GARCHResult(
                model_type=model_type,
                distribution=distribution,
                params=dict(result.params),
                forecasts_annualized=forecast_volatility_annualized,  # Now a scalar
                volatility_annualized=historical_volatility_annualized  # Now a scalar
            )
                
        except Exception as e:
            self.logger.warning(f"Error estimating {model_type}-{distribution}: {str(e)}")
            return None
            
    def estimate(self, returns: np.ndarray, date: pd.Timestamp) -> List[GARCHResult]:
        """Estimate all GARCH models with checkpointing"""
        try:
            # Initialize counters for progress tracking
            total_models = len(self.model_specs)
            loaded_from_checkpoint = 0
            estimated_new = 0
            
            # Try to load from checkpoint first
            if self.checkpoint_manager:
                checkpoint = self.checkpoint_manager.load_checkpoint(date)
                if checkpoint and self._validate_checkpoint(checkpoint):
                    self.logger.debug(f"Loaded valid checkpoint for {date}")
                    loaded_from_checkpoint = len(checkpoint)
                    return checkpoint
            
            # Estimate all models
            results = []
            for i, (model_type, distribution) in enumerate(self.model_specs, 1):
                try:
                    result = self._estimate_single_model(
                        returns=returns,
                        model_type=model_type,
                        distribution=distribution
                    )
                    if result is not None:
                        results.append(result)
                        estimated_new += 1
                        
                    # Show progress every 2 models or when complete
                    if i % 2 == 0 or i == total_models:
                        progress_msg = (
                            f"Progress for {date:%Y-%m-%d}: "
                            f"{i}/{total_models} models processed "
                            f"({estimated_new} estimated, {loaded_from_checkpoint} from checkpoint)"
                        )
                        self.logger.info(progress_msg)
                        
                except Exception as e:
                    self.logger.warning(
                        f"Failed to estimate {model_type}-{distribution} for {date:%Y-%m-%d}: {str(e)}"
                    )
                    continue
            
            # Only save checkpoint if we got valid results
            if self.checkpoint_manager and results and self._validate_checkpoint(results):
                self.checkpoint_manager.save_checkpoint(date, results)
                self.logger.debug(f"Saved checkpoint for {date}")
            
            # Final progress update
            final_msg = (
                f"Completed {date:%Y-%m-%d}: "
                f"{len(results)}/{total_models} models successful "
                f"({estimated_new} estimated, {loaded_from_checkpoint} from checkpoint)"
            )
            self.logger.info(final_msg)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error estimating GARCH models for {date:%Y-%m-%d}: {str(e)}")
            raise

    def _validate_checkpoint(self, results: List[GARCHResult]) -> bool:
        """Validate checkpoint data"""
        try:
            if not results or len(results) == 0:
                return False
                
            for result in results:
                # Check if result has all required attributes
                if not all(hasattr(result, attr) for attr in 
                          ['model_type', 'distribution', 'params', 
                           'forecasts_annualized', 'volatility_annualized']):
                    return False
                    
                # Check if forecasts are valid
                if (not isinstance(result.forecasts_annualized, np.ndarray) or 
                    len(result.forecasts_annualized) == 0):
                    return False
                    
                # Check if volatilities are valid
                if (not isinstance(result.volatility_annualized, np.ndarray) or 
                    len(result.volatility_annualized) == 0):
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
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from arch import arch_model
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress known warnings from arch package
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')

@dataclass
class GARCHResult:
    """Container for GARCH estimation results"""
    params: Dict[str, float]
    forecasts_annualized: np.ndarray  # Store annualized volatility forecasts
    volatility_annualized: np.ndarray  # Store annualized historical volatilities
    model_type: str
    distribution: str
    
class GARCHEstimator:
    """Estimates and manages ensemble of GARCH models"""
    
    def __init__(self, 
                min_observations: int = 1260,
                n_simulations: int = 1000,
                random_seed: int = 42):
        """
        Initialize estimator
        
        Parameters:
        -----------
        min_observations : int
            Minimum required observations for estimation
        n_simulations : int
            Number of Monte Carlo simulations for forecasting
        random_seed : int
            Random seed for reproducibility
        """
        self.min_observations = min_observations
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self.model_specs = [
            ('GARCH', 'normal'),
            ('GARCH', 'studentst'),
            ('EGARCH', 'normal'),
            ('EGARCH', 'studentst'),
            ('GJR-GARCH', 'normal'),
            ('GJR-GARCH', 'studentst')
        ]
        
    def _annualize_variance(self, daily_variance: np.ndarray) -> np.ndarray:
        """Convert daily variance to annualized volatility"""
        return np.sqrt(252 * daily_variance) * 100  # Convert to percentage
        
    def _estimate_single_model(self, 
                             returns: np.ndarray,
                             model_type: str,
                             distribution: str,
                             forecast_horizon: int = 252) -> Optional[GARCHResult]:
        """
        Estimate single GARCH model and generate forecasts
        """
        try:
            # Handle NaN values
            if np.any(np.isnan(returns)):
                returns = returns[~np.isnan(returns)]
                
            if len(returns) < self.min_observations:
                logger.warning(f"Insufficient observations after removing NaN values: {len(returns)}")
                return None
                
            # Scale returns to percentage points for better numerical stability
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
                
            # Fit model with error handling
            try:
                result = model.fit(disp='off', show_warning=False)
            except Exception as e:
                logger.warning(f"Model fitting failed for {model_type}-{distribution}: {str(e)}")
                return None
                
            # Generate forecasts
            try:
                forecasts = result.forecast(
                    horizon=1,  # Start with 1-step forecast
                    method='analytic'  # Use analytic instead of simulation
                )
                
                # Get base volatility forecast
                base_variance = forecasts.variance.values[-1][0]
                if np.isnan(base_variance) or base_variance <= 0:
                    logger.warning(f"Invalid base variance for {model_type}-{distribution}")
                    return None
                    
                # Generate multi-step forecasts using persistence
                forecast_variance = np.zeros(forecast_horizon)
                persistence = result.params['alpha[1]'] + result.params['beta[1]']
                if model_type == 'GJR-GARCH':
                    persistence += result.params['gamma[1]'] / 2
                    
                for h in range(forecast_horizon):
                    forecast_variance[h] = base_variance * (persistence ** h)
                    
                # Convert variance to volatility and annualize
                forecast_volatility = np.sqrt(forecast_variance)
                forecast_volatility_annualized = forecast_volatility * np.sqrt(252) / 100
                historical_volatility_annualized = np.sqrt(result.conditional_volatility) * np.sqrt(252) / 100
                
                return GARCHResult(
                    params=dict(result.params),
                    forecasts_annualized=forecast_volatility_annualized,
                    volatility_annualized=historical_volatility_annualized,
                    model_type=model_type,
                    distribution=distribution
                )
                
            except Exception as e:
                logger.warning(f"Forecast generation failed for {model_type}-{distribution}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error estimating {model_type}-{distribution}: {str(e)}")
            return None
            
    def estimate_models(self, 
                       returns: np.ndarray,
                       forecast_horizon: int = 252,
                       parallel: bool = True) -> List[GARCHResult]:
        """Estimate all GARCH models in parallel"""
        
        if len(returns) < self.min_observations:
            raise ValueError(f"Insufficient observations: {len(returns)} < {self.min_observations}")
            
        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._estimate_single_model,
                        returns,
                        model_type,
                        distribution,
                        forecast_horizon
                    )
                    for model_type, distribution in self.model_specs
                ]
                results = [f.result() for f in futures]
        else:
            results = [
                self._estimate_single_model(
                    returns,
                    model_type,
                    distribution,
                    forecast_horizon
                )
                for model_type, distribution in self.model_specs
            ]
            
        # Filter out failed estimations
        return [r for r in results if r is not None]
        
    def calculate_ensemble_stats(self,
                               results: List[GARCHResult],
                               tau: int) -> Dict[str, float]:
        """
        Calculate ensemble statistics for horizon tau
        
        Args:
            results: List of GARCH estimation results
            tau: Forecast horizon to analyze
            
        Returns:
            Dictionary of ensemble statistics
        """
        if not results:
            raise ValueError("No GARCH results provided")
        
        # Extract forecasts for each model up to tau
        forecasts = np.array([r.forecasts_annualized[:tau] for r in results])
        
        if len(forecasts.shape) < 2:  # Handle single-point forecasts
            forecasts = forecasts.reshape(-1, 1)
        
        # Calculate statistics
        gev = np.mean(forecasts)  # Global Expected Volatility
        
        # Calculate time series statistics if we have multiple horizons
        if forecasts.shape[1] > 1:
            mean_forecasts = np.mean(forecasts, axis=0)
            evoev = np.std(mean_forecasts)  # Evolution of Expected Volatility
            dev = np.mean([np.std(forecasts[:,i]) for i in range(forecasts.shape[1])])  # Dispersion
            kev = np.mean([stats.skew(forecasts[:,i]) for i in range(forecasts.shape[1])])  # Skewness
            sevts = np.mean(forecasts[:,-1] - forecasts[:,0])  # Slope
        else:
            # Single point forecasts
            evoev = np.std(forecasts.flatten())
            dev = np.std(forecasts.flatten())
            kev = stats.skew(forecasts.flatten())
            sevts = 0.0
        
        return {
            'GEV': gev,
            'EVOEV': evoev,
            'DEV': dev,
            'KEV': kev,
            'SEVTS': sevts
        }

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
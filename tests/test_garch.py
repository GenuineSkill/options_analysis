import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pytest
import numpy as np
from garch.estimator import GARCHEstimator, GARCHResult
from scipy import stats
from datetime import datetime

@pytest.fixture
def sample_returns():
    """Generate sample returns with known properties"""
    np.random.seed(42)
    # Generate returns with volatility clustering
    n = 2000
    returns = np.random.normal(0, 1, n) * 0.01
    # Add some volatility clustering
    volatility = np.exp(np.random.normal(0, 0.2, n))
    returns = returns * volatility
    return returns

@pytest.fixture
def estimator():
    """Create GARCH estimator instance"""
    return GARCHEstimator(
        min_observations=1260,
        n_simulations=100,  # Reduced for testing
        random_seed=42
    )

def test_basic_estimation(estimator, sample_returns):
    """Test basic model estimation works"""
    results = estimator.estimate_models(sample_returns, forecast_horizon=252)
    
    # Check we got all 6 models
    assert len(results) == 6
    
    # Check all models estimated
    model_types = set(r.model_type for r in results)
    expected_types = {'GARCH', 'EGARCH', 'GJR-GARCH'}
    assert model_types == expected_types
    
    # Check distributions
    distributions = set(r.distribution for r in results)
    assert distributions == {'normal', 'studentst'}

def test_annualization(estimator, sample_returns):
    """Test variance annualization"""
    results = estimator.estimate_models(sample_returns, forecast_horizon=21)
    
    for result in results:
        # Check forecasts are annualized
        assert np.all(result.forecasts_annualized > 0)
        assert np.all(result.forecasts_annualized < 200)  # Reasonable range for annualized vol
        
        # Check historical volatilities are annualized
        assert np.all(result.volatility_annualized > 0)
        assert np.all(result.volatility_annualized < 200)

def test_ensemble_statistics(estimator, sample_returns):
    """Test ensemble statistics calculation"""
    results = estimator.estimate_models(sample_returns, forecast_horizon=252)
    stats = estimator.calculate_ensemble_stats(results, tau=21)
    
    # Check all statistics are present
    expected_stats = {'GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'}
    assert set(stats.keys()) == expected_stats
    
    # Check values are reasonable
    assert 0 < stats['GEV'] < 100  # Reasonable annualized vol
    assert stats['EVOEV'] > 0  # Positive uncertainty
    assert stats['DEV'] > 0  # Positive dispersion
    assert -5 < stats['KEV'] < 5  # Reasonable skewness range
    assert -50 < stats['SEVTS'] < 50  # Reasonable slope

def test_error_handling(estimator):
    """Test error handling"""
    # Test insufficient observations
    with pytest.raises(ValueError):
        short_returns = np.random.normal(0, 1, 100)
        estimator.estimate_models(short_returns)
    
    # Test missing data handling
    returns_with_nan = np.random.normal(0, 1, 2000)
    returns_with_nan[1000] = np.nan
    results = estimator.estimate_models(returns_with_nan)
    assert len(results) > 0  # Should still estimate some models

def test_parallel_vs_serial(estimator, sample_returns):
    """Test parallel and serial estimation give same results"""
    results_parallel = estimator.estimate_models(sample_returns, parallel=True)
    results_serial = estimator.estimate_models(sample_returns, parallel=False)
    
    # Check same number of models
    assert len(results_parallel) == len(results_serial)
    
    # Check forecasts are similar (not exact due to simulation)
    for rp, rs in zip(results_parallel, results_serial):
        np.testing.assert_allclose(
            rp.forecasts_annualized,
            rs.forecasts_annualized,
            rtol=1e-1  # 10% tolerance due to simulation
        )

def test_simulation_stability(estimator, sample_returns):
    """Test stability of simulation-based forecasts"""
    # Run multiple estimations
    n_runs = 5
    all_forecasts = []
    
    for _ in range(n_runs):
        results = estimator.estimate_models(sample_returns, forecast_horizon=21)
        forecasts = np.mean([r.forecasts_annualized for r in results], axis=0)
        all_forecasts.append(forecasts)
    
    # Convert to array
    all_forecasts = np.array(all_forecasts)
    
    # Check stability across runs
    variation = np.std(all_forecasts, axis=0) / np.mean(all_forecasts, axis=0)
    assert np.all(variation < 0.1)  # Coefficient of variation < 10%

def test_garch_estimation():
    # Generate test data
    n_obs = 3024  # Update from 1260 to match new requirement
    returns = np.random.normal(0, 0.01, n_obs)
    
    estimator = GARCHEstimator(min_observations=3000)
    # ... rest of test

def test_gjr_garch_estimation():
    """Test GJR-GARCH model estimation and forecasting"""
    returns = np.random.normal(0, 0.01, 1000)  # Simulated returns
    
    estimator = GARCHEstimator()
    model_spec = ('gjrgarch', 'normal', {'p': 1, 'o': 1, 'q': 1})
    
    result = estimator._estimate_single_model(returns, model_spec, datetime.now(), 'SPX')
    
    assert result is not None
    assert result.model_type == 'gjrgarch'
    assert 10 <= np.mean(result.forecast_path) <= 40  # Reasonable vol range
    assert len(np.unique(result.forecast_path)) > 1  # Forecasts vary over time

if __name__ == '__main__':
    pytest.main([__file__])
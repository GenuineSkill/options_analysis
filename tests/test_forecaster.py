import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from garch.forecaster import GARCHForecaster, ForecastWindow
from garch.estimator import GARCHEstimator

@pytest.fixture
def sample_data():
    """Generate sample data with known properties"""
    np.random.seed(42)
    n_days = 2000
    dates = pd.date_range('2015-01-01', periods=n_days)
    
    # Generate returns with volatility clustering
    returns = np.random.normal(0, 1, n_days) * 0.01
    volatility = np.exp(np.random.normal(0, 0.2, n_days))
    returns = returns * volatility
    
    return returns, dates.to_numpy()

@pytest.fixture
def forecaster():
    """Create forecaster instance with test configuration"""
    estimator = GARCHEstimator(
        min_observations=1260,
        n_simulations=100,  # Reduced for testing
        random_seed=42
    )
    return GARCHForecaster(
        estimator=estimator,
        min_window=1260,
        step_size=21
    )

def test_initialization(forecaster):
    """Test forecaster initialization"""
    assert forecaster.min_window == 1260
    assert forecaster.step_size == 21
    assert isinstance(forecaster.estimator, GARCHEstimator)
    assert len(forecaster.forecast_windows) == 0

def test_expanding_windows_generation(forecaster, sample_data):
    """Test basic expanding window generation"""
    returns, dates = sample_data
    windows = forecaster.generate_expanding_windows(
        returns=returns,
        dates=dates,
        forecast_horizon=252
    )
    
    # Calculate expected number of windows based on business days
    total_bdays = len(pd.bdate_range(dates[0], dates[-1]))
    expected_windows = max(1, (total_bdays - forecaster.min_window) // forecaster.step_size)
    
    assert len(windows) > 0, "No windows generated"
    assert abs(len(windows) - expected_windows) <= 2, \
        f"Expected ~{expected_windows} windows, got {len(windows)}"
    
    # Check window properties
    for window in windows:
        # Check window size (in business days)
        window_bdays = len(pd.bdate_range(window.start_date, window.end_date))
        assert abs(window_bdays - forecaster.min_window) <= 5, \
            f"Window size mismatch: {window_bdays} vs {forecaster.min_window}"
        
        # Check dates are in order
        assert window.start_date < window.end_date
        
        # Check GARCH results exist
        assert len(window.garch_results) > 0
        
        # Check ensemble stats exist
        assert all(key in window.ensemble_stats for key in ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'])

def test_window_dates(forecaster, sample_data):
    """Test window date handling"""
    returns, dates = sample_data
    windows = forecaster.generate_expanding_windows(returns, dates, 252)
    
    for i, window in enumerate(windows):
        if i > 0:
            # Check windows advance by step_size (in business days)
            prev_window = windows[i-1]
            bdays = len(pd.bdate_range(prev_window.start_date, window.start_date)) - 1
            
            # Allow for some flexibility due to weekends/holidays
            assert abs(bdays - forecaster.step_size) <= 2, \
                f"Window step size mismatch: got {bdays} business days, expected {forecaster.step_size}"
        
        # Check window length (in business days)
        window_bdays = len(pd.bdate_range(window.start_date, window.end_date))
        
        # Allow for some flexibility due to weekends/holidays
        assert abs(window_bdays - forecaster.min_window) <= 5, \
            f"Window length mismatch: got {window_bdays} business days, expected {forecaster.min_window}"

def test_forecast_series_extraction(forecaster, sample_data):
    """Test getting forecast time series"""
    returns, dates = sample_data
    windows = forecaster.generate_expanding_windows(returns, dates, 252)
    
    # Test each statistic
    for stat in ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS']:
        dates, values = forecaster.get_forecast_series(stat)
        
        assert len(dates) == len(windows)
        assert len(values) == len(windows)
        assert not np.any(np.isnan(values))
        assert all(isinstance(v, (int, float)) for v in values)

def test_dataframe_conversion(forecaster, sample_data):
    """Test conversion to DataFrame"""
    returns, dates = sample_data
    forecaster.generate_expanding_windows(returns, dates, 252)
    
    df = forecaster.to_dataframe()
    
    # Check DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(forecaster.forecast_windows)
    assert all(col in df.columns for col in ['start_date', 'end_date', 'GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'])
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df['start_date'])
    assert pd.api.types.is_datetime64_any_dtype(df['end_date'])
    assert all(pd.api.types.is_float_dtype(df[col]) for col in ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'])

def test_error_handling(forecaster):
    """Test error handling"""
    # Test insufficient data
    with pytest.raises(ValueError):
        short_returns = np.random.normal(0, 1, 100)
        short_dates = pd.date_range('2015-01-01', periods=100)
        forecaster.generate_expanding_windows(short_returns, short_dates.to_numpy())
    
    # Test missing data handling
    returns = np.random.normal(0, 1, 2000)
    returns[1000:1100] = np.nan  # Add block of NaN values
    dates = pd.date_range('2015-01-01', periods=2000)
    windows = forecaster.generate_expanding_windows(returns, dates.to_numpy())
    assert len(windows) > 0  # Should still generate some windows

def test_forecast_persistence(forecaster, sample_data):
    """Test consistency of forecasts"""
    returns, dates = sample_data
    windows = forecaster.generate_expanding_windows(returns, dates, 252)
    
    # Get GEV series
    _, values = forecaster.get_forecast_series('GEV')
    
    # Check for reasonable persistence
    changes = np.diff(values)
    assert np.median(np.abs(changes)) < 5.0  # Changes should be moderate
    assert np.max(np.abs(changes)) < 20.0  # No extreme jumps

def test_step_size_impact(sample_data):
    """Test impact of different step sizes"""
    returns, dates = sample_data
    
    # Create forecasters with different step sizes
    step_sizes = [21, 42, 63]  # 1, 2, 3 months
    results = []
    
    for step in step_sizes:
        forecaster = GARCHForecaster(step_size=step)
        windows = forecaster.generate_expanding_windows(returns, dates, 252)
        results.append(len(windows))
    
    # Check number of windows scales inversely with step size
    assert results[0] > results[1] > results[2]
    
    # Check approximate ratios
    assert np.isclose(results[0] / results[1], 2, rtol=0.2)
    assert np.isclose(results[0] / results[2], 3, rtol=0.2)

if __name__ == '__main__':
    pytest.main([__file__])
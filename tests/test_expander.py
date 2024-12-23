import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import tempfile
from datetime import datetime, timedelta
import shutil

# Import handled by conftest.py
from regression.expander import ExpandingWindowAnalyzer

@pytest.fixture
def temp_db_path():
    """Create temporary directory for database"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_days = 2500  # About 10 years of daily data
    dates = pd.date_range('2010-01-01', periods=n_days)
    
    # Generate returns with volatility clustering
    base_returns = np.random.normal(0, 1, n_days) * 0.01
    # Add GARCH-like effects
    volatility = np.exp(np.random.normal(0, 0.2, n_days))
    returns = base_returns * volatility
    
    # Generate implied volatilities with term structure
    tenors = ['1M', '2M', '3M', '6M', '12M']
    base_vol = 15 + np.random.normal(0, 2, n_days)
    implied_vols = pd.DataFrame(
        {tenor: base_vol + np.random.normal(0, 1, n_days) + i  # Add slight upward slope
         for i, tenor in enumerate(tenors)},
        index=dates
    )
    
    return {
        'SPX': {
            'returns': returns,
            'dates': dates.to_numpy(),
            'implied_vols': implied_vols
        }
    }

@pytest.fixture
def estimator():
    """Create properly configured estimator"""
    from garch.estimator import GARCHEstimator
    return GARCHEstimator(
        min_observations=1260,  # 5 years of data
        n_simulations=1000,
        random_seed=42
    )

@pytest.fixture
def forecaster(estimator):
    """Create forecaster with proper window size"""
    from garch.forecaster import GARCHForecaster
    return GARCHForecaster(
        estimator=estimator,
        min_window=1260,  # 5 years of data
        step_size=21      # Monthly steps
    )

@pytest.fixture
def analyzer(temp_db_path, forecaster):
    """Create analyzer with proper configuration"""
    analyzer = ExpandingWindowAnalyzer(
        db_path=temp_db_path,
        min_window=1260,  # 5 years of data
        step_size=21,     # Monthly steps
        forecaster=forecaster
    )
    yield analyzer
    analyzer.close()

def test_initialization(analyzer):
    """Test analyzer initialization"""
    assert analyzer.forecaster.min_window == 1260
    assert analyzer.forecaster.step_size == 21
    assert hasattr(analyzer, 'db')

def test_single_index_processing(analyzer, sample_data):
    """Test processing single index"""
    data = sample_data['SPX']
    results = analyzer.process_index(
        index_id='SPX',
        returns=data['returns'],
        dates=data['dates'],
        implied_vols=data['implied_vols']
    )
    
    # Check results structure
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    
    # Check required columns exist
    required_columns = [
        'date', 'tenor', 'iv', 'gev', 'error',
        'std_error', 'evoev', 'dev', 'kev', 'sevts'
    ]
    assert all(col in results.columns for col in required_columns)
    
    # Check data quality
    assert not results['gev'].isna().any()
    assert not results['error'].isna().any()
    assert not results['std_error'].isna().any()

def test_multiple_indices(analyzer, sample_data):
    """Test processing multiple indices"""
    # Add another index
    data = sample_data.copy()
    data['SX5E'] = data['SPX'].copy()  # Use same data for testing
    
    results = analyzer.process_multiple_indices(data)
    
    # Check results for both indices
    assert set(results.keys()) == {'SPX', 'SX5E'}
    assert all(isinstance(df, pd.DataFrame) for df in results.values())
    assert all(len(df) > 0 for df in results.values())

def test_incremental_processing(analyzer, sample_data):
    """Test incremental processing with existing data"""
    data = sample_data['SPX']
    
    # First run
    first_results = analyzer.process_index(
        index_id='SPX',
        returns=data['returns'][:1000],
        dates=data['dates'][:1000],
        implied_vols=data['implied_vols'].iloc[:1000]
    )
    
    # Second run with more data
    second_results = analyzer.process_index(
        index_id='SPX',
        returns=data['returns'],
        dates=data['dates'],
        implied_vols=data['implied_vols']
    )
    
    # Check incremental processing
    assert len(second_results) > len(first_results)
    
    # Check data consistency
    common_dates = set(first_results['date']).intersection(second_results['date'])
    assert len(common_dates) > 0
    
    for date in common_dates:
        first_data = first_results[first_results['date'] == date]
        second_data = second_results[second_results['date'] == date]
        np.testing.assert_array_almost_equal(
            first_data['gev'].values,
            second_data['gev'].values
        )

def test_error_correction_calculation(analyzer, sample_data):
    """Test error correction term calculations"""
    data = sample_data['SPX']
    results = analyzer.process_index(
        index_id='SPX',
        returns=data['returns'],
        dates=data['dates'],
        implied_vols=data['implied_vols']
    )
    
    # Check error calculations
    assert np.allclose(
        results['error'],
        results['iv'] - results['gev']
    )
    
    # Check standardized errors
    assert np.allclose(
        results['std_error'],
        results['error'] / results['evoev']
    )
    
    # Check reasonable ranges
    assert results['std_error'].abs().mean() < 5.0  # Most errors within 5 std
    assert results['error'].abs().mean() < 20.0  # Most errors within 20 vol points

def test_error_handling(analyzer):
    """Test error handling"""
    # Test with invalid data
    with pytest.raises(Exception):
        analyzer.process_index(
            index_id='INVALID',
            returns=np.array([]),
            dates=np.array([]),
            implied_vols=pd.DataFrame()
        )
    
    # Test with mismatched data
    with pytest.raises(Exception):
        analyzer.process_index(
            index_id='SPX',
            returns=np.random.normal(0, 1, 100),
            dates=pd.date_range('2015-01-01', periods=200),
            implied_vols=pd.DataFrame(np.random.normal(15, 2, (300, 5)))
        )

def test_database_persistence(analyzer, sample_data):
    """Test database storage and retrieval"""
    data = sample_data['SPX']
    
    # Process data
    analyzer.process_index(
        index_id='SPX',
        returns=data['returns'],
        dates=data['dates'],
        implied_vols=data['implied_vols']
    )
    
    # Check database has data
    latest = analyzer.db.get_latest_window('SPX')
    assert latest is not None
    
    # Get stored ensemble stats
    stats = analyzer.db.get_ensemble_stats_series(
        index_id='SPX',
        start_date=data['dates'][0]
    )
    assert len(stats) > 0

def test_window_advancement(analyzer, sample_data):
    """Test proper window advancement"""
    data = sample_data['SPX']
    
    try:
        print("\nProcessing index...")
        results = analyzer.process_index(
            index_id='SPX',
            returns=data['returns'],
            dates=data['dates'],
            implied_vols=data['implied_vols']
        )
        
        # First verify we got results
        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        assert not results.empty, "Results should not be empty"
        
        # Print diagnostic information
        print(f"\nResults shape: {results.shape}")
        print(f"Results columns: {results.columns.tolist()}")
        
        if len(results) > 0:
            print("\nFirst row:")
            print(results.iloc[0])
            print("\nLast row:")
            print(results.iloc[-1])
            
    except Exception as e:
        print(f"\nError type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

def test_invalid_parameters(temp_db_path):
    """Test initialization with invalid parameters"""
    # Test invalid min_window
    with pytest.raises(ValueError):
        ExpandingWindowAnalyzer(temp_db_path, min_window=0)
    
    # Test invalid step_size
    with pytest.raises(ValueError):
        ExpandingWindowAnalyzer(temp_db_path, step_size=0)

def test_data_validation(analyzer):
    """Test data validation in process_index"""
    dates = pd.date_range('2015-01-01', periods=100)
    
    # Test non-matching lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        analyzer.process_index(
            index_id='TEST',
            returns=np.random.normal(0, 1, 90),  # Wrong length
            dates=dates,
            implied_vols=pd.DataFrame(np.random.normal(15, 2, (100, 5)))
        )

def test_cleanup(temp_db_path):
    """Test proper resource cleanup"""
    analyzer = ExpandingWindowAnalyzer(temp_db_path)
    
    # Test context manager usage
    with analyzer:
        assert not analyzer.db.is_closed()
    
    assert analyzer.db.is_closed()

if __name__ == '__main__':
    pytest.main([__file__])
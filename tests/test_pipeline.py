import pytest
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from calculate_historical import run_analysis, initialize_components

@pytest.fixture
def sample_data():
    """Create sample data for testing with sufficient history for GARCH convergence"""
    # Create 6 years of daily data (approximately 1500 trading days)
    dates = pd.date_range('2015-01-01', '2020-12-31', freq='B')  # 'B' for business days
    
    # Generate realistic-looking returns data
    np.random.seed(42)  # For reproducibility
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.01, len(dates)),  # Slight positive drift
        index=dates,
        columns=['returns']
    )
    
    # Generate realistic-looking implied volatility data with proper rolling mean
    iv = pd.DataFrame(
        np.random.uniform(0.15, 0.25, (len(dates), 5)),  # Typical IV range
        index=dates,
        columns=['SPX_1M', 'SPX_2M', 'SPX_3M', 'SPX_6M', 'SPX_12M']
    )
    
    # Add some autocorrelation using proper forward fill
    for col in iv.columns:
        iv[col] = iv[col].rolling(window=20).mean().bfill()
    
    return returns, iv

def test_pipeline(sample_data, tmp_path):
    """Test full pipeline execution"""
    returns_df, iv_df = sample_data
    
    # Log data characteristics
    logger = logging.getLogger('test')
    logger.info(f"Test data range: {returns_df.index[0]} to {returns_df.index[-1]}")
    logger.info(f"Number of observations: {len(returns_df)}")
    
    components = initialize_components()
    
    # Run analysis
    results = run_analysis(
        components=components,
        returns_df=returns_df,
        iv_df=iv_df,
        output_dir=tmp_path,
        logger=logger,
        monitor=None
    )
    
    # Check results more thoroughly
    assert results is not None
    assert 'index_id' in results
    assert results['index_id'] == 'SPX'
    assert len(results['dates']) >= 1260, "Insufficient data for GARCH convergence"
    
    # Check that results contain the expected windows
    assert 'results' in results
    assert len(results['results']) > 0
    
    # Check output files
    results_file = tmp_path / "SPX_results.pkl"
    assert results_file.exists(), "Results file not saved"
    assert results_file.stat().st_size > 0, "Results file is empty"
    
    # Log success with details
    logger.info(f"Test passed with {len(results['results'])} windows processed") 

@pytest.mark.parametrize("data_length", [
    1260,  # Minimum required length
    1500,  # Normal case
    2000   # Extended dataset
])
def test_pipeline_with_different_lengths(data_length, tmp_path):
    """Test pipeline with different data lengths"""
    dates = pd.date_range('2015-01-01', freq='B', periods=data_length)
    returns = pd.DataFrame(
        np.random.normal(0.0001, 0.01, len(dates)),
        index=dates,
        columns=['returns']
    )
    iv = pd.DataFrame(
        np.random.uniform(0.15, 0.25, (len(dates), 5)),
        index=dates,
        columns=['SPX_1M', 'SPX_2M', 'SPX_3M', 'SPX_6M', 'SPX_12M']
    )
    
    components = initialize_components()
    results = run_analysis(
        components=components,
        returns_df=returns,
        iv_df=iv,
        output_dir=tmp_path,
        logger=logging.getLogger('test'),
        monitor=None
    )
    
    assert len(results['dates']) == data_length 

def test_pipeline_insufficient_data():
    """Test pipeline fails gracefully with insufficient data"""
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, len(dates)),
        index=dates,
        columns=['returns']
    )
    iv = pd.DataFrame(
        np.random.uniform(0.15, 0.25, (len(dates), 5)),
        index=dates,
        columns=['SPX_1M', 'SPX_2M', 'SPX_3M', 'SPX_6M', 'SPX_12M']
    )
    
    with pytest.raises(ValueError, match="Insufficient data"):
        components = initialize_components()
        run_analysis(
            components=components,
            returns_df=returns,
            iv_df=iv,
            output_dir=Path("temp"),
            logger=logging.getLogger('test'),
            monitor=None
        ) 
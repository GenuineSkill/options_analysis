import sys
import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import shutil

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.visualization import GARCHVisualizer

plt.ion()  # Turn on interactive mode

@pytest.fixture
def temp_dir():
    """Create temporary directory for saving plots"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    try:
        shutil.rmtree(temp_path)
    except:
        pass

@pytest.fixture
def visualizer():
    """Create visualizer instance"""
    viz = GARCHVisualizer(style='seaborn')
    yield viz
    viz.close_all()

@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=100)
    forecasts = np.random.normal(15, 2, (3, 100))  # 3 models, 100 days
    model_types = ['GARCH', 'EGARCH', 'GJR-GARCH']
    return dates, forecasts, model_types

@pytest.fixture
def sample_ensemble_data():
    """Create sample ensemble statistics data"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=100)
    
    # Create DataFrame with realistic ensemble statistics
    data = {
        'GEV': 15 + np.random.normal(0, 1, 100),
        'EVOEV': 2 + np.random.normal(0, 0.2, 100),
        'DEV': 1.5 + np.random.normal(0, 0.1, 100),
        'KEV': np.random.normal(0, 0.5, 100),
        'SEVTS': np.random.normal(0, 0.3, 100)
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_error_data():
    """Create sample error correction data"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=100)
    
    # Create DataFrame with correlated errors
    base_error = np.random.normal(0, 1, 100)
    data = {
        'error': base_error * 2,  # Raw errors
        'std_error': base_error,  # Standardized errors
        'tenor': '1M'
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_term_structure_data():
    """Create sample term structure data"""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=100)
    tenors = ['1M', '2M', '3M', '6M', '12M']
    
    # Create upward sloping term structure with noise
    base_curve = np.array([15, 16, 17, 18, 19])
    data = pd.DataFrame(
        {tenor: base_curve[i] + np.random.normal(0, 1, 100)
         for i, tenor in enumerate(tenors)},
        index=dates
    )
    return data

def test_initialization(visualizer):
    """Test visualizer initialization"""
    assert hasattr(visualizer, 'colors')
    assert len(plt.style.available) > 0

def test_garch_forecasts_plot(visualizer, sample_forecast_data):
    """Test GARCH forecast plotting"""
    dates, forecasts, model_types = sample_forecast_data
    
    # Create plot
    fig = visualizer.plot_garch_forecasts(
        dates=dates,
        forecasts=forecasts,
        model_types=model_types,
        title="Test Forecasts"
    )
    
    plt.show()
    plt.pause(2)  # Pause to show plot for 2 seconds

def test_ensemble_stats_plot(visualizer, sample_ensemble_data, temp_dir):
    """Test ensemble statistics plotting"""
    save_path = temp_dir / "ensemble.png"
    
    # Create plot
    fig = visualizer.plot_ensemble_stats(
        stats_df=sample_ensemble_data,
        title="Test Ensemble Stats",
        save_path=save_path
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5  # One subplot per statistic
    assert save_path.exists()

def test_error_correction_plot(visualizer, sample_error_data, temp_dir):
    """Test error correction plotting"""
    save_path = temp_dir / "errors.png"
    
    # Create plot
    fig = visualizer.plot_error_correction(
        error_df=sample_error_data,
        tenor="1M",
        title="Test Error Correction",
        save_path=save_path
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # Raw and standardized errors
    assert save_path.exists()

def test_term_structure_plot(visualizer, sample_term_structure_data, temp_dir):
    """Test term structure plotting"""
    save_path = temp_dir / "term_structure.png"
    dates_to_plot = [sample_term_structure_data.index[0], 
                     sample_term_structure_data.index[-1]]
    
    # Create plot
    fig = visualizer.plot_term_structure(
        term_df=sample_term_structure_data,
        dates=dates_to_plot,
        title="Test Term Structure",
        save_path=save_path
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].lines) == len(dates_to_plot)
    assert save_path.exists()

def test_diagnostic_grid(visualizer, sample_error_data, temp_dir):
    """Test diagnostic grid plotting"""
    save_path = temp_dir / "diagnostic.png"
    
    # Add required columns for diagnostic grid
    df = sample_error_data.copy()
    df['iv'] = 15 + np.random.normal(0, 1, len(df))
    df['gev'] = 15 + np.random.normal(0, 1, len(df))
    df['evoev'] = 2 + np.random.normal(0, 0.2, len(df))
    df['dev'] = 1.5 + np.random.normal(0, 0.1, len(df))
    df['kev'] = np.random.normal(0, 0.5, len(df))
    df['sevts'] = np.random.normal(0, 0.3, len(df))
    
    # Create plot
    fig = visualizer.plot_diagnostic_grid(
        df=df,
        tenor="1M",
        title="Test Diagnostics",
        save_path=save_path
    )
    
    # Check figure properties
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5  # Number of diagnostic plots
    assert save_path.exists()

def test_resource_cleanup(visualizer, sample_forecast_data):
    """Test proper figure cleanup"""
    dates, forecasts, model_types = sample_forecast_data
    
    # Create several plots
    for _ in range(5):
        visualizer.plot_garch_forecasts(
            dates=dates,
            forecasts=forecasts,
            model_types=model_types
        )
    
    # Check figures exist
    assert len(plt.get_fignums()) > 0
    
    # Clean up
    visualizer.close_all()
    
    # Check all figures closed
    assert len(plt.get_fignums()) == 0

def test_context_manager(sample_forecast_data):
    """Test context manager functionality"""
    dates, forecasts, model_types = sample_forecast_data
    with GARCHVisualizer() as viz:
        viz.plot_garch_forecasts(
            dates=dates,
            forecasts=forecasts,
            model_types=model_types
        )
        assert len(plt.get_fignums()) > 0
    
    # Check figures cleaned up after context
    assert len(plt.get_fignums()) == 0

def test_invalid_inputs(visualizer):
    """Test handling of invalid inputs"""
    # Test empty data
    with pytest.raises(ValueError):
        visualizer.plot_garch_forecasts(
            dates=np.array([]),
            forecasts=np.array([]),
            model_types=[]
        )
    
    # Test mismatched dimensions
    with pytest.raises(Exception):
        visualizer.plot_garch_forecasts(
            dates=np.array([1, 2, 3]),
            forecasts=np.random.normal(0, 1, (2, 4)),  # Wrong shape
            model_types=['GARCH', 'EGARCH']
        )

def test_save_all_plots(visualizer, sample_forecast_data, sample_ensemble_data, 
                        sample_error_data, sample_term_structure_data):
    """Save all visualization types to files"""
    # Create plots directory in project root
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving plots to: {output_dir.absolute()}")
    
    # 1. GARCH forecasts
    dates, forecasts, model_types = sample_forecast_data
    fig = visualizer.plot_garch_forecasts(
        dates=dates,
        forecasts=forecasts,
        model_types=model_types,
        title="GARCH Forecasts",
        save_path=output_dir / "garch_forecasts.png"
    )
    
    # 2. Ensemble statistics
    fig = visualizer.plot_ensemble_stats(
        stats_df=sample_ensemble_data,
        title="Ensemble Statistics",
        save_path=output_dir / "ensemble_stats.png"
    )
    
    # 3. Error correction
    fig = visualizer.plot_error_correction(
        error_df=sample_error_data,
        tenor="1M",
        title="Error Correction",
        save_path=output_dir / "error_correction.png"
    )
    
    # 4. Term structure
    dates_to_plot = [sample_term_structure_data.index[0], 
                     sample_term_structure_data.index[-1]]
    fig = visualizer.plot_term_structure(
        term_df=sample_term_structure_data,
        dates=dates_to_plot,
        title="Term Structure",
        save_path=output_dir / "term_structure.png"
    )
    
    # 5. Diagnostic grid
    df = sample_error_data.copy()
    df['iv'] = 15 + np.random.normal(0, 1, len(df))
    df['gev'] = 15 + np.random.normal(0, 1, len(df))
    df['evoev'] = 2 + np.random.normal(0, 0.2, len(df))
    df['dev'] = 1.5 + np.random.normal(0, 0.1, len(df))
    df['kev'] = np.random.normal(0, 0.5, len(df))
    df['sevts'] = np.random.normal(0, 0.3, len(df))
    
    fig = visualizer.plot_diagnostic_grid(
        df=df,
        tenor="1M",
        title="Diagnostics",
        save_path=output_dir / "diagnostic_grid.png"
    )
    
    # Verify files were created
    expected_files = [
        "garch_forecasts.png",
        "ensemble_stats.png",
        "error_correction.png",
        "term_structure.png",
        "diagnostic_grid.png"
    ]
    
    for filename in expected_files:
        path = output_dir / filename
        assert path.exists(), f"Plot file not found: {filename}"
        print(f"Created: {path}")

if __name__ == '__main__':
    pytest.main([__file__])
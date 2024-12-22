import sys
import os
import pytest
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_manager.database import GARCHDatabase

# Create mock classes for testing
@dataclass
class MockGARCHResult:
    model_type: str
    distribution: str
    params: dict
    forecasts_annualized: np.ndarray
    volatility_annualized: np.ndarray

@dataclass
class MockForecastWindow:
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: List[MockGARCHResult]
    ensemble_stats: Dict

@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    
    db = GARCHDatabase(db_path)
    yield db
    
    try:
        db.close()
    except:
        pass
        
    try:
        if db_path.exists():
            db_path.unlink()
        Path(temp_dir).rmdir()
    except:
        pass

@pytest.fixture
def sample_window():
    """Create sample forecast window for testing"""
    return MockForecastWindow(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31),
        returns=np.array([0.01, -0.02, 0.03, -0.01, 0.02]),
        garch_results=[
            MockGARCHResult(
                model_type='GARCH',
                distribution='normal',
                params={'omega': 0.1, 'alpha': 0.1, 'beta': 0.8},
                forecasts_annualized=np.array([0.15, 0.16, 0.17]),
                volatility_annualized=np.array([0.14, 0.15, 0.16])
            ),
            MockGARCHResult(
                model_type='EGARCH',
                distribution='studentst',
                params={'omega': 0.1, 'alpha': 0.1, 'beta': 0.8, 'gamma': 0.1},
                forecasts_annualized=np.array([0.16, 0.17, 0.18]),
                volatility_annualized=np.array([0.15, 0.16, 0.17])
            )
        ],
        ensemble_stats={
            'GEV': 15.5,
            'EVOEV': 2.0,
            'DEV': 1.5,
            'KEV': 0.1,
            'SEVTS': -0.5
        }
    )

def test_database_initialization(temp_db):
    """Test database creation and table initialization"""
    # Check tables exist
    tables = temp_db.conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table'
    """).fetchall()
    table_names = [t[0] for t in tables]
    
    assert 'garch_windows' in table_names
    assert 'garch_results' in table_names
    assert 'ensemble_stats' in table_names

def test_store_forecast_window(temp_db, sample_window):
    """Test storing forecast window data"""
    # Store window
    window_id = temp_db.store_forecast_window(sample_window, 'SPX')
    
    # Check window was stored
    window = temp_db.conn.execute("""
        SELECT window_id, index_id, start_date::DATE, end_date::DATE, returns
        FROM garch_windows 
        WHERE window_id = ?
    """, [window_id]).fetchone()
    
    assert window is not None
    assert window[1] == 'SPX'  # index_id is second column
    assert window[2] == sample_window.start_date.date()
    assert window[3] == sample_window.end_date.date()
    
    # Check returns array was stored correctly
    stored_returns = np.frombuffer(window[4])  # returns is fifth column
    np.testing.assert_array_equal(stored_returns, sample_window.returns)
    
    # Check GARCH results were stored
    results = temp_db.conn.execute("""
        SELECT * FROM garch_results WHERE window_id = ?
    """, [window_id]).fetchall()
    
    assert len(results) == len(sample_window.garch_results)
    
    # Check ensemble stats were stored
    stats = temp_db.conn.execute("""
        SELECT gev, evoev, dev, kev, sevts 
        FROM ensemble_stats 
        WHERE window_id = ?
    """, [window_id]).fetchone()
    
    assert stats is not None
    assert abs(stats[0] - sample_window.ensemble_stats['GEV']) < 1e-10
    assert abs(stats[1] - sample_window.ensemble_stats['EVOEV']) < 1e-10

def test_get_latest_window(temp_db, sample_window):
    """Test retrieving latest window"""
    # Store multiple windows
    window_id1 = temp_db.store_forecast_window(sample_window, 'SPX')
    
    # Create newer window
    newer_window = MockForecastWindow(
        start_date=sample_window.start_date + timedelta(days=21),
        end_date=sample_window.end_date + timedelta(days=21),
        returns=sample_window.returns,
        garch_results=sample_window.garch_results,
        ensemble_stats=sample_window.ensemble_stats
    )
    window_id2 = temp_db.store_forecast_window(newer_window, 'SPX')
    
    # Get latest window
    latest = temp_db.get_latest_window('SPX')
    
    assert latest is not None
    assert latest['window_id'] == window_id2
    assert latest['end_date'] == newer_window.end_date.date()

def test_get_ensemble_stats_series(temp_db, sample_window):
    """Test retrieving ensemble statistics time series"""
    # Store multiple windows
    dates = []
    for i in range(5):
        window = MockForecastWindow(
            start_date=sample_window.start_date + timedelta(days=21*i),
            end_date=sample_window.end_date + timedelta(days=21*i),
            returns=sample_window.returns,
            garch_results=sample_window.garch_results,
            ensemble_stats=sample_window.ensemble_stats
        )
        temp_db.store_forecast_window(window, 'SPX')
        dates.append(window.end_date)
    
    # Get stats series
    stats_df = temp_db.get_ensemble_stats_series(
        index_id='SPX',
        start_date=dates[1],
        end_date=dates[-2]
    )
    
    assert len(stats_df) == 3  # Middle 3 dates
    assert all(col in stats_df.columns for col in ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS'])
    assert all(stats_df['GEV'] == sample_window.ensemble_stats['GEV'])

def test_get_model_forecasts(temp_db, sample_window):
    """Test retrieving model forecasts"""
    window_id = temp_db.store_forecast_window(sample_window, 'SPX')
    
    # Get all forecasts
    forecasts = temp_db.get_model_forecasts(window_id)
    assert len(forecasts) == len(sample_window.garch_results)
    
    # Get specific model forecasts
    garch_forecasts = temp_db.get_model_forecasts(
        window_id,
        model_type='GARCH',
        distribution='normal'
    )
    assert len(garch_forecasts) == 1
    np.testing.assert_array_equal(
        garch_forecasts[0],
        sample_window.garch_results[0].forecasts_annualized
    )

def test_error_handling(temp_db):
    """Test database error handling"""
    # Test invalid window ID
    assert temp_db.get_latest_window('INVALID') is None
    
    # Test invalid date range
    empty_df = temp_db.get_ensemble_stats_series(
        'SPX',
        start_date=datetime(2030, 1, 1)
    )
    assert len(empty_df) == 0
    
    # Test database constraints
    with pytest.raises(Exception):
        # Try to insert ensemble stats without window
        temp_db.conn.execute("""
            INSERT INTO ensemble_stats (
                window_id, gev, evoev, dev, kev, sevts
            ) VALUES (999, 0, 0, 0, 0, 0)
        """)

def test_transaction_rollback(temp_db, sample_window):
    """Test transaction rollback on error"""
    # Create a new connection for transaction testing
    with duckdb.connect(str(temp_db.db_path)) as conn:
        try:
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Store window
            conn.execute("""
                INSERT INTO garch_windows (start_date, end_date, index_id, returns)
                VALUES (?, ?, ?, ?)
                RETURNING window_id
            """, (
                sample_window.start_date,
                sample_window.end_date,
                'SPX',
                sample_window.returns.tobytes()
            ))
            
            # Try to insert invalid data - should rollback
            conn.execute("""
                INSERT INTO garch_results (window_id, model_type)
                VALUES (?, ?)
            """, [1, 'invalid'])
            
            # Should not reach here
            conn.execute("COMMIT")
        except:
            conn.execute("ROLLBACK")
    
    # Check window was not stored
    result = temp_db.conn.execute("""
        SELECT COUNT(*) FROM garch_windows WHERE index_id = ?
    """, ['SPX']).fetchone()
    assert result[0] == 0

def test_data_validation(temp_db):
    """Test input data validation"""
    invalid_window = MockForecastWindow(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2019, 12, 31),  # End before start
        returns=np.array([]),
        garch_results=[],
        ensemble_stats={}
    )
    
    with pytest.raises(ValueError):
        temp_db.store_forecast_window(invalid_window, 'SPX')

if __name__ == '__main__':
    pytest.main([__file__])
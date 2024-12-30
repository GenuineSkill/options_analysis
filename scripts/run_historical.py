"""
Full historical analysis script for GARCH ensemble calculations.
"""

import logging
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import signal
import atexit
import gc
import psutil
import warnings

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

# Silence warnings
warnings.filterwarnings('ignore')
from arch.__future__ import reindexing

from utils.progress import ProgressMonitor
from data_manager.data_loader import DataLoader
from data_manager.database import GARCHDatabase
from garch.estimator import GARCHEstimator
from garch.forecaster import GARCHForecaster
from garch.models import ForecastWindow, GARCHResult

@dataclass
class GARCHResult:
    """Data class for GARCH model results"""
    model_type: str  # One of: 'garch', 'egarch', 'gjrgarch'
    distribution: str  # One of: 'normal', 'studentst'
    params: dict
    forecast_path: np.ndarray  # Shape: (252,) annualized volatility
    volatility_path: np.ndarray  # Historical volatilities

@dataclass
class ForecastWindow:
    """Data class for forecast windows"""
    start_date: datetime
    end_date: datetime
    returns: np.ndarray
    garch_results: list
    ensemble_stats: dict

def setup_logging():
    """Configure logging with both file and console output"""
    log_file = 'historical_analysis.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('historical_analysis')

def cleanup_handler():
    """Cleanup function to ensure proper shutdown"""
    logger.info("Cleaning up resources...")
    if 'market_db' in globals() and market_db is not None:
        market_db.close()
    if 'garch_db' in globals() and garch_db is not None:
        garch_db.close()
    gc.collect()
    logger.info("Cleanup complete")

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}. Starting cleanup...")
    cleanup_handler()
    sys.exit(0)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")

def validate_results(garch_db: GARCHDatabase) -> bool:
    """Validate database results"""
    try:
        # Check date ranges
        date_query = """
            SELECT MIN(end_date), MAX(end_date), COUNT(DISTINCT end_date)
            FROM forecast_windows
        """
        dates = garch_db.conn.execute(date_query).fetchone()
        if dates[0].year == 1970 or dates[1].year == 1970:
            logger.error("Invalid dates detected")
            return False
            
        # Check statistics variation
        stats_query = """
            SELECT horizon,
                   COUNT(*) as n,
                   COUNT(DISTINCT dev) as unique_dev,
                   COUNT(DISTINCT kev) as unique_kev
            FROM ensemble_stats
            GROUP BY horizon
        """
        stats = garch_db.conn.execute(stats_query).df()
        if stats['unique_dev'].min() < 10 or stats['unique_kev'].min() < 10:
            logger.error("Insufficient variation in statistics")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False

def log_progress(index_id: str, window_count: int, total_windows: int):
    """Log progress with statistics"""
    progress = (window_count / total_windows) * 100
    logger.info(f"""
    Progress for {index_id}: {progress:.1f}%
    Windows processed: {window_count}/{total_windows}
    Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB
    """)

def prepare_returns(prices: pd.Series) -> np.ndarray:
    """
    Prepare log returns for GARCH estimation
    """
    # Calculate log returns
    returns = np.log(prices / prices.shift(1))
    
    # Remove any NaN/Inf values
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna()
    
    # Validate returns
    if len(returns) < 1260:  # 5 years minimum
        raise ValueError("Insufficient data after cleaning")
        
    # Check for reasonable values
    if returns.std() > 0.1:  # If std > 10%, likely not in decimal form
        logger.warning("Converting returns to decimal form")
        returns = returns / 100
        
    # Final validation
    stats = {
        'mean': returns.mean(),
        'std': returns.std(),
        'skew': returns.skew(),
        'kurt': returns.kurtosis()
    }
    
    logger.info(f"Returns statistics:\n{pd.Series(stats).to_string()}")
    
    return returns.values

def process_index_data(index_data: pd.DataFrame,
                      estimator: GARCHEstimator,
                      garch_db: GARCHDatabase,
                      index_id: str,
                      dev_mode: bool = True):
    """Process historical data for an index"""
    
    # Data parameters
    if dev_mode:
        TRAIN_YEARS = 12  # ~3000 observations minimum
        TEST_YEARS = 2
    else:
        TRAIN_YEARS = 12
        TEST_YEARS = 20

    # Convert years to days
    MIN_TRAIN_DAYS = int(TRAIN_YEARS * 252)  # Minimum ~3024 trading days
    TEST_DAYS = int(TEST_YEARS * 252)
    
    # Create expanding windows
    windows = garch_db.create_expanding_windows(
        data=index_data,
        min_train_size=MIN_TRAIN_DAYS,
        test_size=TEST_DAYS,
        step_size=21  # Monthly steps
    )
    
    logger.info(f"""
    Starting analysis for {index_id}:
    Data range: {index_data.index[0]} to {index_data.index[-1]}
    Total observations: {len(index_data)}
    """)
    
    logger.info(f"""
    Created {len(windows)} expanding windows:
    First window: {windows[0][0].index[0]} to {windows[0][1].index[-1]}
    Last window: {windows[-1][0].index[0]} to {windows[-1][1].index[-1]}
    """)
    
    # Process each window
    for i, (train_data, test_data) in enumerate(windows):
        try:
            logger.info(f"Processing window {i+1}/{len(windows)}")
            logger.info(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
            logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
            
            # Prepare returns data
            returns = prepare_returns(train_data['price'])
            
            # Estimate models - pass returns, end date, and index_id
            garch_results = estimator.estimate_models(
                returns=returns,
                date=train_data.index[-1],
                index_id=index_id
            )
            
            # Calculate ensemble statistics for 1-month horizon
            ensemble_stats = estimator.calculate_ensemble_stats(
                garch_results,
                tau=21  # 1-month horizon
            )
            
            # Store results
            window = ForecastWindow(
                start_date=train_data.index[0],
                end_date=train_data.index[-1],
                returns=returns,
                garch_results=garch_results,
                ensemble_stats=ensemble_stats
            )
            
            garch_db.store_forecast_window(window, index_id)
            
        except Exception as e:
            logger.error(f"Error processing window {i+1}: {str(e)}")
            continue

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting historical analysis...")
    
    market_db = None
    garch_db = None
    
    try:
        # Set paths
        data_file = Path("data_manager/data/Global Equity Vol Durham 12.13.2024.csv")
        output_path = Path("results/historical")
        checkpoint_path = output_path / "checkpoints"
        
        # Ensure directories exist
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        market_db = DataLoader()
        garch_db = GARCHDatabase(output_path / "historical_results.db")
        
        # Load and validate data
        logger.info("Loading and validating market data...")
        data_dict = market_db.load_and_clean_data(str(data_file))
        
        # Initialize GARCH components
        logger.info("Initializing GARCH components...")
        estimator = GARCHEstimator(
            min_observations=3000,
            n_simulations=1000,
            random_seed=42,
            checkpoint_dir=checkpoint_path
        )
        
        # Development mode settings
        DEV_MODE = True  # Set to False for full historical calculation
        
        # Data parameters
        if DEV_MODE:
            TRAIN_YEARS = 12  # Increased from 5 to get 3000 observations
            TEST_YEARS = 2    # Keep 2 years of testing
        else:
            TRAIN_YEARS = 12  # Also update this for production
            TEST_YEARS = 20   # Full historical test period
        
        # Convert years to days
        TRAIN_DAYS = int(TRAIN_YEARS * 252)  # Approximately 3024 trading days
        TEST_DAYS = int(TEST_YEARS * 252)
        
        # Process each index
        for index_id in ['SPX']:
            logger.info(f"\nProcessing {index_id}...")
            log_memory_usage()
            
            # Get index data
            index_data = data_dict['daily_data'][
                data_dict['daily_data']['index_id'] == index_id
            ].sort_values('date')
            
            # Process index with development mode
            process_index_data(
                index_data=index_data,
                estimator=estimator,
                garch_db=garch_db,
                index_id=index_id,
                dev_mode=DEV_MODE
            )
            
            gc.collect()
            log_memory_usage()
            
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    finally:
        cleanup_handler() 
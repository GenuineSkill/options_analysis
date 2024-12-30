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

def process_index_data(index_data: pd.DataFrame, 
                      forecaster: GARCHForecaster,
                      garch_db: GARCHDatabase,
                      index_id: str,
                      dev_mode: bool = False) -> None:
    """Process a single index with optional development mode"""
    try:
        # Ensure date is datetime and set as index
        index_data['date'] = pd.to_datetime(index_data['date'])
        index_data = index_data.set_index('date')  # Set date as index
        
        # Clean and prepare returns data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns = index_data['price'].pct_change()
        
        returns = returns.ffill()  # Forward fill small gaps
        returns = returns.dropna()  # Remove any remaining NAs
        
        if len(returns) < forecaster.min_observations:
            logger.warning(f"Insufficient data for {index_id} after cleaning")
            return
            
        # In dev mode, limit to 5 years training + 2 years testing
        if dev_mode:
            min_idx = forecaster.min_observations  # About 5 years
            max_idx = min_idx + (252 * 2)  # Add 2 years for testing
            max_idx = min(max_idx, len(returns))
            logger.info(f"Development mode: Processing years {returns.index[min_idx].year} to {returns.index[max_idx-1].year}")
        else:
            max_idx = len(returns)
        
        # Set up progress monitoring
        n_windows = max_idx - forecaster.min_observations
        monitor = ProgressMonitor(
            total=n_windows,
            desc=f"Processing {index_id} windows",
            logger=logger
        )
        
        # Run analysis
        logger.info(f"Starting GARCH estimation for {index_id}...")
        
        for i in range(forecaster.min_observations, max_idx):
            try:
                window = returns.iloc[:i]
                date = window.index[-1]
                
                results = forecaster.estimator.estimate_models(
                    returns=window.values,
                    date=date,
                    index_id=index_id
                )
                
                if not results:
                    logger.warning(f"No valid models for {index_id} at {date}")
                    continue
                
                # Log model counts
                model_counts = {}
                for result in results:
                    key = f"{result.model_type}-{result.distribution}"
                    model_counts[key] = model_counts.get(key, 0) + 1
                logger.info(f"Models estimated for {date}: {model_counts}")
                
                if len(results) >= 3:
                    stats = forecaster.calculate_ensemble_stats(results)
                    if not stats:
                        logger.warning(f"Failed to calculate stats for {index_id} at {date}")
                        continue
                        
                    # Log statistics
                    for horizon, horizon_stats in stats.items():
                        logger.info(f"Stats for {date} {horizon}: {horizon_stats}")
                        
                    garch_db.store_forecast_window(
                        window=ForecastWindow(
                            start_date=window.index[0],
                            end_date=date,
                            returns=window.values,
                            garch_results=results,
                            ensemble_stats=stats
                        ),
                        index_id=index_id
                    )
                
                monitor.update()
                
                # Periodic validation and cleanup
                if i % 100 == 0:
                    gc.collect()
                    log_memory_usage()
                    if not validate_results(garch_db):
                        raise ValueError("Invalid results detected")
                        
            except Exception as e:
                logger.error(f"Error processing {index_id} window {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to process {index_id}: {str(e)}")
        raise

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
        forecaster = GARCHForecaster(
            min_observations=1260,  # About 5 years of daily data
            n_simulations=1000,
            random_seed=42,
            checkpoint_dir=checkpoint_path
        )
        
        # Add development mode flag
        DEV_MODE = True  # Set to False for full historical calculation
        
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
                forecaster=forecaster,
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
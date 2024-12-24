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

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from utils.progress import ProgressMonitor
from data_manager.data_loader import DataLoader
from data_manager.database import GARCHDatabase
from garch.estimator import GARCHEstimator
from garch.forecaster import GARCHForecaster

@dataclass
class GARCHResult:
    """Data class for GARCH model results"""
    model_type: str
    distribution: str
    params: dict
    forecasts_annualized: np.ndarray
    volatility_annualized: np.ndarray

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
    
    # Clear existing log file
    with open(log_file, 'w') as f:
        f.write('')
    
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
    if market_db is not None:
        market_db.close()
    if garch_db is not None:
        garch_db.close()
    logger.info("Cleanup complete")

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}. Starting cleanup...")
    cleanup_handler()
    sys.exit(0)

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
        market_db = DataLoader()  # Uses default path for market data
        garch_db = GARCHDatabase(output_path / "historical_results.db")
        
        # Load and validate data
        logger.info("Loading and validating market data...")
        data_dict = market_db.load_and_clean_data(str(data_file))
        
        # Initialize GARCH components
        logger.info("Initializing GARCH components...")
        estimator = GARCHEstimator(checkpoint_dir=checkpoint_path)
        forecaster = GARCHForecaster(estimator=estimator)
        
        # Add at the top, before the loop
        from arch.__future__ import reindexing  # Silence reindexing warning
        
        # Process each index
        for index_id in ['SPX', 'SX5E', 'UKX']:
            logger.info(f"\nProcessing {index_id}...")
            
            # Get index data
            index_data = data_dict['daily_data'][
                data_dict['daily_data']['index_id'] == index_id
            ].sort_values('date')
            
            # Ensure date is datetime
            index_data['date'] = pd.to_datetime(index_data['date'])
            
            # Calculate returns (already validated by DataLoader)
            returns = pd.Series(
                index_data['price'].pct_change().dropna().values,
                index=pd.to_datetime(index_data['date']).iloc[1:]  # Skip first row due to pct_change
            )
            
            # Set up progress monitoring
            n_windows = len(returns) - forecaster.min_window
            monitor = ProgressMonitor(
                total=n_windows,
                desc=f"Processing {index_id} windows",
                logger=logger
            )
            
            # Run analysis
            logger.info(f"Starting GARCH estimation for {index_id}...")
            for i in range(forecaster.min_window, len(returns)):
                window = returns.iloc[:i]
                date = returns.index[i-1]
                
                try:
                    # Process window
                    raw_results = estimator.estimate(window.values, date)
                    
                    # Convert raw results to GARCHResult objects
                    garch_results = []
                    for result in raw_results:
                        if isinstance(result, str):
                            # Parse the string result into components
                            # This is a temporary fix - the estimator should return proper objects
                            model_info = result.split('_')  # Assuming format like "GARCH11_Normal"
                            garch_results.append(GARCHResult(
                                model_type=model_info[0],
                                distribution=model_info[1] if len(model_info) > 1 else "Normal",
                                params={},  # Add proper parameter parsing if needed
                                forecasts_annualized=np.array([]),  # Add proper forecast data
                                volatility_annualized=np.array([])  # Add proper volatility data
                            ))
                        else:
                            garch_results.append(result)
                    
                    forecasts = forecaster.generate_forecasts(window.values, garch_results)
                    
                    # Create forecast window object
                    forecast_window = ForecastWindow(
                        start_date=window.index[0],
                        end_date=window.index[-1],
                        returns=window.values,
                        garch_results=garch_results,
                        ensemble_stats=forecasts
                    )
                    
                    # Store results in GARCH database
                    garch_db.store_forecast_window(forecast_window, index_id)
                    
                    # Log progress every 100 windows
                    if i % 100 == 0:
                        logger.info(f"{index_id}: Processed window {i} of {len(returns)}")
                    
                    monitor.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {index_id} window {i}: {str(e)}")
                    continue
            
            logger.info(f"Completed analysis for {index_id}")
        
        logger.info("Historical analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Historical analysis failed: {str(e)}")
        raise
    finally:
        cleanup_handler() 
#!/usr/bin/env python
"""
Full historical calculation pipeline for SPX volatility analysis.
Coordinates GARCH estimation, expanding windows, and regression analysis.
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import time
import psutil
import traceback
from tqdm import tqdm
import warnings
from arch.__future__ import reindexing

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from garch.estimator import GARCHEstimator
from garch.forecaster import GARCHForecaster
from data_manager.database import GARCHDatabase
from regression.expander import ExpandingWindowAnalyzer
from utils.visualization import GARCHVisualizer

class ProgressMonitor:
    """Tracks and reports progress of analysis pipeline"""
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.checkpoints = {}
        
    def checkpoint(self, name: str):
        """Record timing for a checkpoint"""
        now = time.time()
        duration = now - self.last_checkpoint
        self.checkpoints[name] = {
            'duration': duration,
            'memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
        }
        self.last_checkpoint = now
        
    def report(self):
        """Generate checkpoint report"""
        total_time = time.time() - self.start_time
        report = ["Performance Report:", "-----------------"]
        
        for name, stats in self.checkpoints.items():
            report.append(f"{name}:")
            report.append(f"  Duration: {stats['duration']:.2f} seconds")
            report.append(f"  Memory: {stats['memory']:.2f} MB")
            
        report.append("-----------------")
        report.append(f"Total Time: {total_time:.2f} seconds")
        return "\n".join(report)

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging with both file and console handlers
    
    Parameters:
    -----------
    output_dir : Path
        Directory for log file
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"historical_calculation_{timestamp}.log"
    
    # Configure logging
    logger = logging.getLogger("historical_calculator")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_data(returns_df: pd.DataFrame, iv_df: pd.DataFrame) -> None:
    """Validate the loaded data meets minimum requirements"""
    
    # Check for minimum data requirements (reduced from 1260 to 252)
    if len(returns_df) < 252:
        raise ValueError(f"Insufficient returns data: {len(returns_df)} < 252")
        
    if len(iv_df) < 252:
        raise ValueError(f"Insufficient IV data: {len(iv_df)} < 252")
        
    # Check for missing values
    if returns_df.isna().any().any():
        warnings.warn("Missing values in returns data")
        
    if iv_df.isna().any().any():
        warnings.warn("Missing values in IV data will be interpolated")
        
    # Check for extreme values in returns
    if abs(returns_df['returns']).max() > 20:
        warnings.warn("Extreme return values detected")
        
    # Ensure perfect alignment
    if not returns_df.index.equals(iv_df.index):
        raise ValueError("Index mismatch between returns and IV data")

def load_spx_data(data_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare SPX data with progress reporting"""
    logger.info("Loading SPX data...")
    
    try:
        # Create data directory if it doesn't exist
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Use the correct data file
        data_file = data_path / "Global Equity Vol Durham 12.13.2024.csv"
        
        # Load data
        logger.info(f"Reading data from: {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Convert first column to datetime and set as index
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Sort index to ensure chronological order
        df.sort_index(inplace=True)
        
        # Process returns
        logger.info("Processing returns data...")
        returns = np.log(df['SPX'] / df['SPX'].shift(1)).dropna() * 100
        returns_df = pd.DataFrame({'returns': returns})
        
        # Process implied volatilities
        logger.info("Processing implied volatility data...")
        iv_columns = ['SPX_1M', 'SPX_2M', 'SPX_3M', 'SPX_6M', 'SPX_12M']
        iv_df = df[iv_columns].rename(columns={
            'SPX_1M': '1M',
            'SPX_2M': '2M',
            'SPX_3M': '3M',
            'SPX_6M': '6M',
            'SPX_12M': '12M'
        })
        
        # Handle missing values
        if iv_df.isna().any().any():
            logger.warning("Interpolating missing IV values...")
            iv_df = iv_df.interpolate(method='time')
            
        # Ensure perfect alignment between returns and IV data
        common_dates = returns_df.index.intersection(iv_df.index)
        returns_df = returns_df.loc[common_dates]
        iv_df = iv_df.loc[common_dates]
        
        # Get all available data up to the latest date
        latest_date = returns_df.index.max()
        earliest_date = latest_date - pd.Timedelta(days=1000)  # Get ~3 years of data
        
        returns_df = returns_df[returns_df.index >= earliest_date]
        iv_df = iv_df[iv_df.index >= earliest_date]
        
        # Convert to date objects after processing
        returns_df.index = returns_df.index.date
        iv_df.index = iv_df.index.date
        
        # Validate data
        validate_data(returns_df, iv_df)
        
        logger.info(f"Loaded {len(returns_df)} days of returns and {len(iv_df)} days of IV data")
        return returns_df, iv_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def setup_analysis_components(db_path: Path, logger: logging.Logger) -> Dict:
    """Initialize analysis components with error handling"""
    logger.info("Initializing analysis components...")
    
    try:
        # Create GARCH estimator with smaller minimum window
        logger.info("Creating GARCH estimator...")
        estimator = GARCHEstimator(
            min_observations=252,  # One trading year
            n_simulations=1000,
            random_seed=42
        )
        
        # Create forecaster
        logger.info("Creating forecaster...")
        forecaster = GARCHForecaster(
            estimator=estimator,
            min_window=252  # Match the estimator's minimum observations
        )
        
        # Create analyzer with db_path
        logger.info("Creating analyzer...")
        analyzer = ExpandingWindowAnalyzer(
            db_path=db_path,
            forecaster=forecaster
        )
        
        # Create visualizer
        logger.info("Creating visualizer...")
        visualizer = GARCHVisualizer()
        
        return {
            'estimator': estimator,
            'forecaster': forecaster,
            'analyzer': analyzer,
            'visualizer': visualizer
        }
    except Exception as e:
        logger.error(f"Error setting up components: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def run_analysis(components: Dict, returns_df: pd.DataFrame, iv_df: pd.DataFrame, 
                output_dir: Path, logger: logging.Logger, monitor: Any) -> Dict:
    """Run the analysis pipeline with progress monitoring"""
    logger.info("Starting analysis pipeline...")
    
    try:
        analyzer = components['analyzer']
        visualizer = components['visualizer']
        
        # Get all data up to the latest date
        dates = returns_df.index
        returns = returns_df['returns'].values
        implied_vols = iv_df.values
        
        # Process all available data
        logger.info("Processing expanding windows...")
        results = analyzer.process_index(
            index_id='SPX',
            dates=dates,
            returns=returns,
            implied_vols=implied_vols
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer.plot_results(
            results=results,
            output_path=output_dir,
            show_plots=True
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def cleanup(components: Dict, logger: logging.Logger):
    """Clean up resources"""
    try:
        components['analyzer'].close()
        components['visualizer'].close_all()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main execution function with enhanced error handling"""
    monitor = ProgressMonitor()
    logger = None
    
    try:
        # Setup paths
        project_root = Path(__file__).parent
        data_path = project_root / "data_manager/data"
        db_path = project_root / "results/garch_results.db"
        output_dir = project_root / "results/spx_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        logger = setup_logging(output_dir)
        logger.info("Starting historical calculation pipeline...")
        
        monitor.checkpoint("Initialization")
        
        # Load data
        returns_df, iv_df = load_spx_data(data_path, logger)
        monitor.checkpoint("Data Loading")
        
        # Initialize components
        components = setup_analysis_components(db_path, logger)
        monitor.checkpoint("Component Setup")
        
        # Run analysis
        results = run_analysis(components, returns_df, iv_df, output_dir, logger, monitor)
        
        # Clean up
        cleanup(components, logger)
        
        logger.info("Pipeline completed successfully")
        monitor.checkpoint("Pipeline Complete")
        
    except Exception as e:
        if logger:
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        raise
        
if __name__ == '__main__':
    main()
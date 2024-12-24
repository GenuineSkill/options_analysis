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
        logger.info(f"Total rows in CSV: {len(df)}")
        
        # Convert date column and set index
        df.index = pd.to_datetime(df['Unnamed: 0'])
        logger.info(f"Date range in CSV: {df.index[0]} to {df.index[-1]}")
        
        # Log columns for debugging
        logger.info(f"Columns found: {df.columns.tolist()}")
        
        # Process returns data
        logger.info("Processing returns data...")
        returns_df = pd.DataFrame(index=df.index)
        returns_df['returns'] = df['SPX'].pct_change()
        
        # Process implied volatility data
        logger.info("Processing implied volatility data...")
        iv_columns = ['SPX_1M', 'SPX_2M', 'SPX_3M', 'SPX_6M', 'SPX_12M']
        iv_df = df[iv_columns]
        
        # Handle missing values
        if iv_df.isna().any().any():
            logger.warning("Interpolating missing IV values...")
            iv_df = iv_df.interpolate(method='linear', axis=0)
        
        # Validate data
        validate_data(returns_df, iv_df)
        
        logger.info(f"Loaded {len(returns_df)} days of returns and {len(iv_df)} days of IV data")
        return returns_df, iv_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def initialize_components(logger: logging.Logger = None) -> Dict:
    """Initialize all analysis components"""
    if logger is None:
        logger = logging.getLogger('historical_calculator')
        
    logger.info("Creating GARCH estimator...")
    estimator = GARCHEstimator(
        min_observations=1260,  # Five years as per Durham's paper
        n_simulations=1000,
        random_seed=42
    )
    
    logger.info("Creating forecaster...")
    forecaster = GARCHForecaster(
        estimator=estimator,
        min_window=1260  # Match estimator's minimum observations
    )
    
    logger.info("Creating analyzer...")
    analyzer = ExpandingWindowAnalyzer(forecaster=forecaster)
    
    logger.info("Creating visualizer...")
    visualizer = GARCHVisualizer()
    
    return {
        'estimator': estimator,
        'forecaster': forecaster,
        'analyzer': analyzer,
        'visualizer': visualizer
    }

def run_analysis(components: Dict, returns_df: pd.DataFrame, iv_df: pd.DataFrame, 
                output_dir: Path, logger: logging.Logger, monitor: Any) -> Dict:
    """Run the analysis pipeline with progress monitoring and checkpointing"""
    logger.info("Starting analysis pipeline...")
    
    try:
        analyzer = components['analyzer']
        visualizer = components['visualizer']
        
        # Create checkpoint directory
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")
        
        # Get all data
        dates = returns_df.index
        returns = returns_df['returns'].values
        implied_vols = iv_df.values
        
        # Process expanding windows with checkpointing
        logger.info("Processing expanding windows...")
        results = analyzer.process_index(
            index_id='SPX',
            dates=dates,
            returns=returns,
            implied_vols=implied_vols,
            output_dir=output_dir  # Pass output directory for results saving
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        visualizer.plot_results(
            results=results,
            output_path=plot_dir,
            show_plots=True
        )
        
        logger.info("Pipeline completed successfully")
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
    """Main entry point with configuration and setup"""
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting historical calculation pipeline...")
        
        # Setup directories
        root_dir = Path(__file__).parent
        data_dir = root_dir / "data_manager" / "data"
        output_dir = root_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        returns_df, iv_df = load_spx_data(data_dir, logger)
        
        # Initialize components
        logger.info("Initializing analysis components...")
        components = initialize_components()
        
        # Configure checkpointing in forecaster
        components['forecaster'].checkpoint_dir = output_dir / "checkpoints"
        
        # Run analysis with progress monitoring
        monitor = None  # TODO: Add progress monitoring
        results = run_analysis(components, returns_df, iv_df, output_dir, logger, monitor)
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
        
if __name__ == '__main__':
    main()
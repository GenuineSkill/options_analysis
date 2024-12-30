"""
Pilot script for testing historical calculation pipeline with a small dataset.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from calculate_historical import load_spx_data, initialize_components, run_analysis
from utils.progress import ProgressMonitor
from garch.garch_forecaster import GARCHForecaster

# Data parameters
TRAIN_YEARS = 12  # Update from 5 to 12 years
TRAIN_DAYS = int(TRAIN_YEARS * 252)  # ~3024 days

forecaster = GARCHForecaster(
    min_observations=3000,  # Update from 1260
    n_simulations=1000
)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pilot_run.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('pilot')
    
    # Set paths
    data_path = Path("data_manager/data")
    output_path = Path("results/pilot")
    
    # Run analysis with progress monitoring
    try:
        # Load data
        logger.info("Starting pilot run...")
        returns_df, iv_df = load_spx_data(data_path, logger)
        logger.info(f"Loaded data from {returns_df.index[0]} to {returns_df.index[-1]}")
        
        # Initialize components
        components = initialize_components(logger)
        
        # Run analysis
        results = run_analysis(
            components=components,
            returns_df=returns_df,
            iv_df=iv_df,
            output_dir=output_path,
            logger=logger,
            monitor=ProgressMonitor(total=len(returns_df), desc="Processing windows")
        )
        
        logger.info("Pilot analysis completed successfully")
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Pilot analysis failed: {str(e)}")
        raise
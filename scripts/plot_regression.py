"""Script to generate regression analysis plots"""

import logging
from pathlib import Path
import sys
import duckdb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.level_regression import LevelRegression
from regression.visualization import RegressionVisualizer

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('regression_plots')
    
    try:
        # Connect to results database
        results_db = Path("results/historical/historical_results.db")
        conn = duckdb.connect(str(results_db))
        
        # Initialize regression and visualizer
        level_reg = LevelRegression(conn)
        viz = RegressionVisualizer(level_reg)
        
        # Generate plots for SPX
        logger.info("Generating plots for SPX...")
        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all tenor plots
        viz.save_plots('SPX', output_dir=str(output_dir))
        
        logger.info(f"Plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
"""Script to generate error correction analysis plots"""

import logging
from pathlib import Path
import sys
import duckdb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.visualization import RegressionVisualizer
from regression.error_correction import ErrorCorrectionModel

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('error_correction_plots')
    
    try:
        # Connect to results database
        results_db = Path("results/historical/historical_results.db")
        conn = duckdb.connect(str(results_db))
        
        # Initialize models and visualizer
        ec_model = ErrorCorrectionModel(conn)
        viz = RegressionVisualizer(ec_model)
        
        # Generate plots for SPX
        logger.info("Generating error correction plots for SPX...")
        output_dir = Path("results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create plots for all tenors
        for tenor in ['1M', '2M', '3M', '6M', '12M']:
            logger.info(f"Processing {tenor}...")
            fig = viz.plot_error_correction('SPX', tenor)
            output_file = output_dir / f'SPX_{tenor}_error_correction.html'
            fig.write_html(str(output_file))
            logger.info(f"Saved plot to {output_file}")
        
        logger.info(f"All plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 
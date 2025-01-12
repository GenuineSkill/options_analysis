# scripts/test_regression.py
"""Test script for level regression analysis"""

import logging
from pathlib import Path
import duckdb
import sys
import pandas as pd

# Add the project root directory to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Now import the modules
from data_manager.data_loader import DataLoader
from data_manager.holiday_handler import HolidayHandler
from regression.level_regression import LevelRegression

def setup_ensemble_stats_table(conn, data_file: Path):
    """Create and populate ensemble statistics table from historical results"""
    logger = logging.getLogger('regression_test')
    
    try:
        # First, check raw data in forecast_windows and ensemble_stats
        raw_data = conn.execute("""
            -- Check forecast_windows
            SELECT 'forecast_windows' as table_name,
                   index_id,
                   COUNT(*) as total_records,
                   COUNT(DISTINCT window_id) as unique_windows,
                   COUNT(DISTINCT end_date) as unique_dates,
                   MIN(end_date) as start_date,
                   MAX(end_date) as end_date
            FROM forecast_windows
            GROUP BY index_id
            
            UNION ALL
            
            -- Check ensemble_stats
            SELECT 'ensemble_stats' as table_name,
                   fw.index_id,
                   COUNT(*) as total_records,
                   COUNT(DISTINCT es.window_id) as unique_windows,
                   COUNT(DISTINCT fw.end_date) as unique_dates,
                   MIN(fw.end_date) as start_date,
                   MAX(fw.end_date) as end_date
            FROM ensemble_stats es
            JOIN forecast_windows fw ON es.window_id = fw.window_id
            GROUP BY fw.index_id
            
            ORDER BY table_name, index_id
        """).df()
        
        logger.info("\nRaw data coverage in database:")
        logger.info(raw_data)
        
        # First, check index coverage
        index_coverage = conn.execute("""
            SELECT 
                fw.index_id,
                COUNT(DISTINCT fw.end_date) as dates,
                COUNT(DISTINCT es.window_id) as windows,
                MIN(fw.end_date) as start_date,
                MAX(fw.end_date) as end_date
            FROM forecast_windows fw
            JOIN ensemble_stats es ON es.window_id = fw.window_id
            GROUP BY fw.index_id
            ORDER BY fw.index_id
        """).df()
        
        logger.info("\nEnsemble statistics coverage by index:")
        logger.info(index_coverage)
        
        # Now when setting up implied vol table, maintain index relationships
        logger.info("\nAdjusting implied vol data to match ensemble dates...")
        conn.execute("DROP TABLE IF EXISTS implied_vol_aligned")
        conn.execute("""
            CREATE TABLE implied_vol_aligned AS
            SELECT iv.*
            FROM implied_volatilities iv
            INNER JOIN (
                SELECT DISTINCT fw.end_date, fw.index_id
                FROM ensemble_stats es
                JOIN forecast_windows fw ON es.window_id = fw.window_id
            ) e ON iv.date = e.end_date AND iv.index_id = e.index_id
        """)
        
        # Verify alignment with index preservation
        alignment = conn.execute("""
            SELECT 
                iv.index_id,
                COUNT(DISTINCT iv.date) as iv_dates,
                COUNT(DISTINCT e.end_date) as ensemble_dates,
                COUNT(DISTINCT CASE WHEN e.end_date IS NOT NULL THEN iv.date END) as matching_dates
            FROM implied_vol_aligned iv
            LEFT JOIN (
                SELECT DISTINCT fw.end_date, fw.index_id
                FROM ensemble_stats es
                JOIN forecast_windows fw ON es.window_id = fw.window_id
            ) e ON iv.date = e.end_date AND iv.index_id = e.index_id
            GROUP BY iv.index_id
            ORDER BY iv.index_id
        """).df()
        
        logger.info("\nData alignment summary by index:")
        logger.info(alignment)
        
    except Exception as e:
        logger.error(f"Error setting up ensemble statistics: {str(e)}")
        raise

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('regression_test')
    
    # Define paths using project root
    root_dir = Path(__file__).parent.parent
    results_db = root_dir / "results" / "historical" / "historical_results.db"
    data_file = root_dir / "data_manager" / "data" / "Global Equity Vol Durham 12.13.2024.csv"
    
    logger.info(f"Using data file: {data_file}")
    logger.info(f"Results will be saved to: {results_db}")
    
    try:
        # Create results directory if it doesn't exist
        results_db.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        conn = duckdb.connect(str(results_db))
        
        # Set up tables in correct order
        setup_ensemble_stats_table(conn, data_file)
        
        # Run regressions
        level_reg = LevelRegression(conn)
        try:
            results = level_reg.run_all_regressions()
            
            # Save results
            results_path = results_db.parent / "level_regression_results.csv"
            results.to_csv(results_path, index=False)
            logger.info(f"\nRegression results saved to: {results_path}")
            
            # Log summary statistics
            logger.info("\nRegression Summary:")
            summary = results.groupby(['index_id', 'tenor']).agg({
                'r_squared': 'mean',
                'nobs': 'mean'
            }).round(3)
            logger.info(summary)
            
        except Exception as e:
            logger.error(f"Error running regressions: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in regression test: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
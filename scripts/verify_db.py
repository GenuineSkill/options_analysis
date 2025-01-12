# scripts/verify_db.py
from pathlib import Path
import logging
import duckdb
import pandas as pd

def check_database():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('db_check')
    
    # Define expected paths
    results_path = Path("results/historical")
    db_path = results_path / "historical_results.db"
    
    # Check directory
    logger.info(f"Checking results directory: {results_path}")
    if not results_path.exists():
        logger.error(f"Results directory not found at: {results_path}")
        return
        
    # Check database file
    logger.info(f"Checking database file: {db_path}")
    if not db_path.exists():
        logger.error(f"Database file not found at: {db_path}")
        return
        
    # Try to connect
    try:
        conn = duckdb.connect(str(db_path))
        
        # Get list of tables
        tables = conn.execute("SHOW TABLES").fetchall()
        
        logger.info("\n=== Database Tables ===")
        for table in tables:
            table_name = table[0]
            logger.info(f"\nTable: {table_name}")
            
            # Get column information
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            logger.info("\nColumns:")
            for col in columns:
                logger.info(f"- {col[0]}: {col[1]}")
            
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.info(f"\nRow count: {count:,}")
            
            # Sample data
            logger.info("\nSample data:")
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
            logger.info(sample)
            
        # Analyze forecast windows
        logger.info("\n=== Forecast Windows Analysis ===")
        windows_query = """
            SELECT 
                index_id,
                MIN(start_date)::DATE as earliest_start,
                MAX(end_date)::DATE as latest_end,
                COUNT(*) as n_windows
            FROM forecast_windows
            GROUP BY index_id
            ORDER BY index_id
        """
        windows_df = conn.execute(windows_query).df()
        logger.info("\nForecast windows by index:")
        logger.info(windows_df)
        
        # Analyze ensemble statistics
        logger.info("\n=== Ensemble Statistics Analysis ===")
        stats_query = """
            SELECT 
                horizon,
                horizon_days,
                COUNT(*) as n_observations,
                AVG(n_models) as avg_models_per_window
            FROM ensemble_stats
            GROUP BY horizon, horizon_days
            ORDER BY horizon_days
        """
        stats_df = conn.execute(stats_query).df()
        logger.info("\nEnsemble statistics summary:")
        logger.info(stats_df)
        
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_database()
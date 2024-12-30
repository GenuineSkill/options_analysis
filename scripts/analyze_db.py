"""
Analyze DuckDB database structure and contents from GARCH ensemble analysis
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

def analyze_database(db_path: str):
    """Analyze database structure and contents"""
    logger = logging.getLogger(__name__)
    conn = duckdb.connect(db_path)
    
    try:
        # Get list of tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print("\n=== Database Tables ===")
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            
            # Get column information
            columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
            print("\nColumns:")
            for col in columns:
                print(f"- {col[0]}: {col[1]}")
            
            # Get row count
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"\nRow count: {count:,}")
        
        # Analyze forecast windows
        print("\n=== Forecast Windows Analysis ===")
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
        print("\nForecast windows by index:")
        print(windows_df)
        
        # Analyze GARCH models
        print("\n=== GARCH Models Analysis ===")
        analyze_garch_models(conn)
        
        # Analyze ensemble statistics
        print("\n=== Ensemble Statistics Analysis ===")
        stats_query = """
            WITH horizon_days AS (
                SELECT 
                    horizon,
                    CASE 
                        WHEN horizon = '1M' THEN 21
                        WHEN horizon = '2M' THEN 42
                        WHEN horizon = '3M' THEN 63
                        WHEN horizon = '6M' THEN 126
                        WHEN horizon = '12M' THEN 252
                    END as days
                FROM ensemble_stats
                GROUP BY horizon
            )
            SELECT 
                es.horizon,
                hd.days as horizon_days,
                COUNT(*) as n_observations,
                AVG(n_models) as avg_models_per_window,
                AVG(gev) as avg_gev,
                AVG(evoev) as avg_evoev,
                AVG(dev) as avg_dev,
                AVG(kev) as avg_kev,
                AVG(sevts) as avg_sevts
            FROM ensemble_stats es
            JOIN horizon_days hd ON es.horizon = hd.horizon
            GROUP BY es.horizon, hd.days
            ORDER BY hd.days
        """
        stats_df = conn.execute(stats_query).df()
        print("\nEnsemble statistics summary by horizon:")
        print(stats_df)
        
        # Sample some recent results
        print("\n=== Recent Results Sample ===")
        recent_query = """
            WITH recent_windows AS (
                SELECT 
                    fw.window_id,
                    fw.index_id,
                    fw.end_date::DATE as date,
                    es.horizon,
                    es.gev,
                    es.evoev,
                    es.dev,
                    es.kev,
                    es.sevts
                FROM forecast_windows fw
                JOIN ensemble_stats es ON fw.window_id = es.window_id
                WHERE fw.end_date >= (SELECT MAX(end_date) - INTERVAL '30 days' FROM forecast_windows)
            )
            SELECT *
            FROM recent_windows
            ORDER BY date DESC, index_id, horizon
            LIMIT 15
        """
        recent_df = conn.execute(recent_query).df()
        print("\nSample of recent results:")
        print(recent_df)
        
    finally:
        conn.close()

def analyze_garch_models(conn):
    """Analyze GARCH model distribution"""
    models_query = """
        SELECT 
            CASE 
                WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '0' THEN 'garch'
                WHEN model_type IN ('EGARCH', 'egarch') THEN 'egarch'
                WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '1' THEN 'gjrgarch'
                ELSE model_type
            END as model_type,
            distribution,
            COUNT(*) as n_estimates,
            COUNT(DISTINCT window_id) as n_windows
        FROM garch_results
        GROUP BY 
            CASE 
                WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '0' THEN 'garch'
                WHEN model_type IN ('EGARCH', 'egarch') THEN 'egarch'
                WHEN model_type IN ('GARCH', 'garch') AND parameters::json->>'o' = '1' THEN 'gjrgarch'
                ELSE model_type
            END,
            distribution
        ORDER BY model_type, distribution
    """
    models_df = conn.execute(models_query).df()
    print("\nGARCH model estimates:")
    print(models_df)
    
    # Add model type distribution analysis
    print("\nModel type distribution:")
    for model_type in ['garch', 'egarch', 'gjrgarch']:
        subset = models_df[models_df['model_type'] == model_type]
        if not subset.empty:
            print(f"\n{model_type.upper()}:")
            print(f"  Total estimates: {subset['n_estimates'].sum():,}")
            print(f"  Unique windows: {subset['n_windows'].sum():,}")
            print("  Distributions:")
            for _, row in subset.iterrows():
                print(f"    {row['distribution']}: {row['n_estimates']:,} estimates")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Set path to database
    db_path = Path("results/historical/historical_results.db")
    
    print(f"Analyzing database: {db_path}")
    analyze_database(str(db_path))
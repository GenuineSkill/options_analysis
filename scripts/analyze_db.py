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
        
        # Analyze ensemble statistics by horizon with row-level inspection
        print("\n=== Ensemble Statistics Analysis ===")
        stats_query = """
            WITH typed_stats AS (
                SELECT 
                    horizon,
                    horizon_days,
                    n_models,
                    gev,
                    evoev,
                    -- Use string cast as intermediate step
                    CAST(CAST(dev AS VARCHAR) AS DOUBLE) as dev,
                    CAST(CAST(kev AS VARCHAR) AS DOUBLE) as kev,
                    sevts
                FROM ensemble_stats
                WHERE dev IS NOT NULL 
                  AND kev IS NOT NULL
            ),
            horizon_stats AS (
                SELECT 
                    horizon,
                    horizon_days,
                    COUNT(*) as n_observations,
                    AVG(n_models) as avg_models_per_window,
                    SUM(gev) as sum_gev,
                    SUM(evoev) as sum_evoev,
                    SUM(dev) as sum_dev,
                    SUM(kev) as sum_kev,
                    SUM(sevts) as sum_sevts,
                    COUNT(dev) as dev_count,
                    COUNT(kev) as kev_count
                FROM typed_stats
                GROUP BY horizon, horizon_days
            )
            SELECT 
                horizon,
                horizon_days,
                n_observations,
                avg_models_per_window,
                sum_gev / n_observations as avg_gev,
                sum_evoev / n_observations as avg_evoev,
                CASE 
                    WHEN dev_count > 0 THEN sum_dev / dev_count 
                    ELSE NULL 
                END as avg_dev,
                CASE 
                    WHEN kev_count > 0 THEN sum_kev / kev_count 
                    ELSE NULL 
                END as avg_kev,
                sum_sevts / n_observations as avg_sevts
            FROM horizon_stats
            ORDER BY horizon_days
        """
        
        try:
            print("\nSample data inspection:")
            sample_df = conn.execute(stats_query).df()
            print(sample_df)
            
            print("\nCalculating statistics by horizon...")
            stats_query = """
                WITH typed_stats AS (
                    SELECT 
                        horizon,
                        horizon_days,
                        n_models,
                        gev,
                        evoev,
                        -- Use string cast as intermediate step
                        CAST(CAST(dev AS VARCHAR) AS DOUBLE) as dev,
                        CAST(CAST(kev AS VARCHAR) AS DOUBLE) as kev,
                        sevts
                    FROM ensemble_stats
                    WHERE dev IS NOT NULL 
                      AND kev IS NOT NULL
                ),
                horizon_stats AS (
                    SELECT 
                        horizon,
                        horizon_days,
                        COUNT(*) as n_observations,
                        AVG(n_models) as avg_models_per_window,
                        SUM(gev) as sum_gev,
                        SUM(evoev) as sum_evoev,
                        SUM(dev) as sum_dev,
                        SUM(kev) as sum_kev,
                        SUM(sevts) as sum_sevts,
                        COUNT(dev) as dev_count,
                        COUNT(kev) as kev_count
                    FROM typed_stats
                    GROUP BY horizon, horizon_days
                )
                SELECT 
                    horizon,
                    horizon_days,
                    n_observations,
                    avg_models_per_window,
                    sum_gev / n_observations as avg_gev,
                    sum_evoev / n_observations as avg_evoev,
                    CASE 
                        WHEN dev_count > 0 THEN sum_dev / dev_count 
                        ELSE NULL 
                    END as avg_dev,
                    CASE 
                        WHEN kev_count > 0 THEN sum_kev / kev_count 
                        ELSE NULL 
                    END as avg_kev,
                    sum_sevts / n_observations as avg_sevts
                FROM horizon_stats
                ORDER BY horizon_days
            """
            stats_df = conn.execute(stats_query).df()
            print("\nEnsemble statistics summary by horizon:")
            pd.set_option('display.float_format', lambda x: '%.3f' % x if pd.notnull(x) else 'NA')
            print(stats_df)
            
        except Exception as e:
            print(f"\nError in detailed analysis: {str(e)}")
            print("Running simplified analysis...")
            simple_query = """
                WITH typed_stats AS (
                    SELECT 
                        horizon,
                        horizon_days,
                        n_models,
                        gev,
                        evoev,
                        -- Use string cast as intermediate step
                        CAST(CAST(dev AS VARCHAR) AS DOUBLE) as dev,
                        CAST(CAST(kev AS VARCHAR) AS DOUBLE) as kev,
                        sevts
                    FROM ensemble_stats
                )
                SELECT 
                    horizon,
                    horizon_days,
                    COUNT(*) as n_observations,
                    AVG(n_models) as avg_models_per_window,
                    AVG(gev) as avg_gev,
                    AVG(evoev) as avg_evoev,
                    AVG(dev) as avg_dev,
                    AVG(kev) as avg_kev,
                    AVG(sevts) as avg_sevts
                FROM typed_stats
                GROUP BY horizon, horizon_days
                ORDER BY horizon_days
            """
            stats_df = conn.execute(simple_query).df()
            print("\nSimplified ensemble statistics summary by horizon:")
            pd.set_option('display.float_format', lambda x: '%.3f' % x)
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
            LIMIT 25
        """
        recent_df = conn.execute(recent_query).df()
        print("\nSample of recent results:")
        print(recent_df)
        
        # Analyze term structure patterns
        print("\n=== Term Structure Analysis ===")
        term_structure_query = """
            WITH window_stats AS (
                SELECT 
                    fw.window_id,
                    fw.end_date::DATE as date,
                    es.horizon,
                    es.horizon_days,
                    es.gev,
                    es.sevts
                FROM forecast_windows fw
                JOIN ensemble_stats es ON fw.window_id = es.window_id
            )
            SELECT 
                horizon,
                horizon_days,
                COUNT(*) as n_obs,
                AVG(gev) as avg_gev,
                AVG(sevts) as avg_sevts,
                CASE 
                    WHEN COUNT(*) > 1 THEN CORR(gev, sevts)
                    ELSE NULL 
                END as gev_sevts_corr
            FROM window_stats
            GROUP BY horizon, horizon_days
            ORDER BY horizon_days
        """
        
        try:
            term_df = conn.execute(term_structure_query).df()
            print("\nTerm structure patterns by horizon:")
            pd.set_option('display.float_format', lambda x: '%.3f' % x if pd.notnull(x) else 'NA')
            print(term_df)
        except Exception as e:
            print(f"\nError analyzing term structure: {str(e)}")
            print("Running simplified term structure analysis...")
            simple_term_query = """
                SELECT 
                    horizon,
                    horizon_days,
                    COUNT(*) as n_obs,
                    AVG(gev) as avg_gev,
                    AVG(sevts) as avg_sevts
                FROM ensemble_stats
                GROUP BY horizon, horizon_days
                ORDER BY horizon_days
            """
            term_df = conn.execute(simple_term_query).df()
            print("\nSimplified term structure patterns by horizon:")
            print(term_df)
        
        # Add diagnostic query for NaN investigation
        print("\n=== NaN Investigation ===")
        nan_query = """
            SELECT 
                horizon,
                horizon_days,
                COUNT(*) as total_rows,
                COUNT(dev) as dev_count,
                COUNT(kev) as kev_count,
                MIN(dev) as min_dev,
                MAX(dev) as max_dev,
                MIN(kev) as min_kev,
                MAX(kev) as max_kev,
                SUM(CASE WHEN dev IS NULL THEN 1 ELSE 0 END) as null_dev_count,
                SUM(CASE WHEN kev IS NULL THEN 1 ELSE 0 END) as null_kev_count
            FROM ensemble_stats
            GROUP BY horizon, horizon_days
            ORDER BY horizon_days
        """
        nan_df = conn.execute(nan_query).df()
        print("\nNaN value analysis:")
        print(nan_df)
        
        # Sample rows with NULL values
        print("\nSample rows with NULL values:")
        null_sample_query = """
            SELECT *
            FROM ensemble_stats
            WHERE dev IS NULL OR kev IS NULL
            LIMIT 5
        """
        null_sample = conn.execute(null_sample_query).df()
        print(null_sample)
        
        # Direct value inspection
        print("\nDirect value inspection:")
        value_query = """
            SELECT 
                horizon,
                horizon_days,
                COUNT(*) as count,
                MIN(dev) as min_dev,
                MAX(dev) as max_dev,
                MIN(kev) as min_kev,
                MAX(kev) as max_kev
            FROM ensemble_stats
            WHERE dev != 'NaN' AND kev != 'NaN'
            GROUP BY horizon, horizon_days
            ORDER BY horizon_days
        """
        value_df = conn.execute(value_query).df()
        print(value_df)
        
    finally:
        conn.close()

def analyze_garch_models(conn):
    """Analyze GARCH model distribution"""
    models_query = """
        SELECT 
            model_type,
            distribution,
            COUNT(*) as n_estimates,
            COUNT(DISTINCT window_id) as n_windows,
            AVG(CAST(parameters::json->>'omega' as FLOAT)) as avg_omega,
            AVG(CAST(parameters::json->>'alpha[1]' as FLOAT)) as avg_alpha,
            AVG(CAST(parameters::json->>'beta[1]' as FLOAT)) as avg_beta
        FROM garch_results
        GROUP BY model_type, distribution
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
                print(f"      ω: {row['avg_omega']:.6f}")
                print(f"      α: {row['avg_alpha']:.6f}")
                print(f"      β: {row['avg_beta']:.6f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Set path to database
    db_path = Path("results/historical/historical_results.db")
    
    print(f"Analyzing database: {db_path}")
    analyze_database(str(db_path))
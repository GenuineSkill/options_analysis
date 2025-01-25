"""Validation checks for regression sequence"""

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

def verify_residual_storage(conn, index_id: str, tenor: str):
    """Verify expanding window residuals are properly stored"""
    try:
        # Check residual availability
        results = conn.execute("""
            SELECT 
                COUNT(*) as total_obs,
                COUNT(DISTINCT window_end_date) as n_windows,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM expanding_window_residuals
            WHERE index_id = ?
                AND tenor = ?
        """, [index_id, tenor]).df()
        
        if results['total_obs'].iloc[0] == 0:
            raise ValueError(f"No expanding window residuals found for {index_id} {tenor}")
            
        logger.info(f"""
        Verified residual storage:
        Total observations: {results['total_obs'].iloc[0]}
        Number of windows: {results['n_windows'].iloc[0]}
        Date range: {results['start_date'].iloc[0]} to {results['end_date'].iloc[0]}
        """)
        
    except Exception as e:
        logger.error(f"Error verifying residual storage: {str(e)}")
        raise

def verify_temporal_consistency(
    conn,
    index_id: str,
    tenor: str,
    tolerance_days: int = 5
):
    """Verify no look-ahead bias in error correction model"""
    try:
        # Check temporal alignment
        results = conn.execute("""
            WITH ec_dates AS (
                SELECT 
                    date,
                    error_correction,
                    LAG(date) OVER (ORDER BY date) as prev_date
                FROM error_correction_results
                WHERE index_id = ?
                    AND tenor = ?
            )
            SELECT 
                r.date,
                r.window_end_date,
                ec.error_correction,
                ec.prev_date
            FROM expanding_window_residuals r
            LEFT JOIN ec_dates ec ON r.residual = ec.error_correction
            WHERE r.index_id = ?
                AND r.tenor = ?
                AND r.window_end_date >= ec.date
        """, [index_id, tenor, index_id, tenor]).df()
        
        if len(results) > 0:
            raise ValueError(f"""
            Found {len(results)} instances of look-ahead bias:
            First violation: Window end {results['window_end_date'].iloc[0]}
            used for date {results['date'].iloc[0]}
            """)
            
        logger.info("Verified temporal consistency: No look-ahead bias found")
        
    except Exception as e:
        logger.error(f"Error verifying temporal consistency: {str(e)}")
        raise 
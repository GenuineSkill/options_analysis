from pathlib import Path
import duckdb
import pandas as pd
from typing import Dict, List
import numpy as np
import json
import logging

class DatabaseManager:
    def __init__(self, db_path: Path):
        """Initialize database connection and create tables if needed"""
        self.logger = logging.getLogger('db_manager')
        self.db_path = db_path
        
        try:
            # Try to connect with exclusive access
            self.conn = duckdb.connect(str(db_path), read_only=False)
            self._initialize_tables()
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            # Try to identify blocking process
            self.logger.error("If database is locked, try running: Stop-Process -Id <PID>")
            raise
            
    def _initialize_tables(self):
        """Create tables if they don't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS garch_params (
                date DATE,
                model_type VARCHAR,
                parameters JSON,
                convergence_status BOOLEAN,
                log_likelihood DOUBLE,
                PRIMARY KEY (date, model_type)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS volatility_forecasts (
                estimation_date DATE,
                model_type VARCHAR,
                horizon_date DATE,
                forecast_value DOUBLE,
                PRIMARY KEY (estimation_date, model_type, horizon_date)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_stats (
                date DATE,
                gev DOUBLE,
                evoev DOUBLE,
                dev DOUBLE,
                kev DOUBLE,
                sevts DOUBLE,
                n_models INTEGER,
                PRIMARY KEY (date)
            )
        """)
        
    def save_garch_results(self, date: pd.Timestamp, model_type: str, 
                          params: Dict, converged: bool, loglik: float):
        """Save GARCH model parameters"""
        self.conn.execute("""
            INSERT INTO garch_params (date, model_type, parameters, convergence_status, log_likelihood)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (date, model_type) DO UPDATE SET
                parameters = EXCLUDED.parameters,
                convergence_status = EXCLUDED.convergence_status,
                log_likelihood = EXCLUDED.log_likelihood
        """, (date, model_type, json.dumps(params), converged, loglik))
        
    def save_forecasts(self, estimation_date: pd.Timestamp, model_type: str,
                      forecasts: np.ndarray):
        """Save volatility forecasts"""
        # Create dates for the forecast horizon (252 trading days)
        horizon_dates = pd.date_range(
            start=estimation_date,
            periods=len(forecasts),
            freq='B'  # Business days
        )
        
        # Create DataFrame from numpy array
        df = pd.DataFrame({
            'estimation_date': [estimation_date] * len(forecasts),
            'model_type': [model_type] * len(forecasts),
            'horizon_date': horizon_dates,
            'forecast_value': forecasts.astype(float)  # Ensure float type
        })
        
        # Save to database
        self.conn.execute("""
            INSERT INTO volatility_forecasts 
            SELECT * FROM df
            ON CONFLICT (estimation_date, model_type, horizon_date) 
            DO UPDATE SET forecast_value = EXCLUDED.forecast_value
        """)
        
    def save_ensemble_stats(self, date: pd.Timestamp, stats: Dict):
        """Save ensemble statistics"""
        self.conn.execute("""
            INSERT INTO ensemble_stats 
            (date, gev, evoev, dev, kev, sevts, n_models)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (date) DO UPDATE SET
                gev = EXCLUDED.gev,
                evoev = EXCLUDED.evoev,
                dev = EXCLUDED.dev,
                kev = EXCLUDED.kev,
                sevts = EXCLUDED.sevts,
                n_models = EXCLUDED.n_models
        """, (date, stats['gev'], stats['evoev'], stats['dev'], 
              stats['kev'], stats['sevts'], stats['n_models']))
        
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
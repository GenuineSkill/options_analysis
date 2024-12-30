import logging
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from models import ForecastWindow, GARCHResult
import os

logger = logging.getLogger(__name__)

class GARCHDatabase:
    def __init__(self, db_path: str):
        """Initialize database connection"""
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Initialize connection
        self.conn = duckdb.connect(db_path)
        
        # Create tables if they don't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_windows (
                window_id INTEGER PRIMARY KEY,
                index_id VARCHAR,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                returns BLOB
            );
            
            CREATE TABLE IF NOT EXISTS garch_results (
                result_id INTEGER PRIMARY KEY,
                window_id INTEGER,
                model_type VARCHAR,
                distribution VARCHAR,
                parameters JSON,
                forecast_path BLOB,
                volatility_path BLOB,
                FOREIGN KEY (window_id) REFERENCES forecast_windows (window_id)
            );
            
            CREATE TABLE IF NOT EXISTS ensemble_stats (
                stat_id INTEGER PRIMARY KEY,
                window_id INTEGER,
                horizon VARCHAR,
                horizon_days INTEGER,
                gev DOUBLE,
                evoev DOUBLE,
                dev DOUBLE,
                kev DOUBLE,
                sevts DOUBLE,
                n_models INTEGER,
                FOREIGN KEY (window_id) REFERENCES forecast_windows (window_id)
            );
        """)
        
        self.logger.info(f"Initialized database at {db_path}")

    def _connect(self):
        """Establish database connection"""
        try:
            if self.conn is None:
                self.conn = duckdb.connect(str(self.db_path))
                self._initialize_tables()
            else:
                # Test connection by executing a simple query
                try:
                    self.conn.execute("SELECT 1").fetchone()
                except:
                    self.conn = duckdb.connect(str(self.db_path))
                    self._initialize_tables()
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}")
            raise
    
    def _initialize_tables(self):
        """Initialize database tables"""
        try:
            self.conn.execute("""
                -- Create sequence for window IDs
                CREATE SEQUENCE IF NOT EXISTS window_id_seq;
                
                -- Create forecast windows table
                CREATE TABLE forecast_windows (
                    window_id INTEGER PRIMARY KEY DEFAULT nextval('window_id_seq'),
                    index_id TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    returns BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create GARCH results table
                CREATE TABLE garch_results (
                    window_id INTEGER,
                    model_type TEXT NOT NULL CHECK (model_type IN ('garch', 'egarch', 'gjrgarch')),
                    distribution TEXT NOT NULL CHECK (distribution IN ('normal', 'studentst')),
                    parameters TEXT NOT NULL,
                    forecast_path BLOB NOT NULL,
                    volatility_path BLOB NOT NULL,
                    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
                );
                
                -- Create ensemble stats table
                CREATE TABLE ensemble_stats (
                    window_id INTEGER,
                    horizon TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    gev DOUBLE,
                    evoev DOUBLE,
                    dev DOUBLE,
                    kev DOUBLE,
                    sevts DOUBLE,
                    n_models INTEGER,
                    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
                );
                
                -- Create indices
                CREATE INDEX idx_window_date ON forecast_windows(end_date);
                CREATE INDEX idx_stats_horizon ON ensemble_stats(horizon);
            """)
            
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {str(e)}")
            raise

    def store_forecast_window(self, window: ForecastWindow, index_id: str) -> None:
        """Store forecast window in database
        
        Args:
            window: ForecastWindow object containing the results
            index_id: Identifier for the market index (e.g., 'SPX')
        """
        try:
            # Get next window ID
            window_id = self.get_next_window_id()
            
            # Store window data
            self.conn.execute("""
                INSERT INTO forecast_windows (
                    window_id, index_id, start_date, end_date, returns
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                window_id,
                index_id,
                pd.Timestamp(window.start_date),
                pd.Timestamp(window.end_date),
                window.returns.tobytes()
            ))
            
            # Store GARCH results if available
            if window.garch_results:
                for result in window.garch_results:
                    self.conn.execute("""
                        INSERT INTO garch_results (
                            window_id, model_type, distribution, parameters,
                            forecast_path, volatility_path
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        window_id,
                        result.model_type,
                        result.distribution,
                        json.dumps(result.params),
                        result.forecast_path.tobytes(),
                        result.volatility_path.tobytes()
                    ))
            
            # Store ensemble stats if available
            if window.ensemble_stats:
                for horizon, stats in window.ensemble_stats.items():
                    # Extract horizon days from string (e.g., 'T21' -> 21)
                    try:
                        horizon_days = int(horizon.replace('T', ''))
                    except ValueError:
                        self.logger.warning(f"Invalid horizon format: {horizon}, skipping")
                        continue
                        
                    self.conn.execute("""
                        INSERT INTO ensemble_stats (
                            window_id, horizon, horizon_days,
                            gev, evoev, dev, kev, sevts, n_models
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        window_id,
                        horizon,
                        horizon_days,
                        float(stats['gev']),
                        float(stats['evoev']),
                        float(stats['dev']),
                        float(stats['kev']),
                        float(stats['sevts']),
                        len(window.garch_results) if window.garch_results else 0
                    ))
            
            self.conn.commit()
            self.logger.info(f"Stored forecast window for {index_id} at {window.end_date}")
            
        except Exception as e:
            self.logger.error(f"Error storing forecast window: {str(e)}")
            self.conn.rollback()
            raise

    def get_ensemble_stats_series(self,
                                index_id: str,
                                horizon: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get time series of ensemble statistics for a specific horizon"""
        query = f"""
            SELECT
                w.end_date::DATE as date,
                e.gev as "GEV",
                e.evoev as "EVOEV",
                e.dev as "DEV",
                e.kev as "KEV",
                e.sevts as "SEVTS"
            FROM ensemble_stats e
            JOIN forecast_windows w ON e.window_id = w.window_id
            WHERE w.index_id = ? AND e.horizon = ?
        """
        params = [index_id, horizon]

        if start_date:
            query += " AND w.end_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND w.end_date <= ?"
            params.append(end_date)

        query += " ORDER BY w.end_date"
        
        results = self.conn.execute(query, params).fetchall()
        return pd.DataFrame(results, columns=['date', 'GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS']).set_index('date')

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

    def create_expanding_windows(self,
                              data: pd.DataFrame,
                              min_train_size: int = 3000,
                              test_size: int = 504,
                              step_size: int = 21
                              ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create expanding windows of training and test data"""
        # Ensure data is sorted by date and has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index(pd.to_datetime(data['date']))
            else:
                raise ValueError("Data must have either a datetime index or a 'date' column")
        
        data = data.sort_index()
        
        if len(data) < (min_train_size + test_size):
            raise ValueError(
                f"Insufficient data: {len(data)} rows, "
                f"need at least {min_train_size + test_size}"
            )
        
        windows = []
        start_idx = 0
        
        while (start_idx + min_train_size + test_size) <= len(data):
            # Get date-based slices
            train_end = data.index[start_idx + min_train_size - 1]
            test_end = data.index[start_idx + min_train_size + test_size - 1]
            
            train = data.loc[:train_end]
            test = data.loc[train_end:test_end]
            
            # Validate window dates
            if pd.Timestamp(train.index[0]).year < 1980 or pd.Timestamp(test.index[-1]).year < 1980:
                self.logger.error(f"Invalid dates detected: train={train.index[0]}, test={test.index[-1]}")
                continue
            
            windows.append((train, test))
            start_idx += step_size
        
        return windows

    def get_next_window_id(self) -> int:
        """Get next available window ID"""
        result = self.conn.execute("""
            SELECT COALESCE(MAX(window_id), 0) + 1
            FROM forecast_windows
        """).fetchone()
        return result[0]
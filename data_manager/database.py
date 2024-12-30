import logging
from pathlib import Path
from typing import Union, Optional, Dict, List
import json
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class GARCHDatabase:
    def __init__(self, db_path: Union[str, Path]):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        
        # If database exists, delete it to start fresh
        if self.db_path.exists():
            self.db_path.unlink()
        
        # Create parent directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to new database
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_tables()

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

    def store_forecast_window(self, window, index_id: str) -> int:
        """Store forecast window with mean forecast paths"""
        try:
            # Define horizons dictionary
            horizons = {
                '1M': 21, '2M': 42, '3M': 63, 
                '6M': 126, '12M': 252
            }
            
            # Store basic window info
            result = self.conn.execute("""
                INSERT INTO forecast_windows (start_date, end_date, index_id, returns)
                VALUES (?, ?, ?, ?)
                RETURNING window_id
            """, (
                window.start_date.strftime('%Y-%m-%d %H:%M:%S'),
                window.end_date.strftime('%Y-%m-%d %H:%M:%S'),
                str(index_id),
                window.returns.astype(np.float64).tobytes()
            )).fetchone()
            
            window_id = int(result[0])
            
            # Store GARCH results with mean forecast paths
            for result in window.garch_results:
                self.conn.execute("""
                    INSERT INTO garch_results (
                        window_id, model_type, distribution, parameters,
                        forecast_path, volatility_path
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    window_id,
                    str(result.model_type),
                    str(result.distribution),
                    json.dumps(result.params),
                    result.forecast_path.astype(np.float64).tobytes(),
                    result.volatility_path.astype(np.float64).tobytes()
                ))
            
            # Store ensemble stats for each horizon
            for horizon, stats in window.ensemble_stats.items():
                self.conn.execute("""
                    INSERT INTO ensemble_stats (
                        window_id, horizon, horizon_days,
                        gev, evoev, dev, kev, sevts, n_models
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    window_id,
                    str(horizon),
                    int(horizons[horizon]),  # Now horizons is defined
                    float(stats['GEV']),
                    float(stats['EVOEV']),
                    float(stats['DEV']),
                    float(stats['KEV']),
                    float(stats['SEVTS']),
                    int(stats['n_models'])
                ))
            
            self.conn.commit()
            return window_id
                
        except Exception as e:
            logger.error(f"Error storing forecast window: {str(e)}")
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
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
            # Create everything in a single transaction in the correct order
            self.conn.execute("""
                -- Create sequence first
                CREATE SEQUENCE IF NOT EXISTS window_id_seq;
                
                -- Create main table with sequence-based default
                CREATE TABLE forecast_windows (
                    window_id INTEGER PRIMARY KEY DEFAULT nextval('window_id_seq'),
                    index_id TEXT NOT NULL,
                    start_date TIMESTAMP NOT NULL,
                    end_date TIMESTAMP NOT NULL,
                    returns BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Create garch_results table
                CREATE TABLE garch_results (
                    window_id INTEGER,
                    model_type TEXT NOT NULL,
                    distribution TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON string
                    forecasts_annualized DOUBLE NOT NULL,
                    volatility_annualized DOUBLE NOT NULL,
                    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
                );
                
                -- Create dependent table with foreign key
                CREATE TABLE ensemble_stats (
                    window_id INTEGER,
                    gev DOUBLE,
                    evoev DOUBLE,
                    dev DOUBLE,
                    kev DOUBLE,
                    sevts DOUBLE,
                    n_models INTEGER,
                    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id)
                );
            """)
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
            if hasattr(self, 'conn'):
                self.conn.close()
            raise

    def store_forecast_window(self, window, index_id: str) -> int:
        """Store forecast window and associated results"""
        try:
            # Validate data
            if window.end_date <= window.start_date:
                raise ValueError("End date must be after start date")
            if len(window.returns) == 0:
                raise ValueError("Returns array cannot be empty")

            # Store window and get ID (removed window_id from INSERT fields)
            result = self.conn.execute("""
                INSERT INTO forecast_windows (start_date, end_date, index_id, returns)
                VALUES (?, ?, ?, ?)
                RETURNING window_id
            """, (
                window.start_date,
                window.end_date,
                index_id,
                window.returns.tobytes()
            )).fetchone()
            
            window_id = result[0]
            
            # Store GARCH results
            for result in window.garch_results:
                # Convert scalar forecasts to numpy arrays if needed
                forecasts = np.array([result.forecasts_annualized]) if np.isscalar(result.forecasts_annualized) else result.forecasts_annualized
                volatility = np.array([result.volatility_annualized]) if np.isscalar(result.volatility_annualized) else result.volatility_annualized
                
                self.conn.execute("""
                    INSERT INTO garch_results (
                        window_id, model_type, distribution, parameters,
                        forecasts_annualized, volatility_annualized
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    window_id,
                    result.model_type,
                    result.distribution,
                    json.dumps(result.params),
                    float(np.mean(forecasts)),  # Store mean value
                    float(np.mean(volatility))  # Store mean value
                ))
            
            # Store ensemble stats
            self.conn.execute("""
                INSERT INTO ensemble_stats (
                    window_id, gev, evoev, dev, kev, sevts, n_models
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                window_id,
                window.ensemble_stats['GEV'],
                window.ensemble_stats['EVOEV'],
                window.ensemble_stats['DEV'],
                window.ensemble_stats['KEV'],
                window.ensemble_stats['SEVTS'],
                window.ensemble_stats.get('n_models', len(window.garch_results))
            ))
            
            self.conn.commit()
            return window_id
                
        except Exception as e:
            logger.error(f"Error storing forecast window: {str(e)}")
            raise

    def get_latest_window(self, index_id: str) -> Optional[Dict]:
        """Get most recent forecast window for an index"""
        result = self.conn.execute("""
            SELECT window_id, start_date::DATE, end_date::DATE, returns
            FROM forecast_windows
            WHERE index_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, [index_id]).fetchone()
        
        if result:
            return {
                'window_id': result[0],
                'start_date': result[1],
                'end_date': result[2],
                'returns': np.frombuffer(result[3])
            }
        return None

    def get_ensemble_stats_series(self,
                                index_id: str,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get time series of ensemble statistics"""
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
            WHERE w.index_id = '{index_id}'
        """

        if start_date:
            query += f" AND w.end_date >= DATE '{str(start_date)[:10]}'"
        if end_date:
            query += f" AND w.end_date <= DATE '{str(end_date)[:10]}'"

        query += " ORDER BY w.end_date"
        
        results = self.conn.execute(query).fetchall()
        return pd.DataFrame(results, columns=['date', 'GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS']).set_index('date')

    def get_model_forecasts(self,
                          window_id: int,
                          model_type: Optional[str] = None,
                          distribution: Optional[str] = None) -> List[np.ndarray]:
        """Get forecasts for specific models"""
        query = """
            SELECT forecasts_annualized
            FROM garch_results
            WHERE window_id = ?
        """
        params = [window_id]
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        if distribution:
            query += " AND distribution = ?"
            params.append(distribution)
            
        results = self.conn.execute(query, params).fetchall()
        return [np.frombuffer(r[0]) for r in results]

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

    def is_closed(self) -> bool:
        """Check if database connection is closed"""
        try:
            self.conn.execute("SELECT 1")
            return False
        except:
            return True
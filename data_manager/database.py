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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._initialize_tables()

    def _initialize_tables(self):
        """Create database tables if they don't exist"""
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS window_id_seq;
            
            CREATE TABLE IF NOT EXISTS garch_windows (
                window_id BIGINT PRIMARY KEY DEFAULT nextval('window_id_seq'),
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                index_id VARCHAR NOT NULL,
                returns BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE SEQUENCE IF NOT EXISTS result_id_seq;
            
            CREATE TABLE IF NOT EXISTS garch_results (
                result_id BIGINT PRIMARY KEY DEFAULT nextval('result_id_seq'),
                window_id BIGINT NOT NULL,
                model_type VARCHAR NOT NULL,
                distribution VARCHAR NOT NULL,
                parameters JSON NOT NULL,
                forecasts_annualized BLOB NOT NULL,
                volatility_annualized BLOB NOT NULL,
                FOREIGN KEY(window_id) REFERENCES garch_windows(window_id)
            );
            
            CREATE SEQUENCE IF NOT EXISTS stat_id_seq;
            
            CREATE TABLE IF NOT EXISTS ensemble_stats (
                stat_id BIGINT PRIMARY KEY DEFAULT nextval('stat_id_seq'),
                window_id BIGINT NOT NULL,
                gev DOUBLE PRECISION NOT NULL,
                evoev DOUBLE PRECISION NOT NULL,
                dev DOUBLE PRECISION NOT NULL,
                kev DOUBLE PRECISION NOT NULL,
                sevts DOUBLE PRECISION NOT NULL,
                FOREIGN KEY(window_id) REFERENCES garch_windows(window_id)
            );
        """)

    def store_forecast_window(self, window, index_id: str) -> int:
        """Store forecast window and associated results"""
        try:
            # Validate data
            if window.end_date <= window.start_date:
                raise ValueError("End date must be after start date")
            if len(window.returns) == 0:
                raise ValueError("Returns array cannot be empty")

            # Store window and get ID
            result = self.conn.execute("""
                INSERT INTO garch_windows (start_date, end_date, index_id, returns)
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
                    result.forecasts_annualized.tobytes(),
                    result.volatility_annualized.tobytes()
                ))
            
            # Store ensemble stats
            self.conn.execute("""
                INSERT INTO ensemble_stats (
                    window_id, gev, evoev, dev, kev, sevts
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                window_id,
                window.ensemble_stats['GEV'],
                window.ensemble_stats['EVOEV'],
                window.ensemble_stats['DEV'],
                window.ensemble_stats['KEV'],
                window.ensemble_stats['SEVTS']
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
            FROM garch_windows
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
            JOIN garch_windows w ON e.window_id = w.window_id
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
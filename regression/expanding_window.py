"""Expanding window regression analysis"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ExpandingWindowRegression:
    def __init__(self, conn, min_window_size: int = 252):
        """
        Initialize expanding window regression
        
        Parameters:
        - conn: DuckDB connection
        - min_window_size: Minimum observations for first window (default: 1 year)
        """
        self.conn = conn
        self.min_window_size = min_window_size
        self.setup_tables()
        
    def setup_tables(self):
        """Create tables for expanding window results"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS expanding_window_regressions (
                    window_end_date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    window_start_date DATE,
                    n_observations INTEGER,
                    r_squared DECIMAL,
                    adj_r_squared DECIMAL,
                    PRIMARY KEY (window_end_date, index_id, tenor)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS expanding_window_coefficients (
                    window_end_date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    variable VARCHAR,
                    coefficient DECIMAL,
                    t_stat DECIMAL,
                    p_value DECIMAL,
                    PRIMARY KEY (window_end_date, index_id, tenor, variable)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS expanding_window_residuals (
                    date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    window_end_date DATE,
                    residual DECIMAL,
                    standardized_residual DECIMAL,
                    iv_actual DECIMAL,
                    iv_fitted DECIMAL,
                    PRIMARY KEY (date, index_id, tenor, window_end_date)
                )
            """)
            
        except Exception as e:
            logger.error(f"Error setting up expanding window tables: {str(e)}")
            raise
            
    def get_estimation_windows(self, 
                             index_id: str, 
                             tenor: str,
                             step_size: int = 21) -> List[Dict[str, Any]]:
        """
        Generate expanding windows for estimation
        
        Parameters:
        - index_id: Market identifier
        - tenor: Option expiry tenor
        - step_size: Days between windows (default: 21 for monthly)
        """
        try:
            # Get all available dates with complete data
            dates = self.conn.execute("""
                SELECT DISTINCT r.date
                FROM regression_residuals r
                JOIN forecast_windows fw ON r.window_id = fw.window_id
                WHERE r.index_id = ? 
                    AND r.tenor = ?
                    AND r.residual IS NOT NULL
                ORDER BY r.date
            """, [index_id, tenor]).df()
            
            if len(dates) < self.min_window_size:
                raise ValueError(f"Insufficient data for {index_id} {tenor}")
                
            windows = []
            start_idx = 0
            
            # Create expanding windows
            for end_idx in range(self.min_window_size, len(dates), step_size):
                windows.append({
                    'start_date': dates['date'].iloc[start_idx],
                    'end_date': dates['date'].iloc[end_idx],
                    'n_observations': end_idx - start_idx + 1
                })
                
            logger.info(f"""
            Created {len(windows)} expanding windows for {index_id} {tenor}:
            First window: {windows[0]['start_date']} to {windows[0]['end_date']}
            Last window: {windows[-1]['start_date']} to {windows[-1]['end_date']}
            """)
            
            return windows
            
        except Exception as e:
            logger.error(f"Error generating windows: {str(e)}")
            raise
            
    def estimate_window(self, 
                       index_id: str, 
                       tenor: str,
                       start_date: datetime,
                       end_date: datetime) -> Dict[str, Any]:
        """Estimate regression for a single window"""
        try:
            # Get data for window
            data = self.conn.execute("""
                SELECT 
                    r.date,
                    r.iv_actual as IV,
                    es.gev as GEV,
                    es.evoev as EVOEV,
                    es.dev as DEV,
                    es.kev as KEV,
                    es.sevts as SEVTS,
                    CASE 
                        WHEN es.horizon_days = ? THEN 1 
                        ELSE 0 
                    END as NTRADE
                FROM regression_residuals r
                JOIN forecast_windows fw ON r.window_id = fw.window_id
                JOIN ensemble_stats es ON fw.window_id = es.window_id
                WHERE r.index_id = ?
                    AND r.tenor = ?
                    AND r.date BETWEEN ? AND ?
                ORDER BY r.date
            """, [
                int(tenor.replace('M', '')) * 21,  # Convert tenor to days
                index_id, 
                tenor,
                start_date,
                end_date
            ]).df()
            
            # Run regression
            Y = data['IV']
            X = data[['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS', 'NTRADE']]
            X = sm.add_constant(X)
            
            model = sm.OLS(Y, X)
            results = model.fit()
            
            return {
                'results': results,
                'data': data,
                'window_end': end_date,
                'n_obs': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error estimating window: {str(e)}")
            raise
            
    def run_expanding_windows(self, 
                            index_id: str, 
                            tenor: str,
                            step_size: int = 21):
        """Run expanding window regression analysis"""
        try:
            # Get windows
            windows = self.get_estimation_windows(index_id, tenor, step_size)
            
            for window in windows:
                # Estimate window
                results = self.estimate_window(
                    index_id=index_id,
                    tenor=tenor,
                    start_date=window['start_date'],
                    end_date=window['end_date']
                )
                
                # Store regression results
                self.store_window_results(
                    index_id=index_id,
                    tenor=tenor,
                    window=window,
                    results=results
                )
                
            logger.info(f"Completed expanding window analysis for {index_id} {tenor}")
            
        except Exception as e:
            logger.error(f"Error in expanding window analysis: {str(e)}")
            raise
            
    def store_window_results(self,
                            index_id: str,
                            tenor: str,
                            window: Dict[str, Any],
                            results: Dict[str, Any]):
        """Store regression results for a single window"""
        try:
            # Store regression metadata
            self.conn.execute("""
                INSERT OR REPLACE INTO expanding_window_regressions
                (window_end_date, index_id, tenor, window_start_date, 
                 n_observations, r_squared, adj_r_squared)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                window['end_date'],
                index_id,
                tenor,
                window['start_date'],
                results['n_obs'],
                results['results'].rsquared,
                results['results'].rsquared_adj
            ])
            
            # Store coefficients
            coef_data = []
            for var, coef in results['results'].params.items():
                coef_data.append([
                    window['end_date'],
                    index_id,
                    tenor,
                    var,
                    coef,
                    results['results'].tvalues[var],
                    results['results'].pvalues[var]
                ])
            
            for row in coef_data:
                self.conn.execute("""
                    INSERT OR REPLACE INTO expanding_window_coefficients
                    (window_end_date, index_id, tenor, variable, 
                     coefficient, t_stat, p_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, row)
            
            # Store residuals
            residuals_df = pd.DataFrame({
                'date': results['data']['date'],
                'index_id': index_id,
                'tenor': tenor,
                'window_end_date': window['end_date'],
                'residual': results['results'].resid,
                'standardized_residual': results['results'].resid / results['results'].resid.std(),
                'iv_actual': results['data']['IV'],
                'iv_fitted': results['results'].fittedvalues
            })
            
            for _, row in residuals_df.iterrows():
                self.conn.execute("""
                    INSERT OR REPLACE INTO expanding_window_residuals
                    (date, index_id, tenor, window_end_date, residual,
                     standardized_residual, iv_actual, iv_fitted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row['date'], row['index_id'], row['tenor'],
                    row['window_end_date'], row['residual'],
                    row['standardized_residual'], row['iv_actual'],
                    row['iv_fitted']
                ])
            
            logger.info(f"""
            Stored results for window ending {window['end_date']}:
            Index: {index_id}
            Tenor: {tenor}
            Observations: {results['n_obs']}
            R-squared: {results['results'].rsquared:.4f}
            """)
            
        except Exception as e:
            logger.error(f"Error storing window results: {str(e)}")
            raise
            
    def get_window_results(self,
                          index_id: str,
                          tenor: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Retrieve expanding window results
        
        Returns dictionary with:
        - regressions: Window-level statistics
        - coefficients: Time series of coefficient values
        - residuals: All residuals for all windows
        """
        try:
            # Base date conditions
            date_cond = ""
            params = [index_id, tenor]
            
            if start_date:
                date_cond += " AND window_end_date >= ?"
                params.append(start_date)
            if end_date:
                date_cond += " AND window_end_date <= ?"
                params.append(end_date)
                
            # Get regression results
            regressions = self.conn.execute(f"""
                SELECT *
                FROM expanding_window_regressions
                WHERE index_id = ?
                    AND tenor = ?
                    {date_cond}
                ORDER BY window_end_date
            """, params).df()
            
            # Get coefficients
            coefficients = self.conn.execute(f"""
                SELECT *
                FROM expanding_window_coefficients
                WHERE index_id = ?
                    AND tenor = ?
                    {date_cond}
                ORDER BY window_end_date, variable
            """, params).df()
            
            # Get residuals
            residuals = self.conn.execute(f"""
                SELECT *
                FROM expanding_window_residuals
                WHERE index_id = ?
                    AND tenor = ?
                    {date_cond}
                ORDER BY date, window_end_date
            """, params).df()
            
            logger.info(f"""
            Retrieved expanding window results for {index_id} {tenor}:
            Windows: {len(regressions)}
            Coefficient series: {len(coefficients)}
            Residual observations: {len(residuals)}
            """)
            
            return {
                'regressions': regressions,
                'coefficients': coefficients,
                'residuals': residuals
            }
            
        except Exception as e:
            logger.error(f"Error retrieving window results: {str(e)}")
            raise
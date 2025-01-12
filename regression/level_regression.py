# regression/level_regression.py
"""Implements level regression analysis for implied volatilities"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import statsmodels.api as sm

logger = logging.getLogger(__name__)

class LevelRegression:
    def __init__(self, conn):
        """Initialize with database connection"""
        self.conn = conn
        self.setup_tables()
        
    def setup_tables(self):
        """Create tables for storing regression results and residuals"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS regression_residuals (
                    date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    residual DECIMAL,
                    standardized_residual DECIMAL,
                    window_id INTEGER,
                    iv_actual DECIMAL,      -- Store actual IV for validation
                    iv_fitted DECIMAL,      -- Store fitted values
                    FOREIGN KEY (window_id) REFERENCES forecast_windows(window_id),
                    PRIMARY KEY (date, index_id, tenor)
                )
            """)
            
            # Index for efficient querying
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_residuals_lookup 
                ON regression_residuals(index_id, tenor, date)
            """)
            
        except Exception as e:
            logger.error(f"Error setting up regression tables: {str(e)}")
            raise
            
    def store_regression_results(self, 
                               data: pd.DataFrame, 
                               results: sm.regression.linear_model.RegressionResultsWrapper,
                               index_id: str,
                               tenor: str):
        """Store regression residuals and fitted values"""
        try:
            # Calculate residuals and standardized residuals
            residuals_df = pd.DataFrame({
                'date': data['date'],
                'index_id': index_id,
                'tenor': tenor,
                'residual': results.resid,
                'standardized_residual': results.resid / results.resid.std(),
                'iv_actual': data['IV'],
                'iv_fitted': results.fittedvalues
            })
            
            # Get window_ids for each date
            window_ids = self.conn.execute("""
                SELECT end_date, window_id 
                FROM forecast_windows 
                WHERE index_id = ?
            """, [index_id]).df()
            
            # Merge window_ids
            residuals_df = residuals_df.merge(
                window_ids,
                left_on='date',
                right_on='end_date',
                how='left'
            )
            
            # Store in database - row by row to match parameter count
            for _, row in residuals_df.iterrows():
                self.conn.execute("""
                    INSERT OR REPLACE INTO regression_residuals
                    (date, index_id, tenor, residual, standardized_residual, 
                     window_id, iv_actual, iv_fitted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [row['date'], row['index_id'], row['tenor'], 
                     row['residual'], row['standardized_residual'],
                     row['window_id'], row['iv_actual'], row['iv_fitted']])
            
            logger.info(f"""
            Stored regression results for {index_id} {tenor}:
            Dates: {residuals_df['date'].min()} to {residuals_df['date'].max()}
            Observations: {len(residuals_df)}
            Mean residual: {residuals_df['residual'].mean():.4f}
            Std residual: {residuals_df['residual'].std():.4f}
            """)
            
        except Exception as e:
            logger.error(f"Error storing regression results: {str(e)}")
            raise
            
    def prepare_data(self, index_id: str, tenor: str) -> pd.DataFrame:
        """
        Prepare data for regression analysis by joining IV data with ensemble statistics
        """
        try:
            # First, verify index_id exists in both tables
            index_check = self.conn.execute("""
                SELECT 
                    COUNT(DISTINCT iv.index_id) as iv_indices,
                    (
                        SELECT COUNT(DISTINCT fw.index_id) 
                        FROM ensemble_stats es
                        JOIN forecast_windows fw ON es.window_id = fw.window_id
                        WHERE fw.index_id = ?
                    ) as ensemble_indices
                FROM implied_vol_aligned iv
                WHERE iv.index_id = ?
            """, [index_id, index_id]).df()
            
            logger.info(f"\nIndex validation for {index_id}:")
            logger.info(index_check)
            
            if index_check['ensemble_indices'].iloc[0] == 0:
                raise ValueError(f"No ensemble statistics found for index {index_id}")
            
            # Rest of data checks...
            data_check = self.conn.execute("""
                WITH iv_stats AS (
                    SELECT 
                        COUNT(*) as iv_count,
                        COUNT(DISTINCT date) as iv_dates
                    FROM implied_vol_aligned
                    WHERE index_id = ?
                ),
                es_stats AS (
                    SELECT 
                        COUNT(*) as es_count,
                        COUNT(DISTINCT fw.end_date) as es_dates
                    FROM ensemble_stats es
                    JOIN forecast_windows fw ON es.window_id = fw.window_id
                    WHERE fw.index_id = ?
                )
                SELECT 
                    iv_count, iv_dates, es_count, es_dates,
                    (
                        SELECT COUNT(DISTINCT iv.date)
                        FROM implied_vol_aligned iv
                        JOIN forecast_windows fw ON iv.date = fw.end_date
                        JOIN ensemble_stats es ON es.window_id = fw.window_id
                        WHERE iv.index_id = ? AND fw.index_id = ?
                    ) as matching_dates
                FROM iv_stats, es_stats
            """, [index_id, index_id, index_id, index_id]).df()
            
            logger.info(f"\nData coverage for {index_id}:")
            logger.info(data_check)
            
            # Convert tenor to horizon days for matching
            tenor_map = {
                '12M': 'T252',
                '6M': 'T126',
                '3M': 'T63',
                '2M': 'T42',
                '1M': 'T21'
            }
            
            horizon_days_map = {
                'T252': 252,
                'T126': 126,
                'T63': 63,
                'T42': 42,
                'T21': 21
            }
            
            mapped_tenor = tenor_map.get(tenor)
            if not mapped_tenor:
                raise ValueError(f"Invalid tenor: {tenor}. Expected one of: {list(tenor_map.keys())}")
                
            horizon_days = horizon_days_map[mapped_tenor]
            
            # Original query with explicit index_id checks
            query = """
                SELECT 
                    iv.date,
                    iv.implied_vol as IV,
                    es.gev as GEV,
                    es.evoev as EVOEV,
                    es.dev as DEV,
                    es.kev as KEV,
                    es.sevts as SEVTS,
                    CASE 
                        WHEN es.horizon_days = ? THEN 1 
                        ELSE 0 
                    END as NTRADE
                FROM implied_vol_aligned iv
                JOIN forecast_windows fw 
                    ON iv.date = fw.end_date 
                    AND iv.index_id = fw.index_id
                JOIN ensemble_stats es 
                    ON es.window_id = fw.window_id
                WHERE iv.index_id = ?
                    AND fw.index_id = ?
                    AND iv.tenor = ?
                    AND es.horizon_days = ?
                ORDER BY iv.date
            """
            
            data = self.conn.execute(query, [
                horizon_days, 
                index_id, 
                index_id,
                tenor, 
                horizon_days
            ]).df()
            
            logger.info(f"\nPrepared {len(data):,} observations for {index_id} {tenor}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def estimate(self, index_id: str, tenor: str) -> Dict[str, Any]:
        """
        Estimate level regression and store residuals
        """
        try:
            # Get data
            data = self.prepare_data(index_id, tenor)
            if len(data) == 0:
                raise ValueError(f"No data available for {index_id} {tenor}")
                
            # Check for NaN/inf values before regression
            for col in ['IV', 'GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS', 'NTRADE']:
                nan_count = data[col].isna().sum()
                inf_count = np.isinf(data[col]).sum()
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"Found {nan_count} NaN and {inf_count} inf values in {col}")
            
            # Remove rows with any NaN/inf values
            clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
            dropped_rows = len(data) - len(clean_data)
            if dropped_rows > 0:
                logger.warning(f"Dropped {dropped_rows} rows with NaN/inf values")
            
            if len(clean_data) == 0:
                raise ValueError(f"No valid data remaining after cleaning for {index_id} {tenor}")
            
            # Prepare variables
            Y = clean_data['IV']
            X = clean_data[['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS', 'NTRADE']]
            X = sm.add_constant(X)
            
            # Run regression
            model = sm.OLS(Y, X)
            results = model.fit()
            
            # Store residuals
            self.store_regression_results(clean_data, results, index_id, tenor)
            
            # Extract results
            return {
                'coefficients': results.params.to_dict(),
                't_stats': results.tvalues.to_dict(),
                'p_values': results.pvalues.to_dict(),
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'nobs': results.nobs,
                'start_date': data['date'].min(),
                'end_date': data['date'].max()
            }
            
        except Exception as e:
            logger.error(f"Error in regression estimation: {str(e)}")
            raise
            
    def run_all_regressions(self) -> pd.DataFrame:
        """Run regressions for all index/tenor combinations"""
        results = []
        tenors = ['12M', '6M', '3M', '2M', '1M']
        
        # First, get list of indices that have ensemble statistics
        valid_indices = self.conn.execute("""
            SELECT DISTINCT fw.index_id
            FROM ensemble_stats es
            JOIN forecast_windows fw ON es.window_id = fw.window_id
            ORDER BY fw.index_id
        """).df()
        
        logger.info(f"\nFound ensemble statistics for indices: {valid_indices['index_id'].tolist()}")
        
        for index_id in valid_indices['index_id']:
            for tenor in tenors:
                logger.info(f"Running regression for {index_id} {tenor}")
                try:
                    reg_results = self.estimate(index_id, tenor)
                    results.append({
                        'index_id': index_id,
                        'tenor': tenor,
                        **reg_results['coefficients'],
                        **{f't_{k}': v for k, v in reg_results['t_stats'].items()},
                        **{f'p_{k}': v for k, v in reg_results['p_values'].items()},
                        'r_squared': reg_results['r_squared'],
                        'adj_r_squared': reg_results['adj_r_squared'],
                        'nobs': reg_results['nobs'],
                        'start_date': reg_results['start_date'],
                        'end_date': reg_results['end_date']
                    })
                except Exception as e:
                    logger.warning(f"Skipping {index_id} {tenor}: {str(e)}")
                    continue
        
        if not results:
            raise ValueError("No successful regressions completed")
        
        return pd.DataFrame(results)
            
    def validate_residuals(self, index_id: str, tenor: str) -> pd.DataFrame:
        """
        Validate stored residuals for a given index/tenor combination
        Returns validation report DataFrame
        """
        try:
            # Check residuals statistics
            stats = self.conn.execute("""
                WITH residual_stats AS (
                    SELECT 
                        COUNT(*) as n_obs,
                        COUNT(DISTINCT date) as n_dates,
                        AVG(residual) as mean_residual,
                        AVG(standardized_residual) as mean_std_residual,
                        MIN(date) as start_date,
                        MAX(date) as end_date,
                        COUNT(*) FILTER (WHERE residual IS NULL) as null_residuals,
                        COUNT(*) FILTER (WHERE ABS(standardized_residual) > 3) as outliers
                    FROM regression_residuals
                    WHERE index_id = ? AND tenor = ?
                ),
                date_gaps AS (
                    SELECT 
                        AVG(date_diff('day', lag_date, date)) as avg_gap,
                        MAX(date_diff('day', lag_date, date)) as max_gap
                    FROM (
                        SELECT 
                            date,
                            LAG(date) OVER (ORDER BY date) as lag_date
                        FROM regression_residuals
                        WHERE index_id = ? AND tenor = ?
                    ) t
                    WHERE lag_date IS NOT NULL
                )
                SELECT * FROM residual_stats, date_gaps
            """, [index_id, tenor, index_id, tenor]).df()
            
            logger.info(f"\nResidual validation for {index_id} {tenor}:")
            logger.info(stats)
            
            # Check for autocorrelation
            residuals = self.get_residuals(index_id, tenor)
            if len(residuals) > 0:
                acf = np.correlate(residuals['residual'], residuals['residual'], mode='full')
                acf = acf[len(acf)//2:] / len(residuals)  # Normalize
                
                logger.info(f"""
                Autocorrelation check:
                Lag 1: {acf[1]:.4f}
                Lag 2: {acf[2]:.4f}
                Lag 3: {acf[3]:.4f}
                """)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error validating residuals: {str(e)}")
            raise
            
    def get_residuals(self, 
                      index_id: str, 
                      tenor: str, 
                      start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Retrieve residuals for analysis or strategy implementation
        
        Parameters:
        - index_id: Market identifier
        - tenor: Option expiry tenor
        - start_date: Optional start date filter (YYYY-MM-DD)
        - end_date: Optional end date filter (YYYY-MM-DD)
        """
        try:
            query = """
                SELECT 
                    r.date,
                    r.residual,
                    r.standardized_residual,
                    r.iv_actual,
                    r.iv_fitted,
                    fw.start_date as window_start,  -- Include window info
                    fw.end_date as window_end
                FROM regression_residuals r
                LEFT JOIN forecast_windows fw 
                    ON r.window_id = fw.window_id
                WHERE r.index_id = ? 
                    AND r.tenor = ?
            """
            
            params = [index_id, tenor]
            
            if start_date:
                query += " AND r.date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND r.date <= ?"
                params.append(end_date)
                
            query += " ORDER BY r.date"
            
            residuals = self.conn.execute(query, params).df()
            
            logger.info(f"""
            Retrieved residuals for {index_id} {tenor}:
            Date range: {residuals['date'].min()} to {residuals['date'].max()}
            Observations: {len(residuals)}
            Mean residual: {residuals['residual'].mean():.4f}
            Mean std residual: {residuals['standardized_residual'].mean():.4f}
            """)
            
            return residuals
            
        except Exception as e:
            logger.error(f"Error retrieving residuals: {str(e)}")
            raise
            
    def get_trading_signals(self, 
                           index_id: str, 
                           tenor: str, 
                           z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Get trading signals based on standardized residuals
        
        Parameters:
        - index_id: Market identifier
        - tenor: Option expiry tenor
        - z_threshold: Z-score threshold for signals (default 2.0)
        """
        try:
            signals = self.conn.execute("""
                SELECT 
                    date,
                    standardized_residual as z_score,
                    CASE 
                        WHEN standardized_residual <= -? THEN 1    -- Buy signal
                        WHEN standardized_residual >= ? THEN -1    -- Sell signal
                        ELSE 0                                     -- No signal
                    END as signal,
                    iv_actual,
                    iv_fitted
                FROM regression_residuals
                WHERE index_id = ? 
                    AND tenor = ?
                ORDER BY date
            """, [z_threshold, z_threshold, index_id, tenor]).df()
            
            logger.info(f"""
            Generated trading signals for {index_id} {tenor}:
            Total observations: {len(signals)}
            Buy signals: {(signals['signal'] == 1).sum()}
            Sell signals: {(signals['signal'] == -1).sum()}
            No signals: {(signals['signal'] == 0).sum()}
            """)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {str(e)}")
            raise

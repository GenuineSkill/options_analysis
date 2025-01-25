"""Error correction model for implied volatility regression"""

import logging
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorCorrectionModel:
    def __init__(self, conn, dev_mode: bool = True):
        """
        Initialize error correction model
        
        Parameters:
        - conn: DuckDB connection
        - dev_mode: If True, use 21-day steps; if False, use daily steps
        """
        self.conn = conn
        self.dev_mode = dev_mode
        self.step_size = 1 if not dev_mode else 21  # trading days
        
        # Force rebuild and verify windows
        self.setup_forecast_windows()
        if not self.verify_window_sequence():
            raise ValueError("Window sequence verification failed")
        
        logger.info(f"Error correction model initialized (dev_mode={dev_mode})")
        
        self.setup_tables()
        
    def setup_tables(self):
        """Create tables for error correction results"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS error_correction_results (
                    date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    delta_iv DOUBLE,          -- Changed from DECIMAL to DOUBLE
                    delta_gev DOUBLE,         -- Changed from DECIMAL to DOUBLE
                    delta_evoev DOUBLE,       -- Changed from DECIMAL to DOUBLE
                    delta_dev DOUBLE,         -- Changed from DECIMAL to DOUBLE
                    delta_kev DOUBLE,         -- Changed from DECIMAL to DOUBLE
                    delta_sevts DOUBLE,       -- Changed from DECIMAL to DOUBLE
                    error_correction DOUBLE,   -- Changed from DECIMAL to DOUBLE
                    fitted_change DOUBLE,      -- Changed from DECIMAL to DOUBLE
                    residual DOUBLE,          -- Changed from DECIMAL to DOUBLE
                    r_squared DOUBLE,         -- Changed from DECIMAL to DOUBLE
                    PRIMARY KEY (date, index_id, tenor)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS error_correction_coefficients (
                    date DATE,
                    index_id VARCHAR,
                    tenor VARCHAR,
                    variable VARCHAR,
                    coefficient DOUBLE,       -- Changed from DECIMAL to DOUBLE
                    t_stat DOUBLE,           -- Changed from DECIMAL to DOUBLE
                    p_value DOUBLE,          -- Changed from DECIMAL to DOUBLE
                    PRIMARY KEY (date, index_id, tenor, variable)
                )
            """)
            
        except Exception as e:
            logger.error(f"Error setting up error correction tables: {str(e)}")
            raise
            
    def validate_window_sequence(self, data: pd.DataFrame) -> None:
        """
        Validate temporal consistency of expanding windows
        
        Parameters:
        - data: DataFrame with 'date' and 'window_end' columns
        
        Raises:
        - ValueError if any temporal consistency rules are violated
        """
        try:
            violations = []
            
            # Sort by date to ensure proper sequence checking
            sorted_data = data.sort_values('date')
            
            # Check each observation
            for idx, row in sorted_data.iterrows():
                current_date = row['date']
                window_end = row['window_end']
                
                # Rule 1: Window end date cannot be after current date
                if window_end > current_date:
                    violations.append({
                        'type': 'look_ahead',
                        'date': current_date,
                        'window_end': window_end,
                        'gap_days': (window_end - current_date).days
                    })
                
                # Rule 2: Windows should expand or stay same size (not contract)
                if idx > 0:
                    prev_window_end = sorted_data.iloc[idx-1]['window_end']
                    if window_end < prev_window_end:
                        violations.append({
                            'type': 'window_contraction',
                            'date': current_date,
                            'current_end': window_end,
                            'previous_end': prev_window_end
                        })
            
            if violations:
                look_ahead = [v for v in violations if v['type'] == 'look_ahead']
                contractions = [v for v in violations if v['type'] == 'window_contraction']
                
                error_msg = []
                if look_ahead:
                    error_msg.append(f"""
                    Found {len(look_ahead)} instances of look-ahead bias:
                    First violation: Window end {look_ahead[0]['window_end']}
                    used for date {look_ahead[0]['date']}
                    Maximum look-ahead: {max(v['gap_days'] for v in look_ahead)} days
                    """)
                
                if contractions:
                    error_msg.append(f"""
                    Found {len(contractions)} instances of window contraction:
                    First violation: At date {contractions[0]['date']}
                    Window end moved backward from {contractions[0]['previous_end']}
                    to {contractions[0]['current_end']}
                    """)
                    
                raise ValueError("\n".join(error_msg))
                
            logger.info(f"""
            Window sequence validation passed:
            Date range: {sorted_data['date'].min()} to {sorted_data['date'].max()}
            Window range: {sorted_data['window_end'].min()} to {sorted_data['window_end'].max()}
            Observations: {len(sorted_data)}
            """)
            
        except Exception as e:
            logger.error(f"Error verifying temporal consistency: {str(e)}")
            raise

    def prepare_differences(self, index_id: str, tenor: str) -> pd.DataFrame:
        """Calculate first differences and get expanding window residuals"""
        try:
            # Map tenor to horizon FIRST
            tenor_to_horizon = {
                '1M': 'T21',    # ~21 trading days
                '2M': 'T42',    # ~42 trading days
                '3M': 'T63',    # ~63 trading days
                '6M': 'T126',   # ~126 trading days
                '12M': 'T252'   # ~252 trading days
            }
            horizon = tenor_to_horizon.get(tenor)
            if not horizon:
                raise ValueError(f"Invalid tenor: {tenor}")
            
            # First check what horizons actually exist
            horizon_check = self.conn.execute("""
                SELECT DISTINCT horizon, COUNT(*) as count
                FROM ensemble_stats
                GROUP BY horizon
                ORDER BY horizon
            """).df()
            
            logger.info(f"""
            Available horizons in ensemble_stats:
            {horizon_check.to_string()}
            """)
            
            if len(horizon_check) == 0:
                raise ValueError("No ensemble statistics found in database")
            
            # Check ALL dates first, then filtered
            date_check = self.conn.execute("""
                SELECT 
                    MIN(fw.start_date) as min_start,
                    MAX(fw.start_date) as max_start,
                    MIN(fw.end_date) as min_end,
                    MAX(fw.end_date) as max_end,
                    COUNT(*) as count,
                    COUNT(DISTINCT fw.window_id) as distinct_windows
                FROM forecast_windows fw
                WHERE fw.index_id = ?
            """, [index_id]).df()
            
            logger.info(f"""
            Checking forecast_windows for {index_id}:
            Total count: {date_check['count'].iloc[0]}
            Distinct windows: {date_check['distinct_windows'].iloc[0]}
            Start date range: {date_check['min_start'].iloc[0]} to {date_check['max_start'].iloc[0]}
            End date range: {date_check['min_end'].iloc[0]} to {date_check['max_end'].iloc[0]}
            Target range: 2005-01-14 to 2022-07-05
            """)
            
            # Check table structure
            table_info = self.conn.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'forecast_windows'
            """).df()
            
            logger.info(f"""
            Forecast windows table structure:
            {table_info.to_string()}
            """)
            
            # Check sample data
            sample_data = self.conn.execute("""
                SELECT window_id, index_id, start_date, end_date
                FROM forecast_windows
                WHERE index_id = ?
                LIMIT 5
            """, [index_id]).df()
            
            logger.info(f"""
            Sample forecast windows data:
            {sample_data.to_string()}
            """)
            
            # Check ensemble stats for this index using end_date
            ensemble_check = self.conn.execute("""
                SELECT 
                    es.horizon,
                    COUNT(*) as count,
                    MIN(fw.end_date) as min_date,
                    MAX(fw.end_date) as max_date
                FROM ensemble_stats es
                JOIN forecast_windows fw ON es.window_id = fw.window_id
                    AND fw.index_id = ?
                    AND fw.end_date >= '2005-01-14'
                    AND fw.end_date <= '2022-07-05'
                GROUP BY es.horizon
                ORDER BY es.horizon
            """, [index_id]).df()
            
            logger.info(f"""
            Checking ensemble_stats by horizon for {index_id}:
            {ensemble_check.to_string()}
            """)
            
            # Get the filtered dates using end_date and standardize to month-end
            step_check = self.conn.execute("""
                WITH monthly_dates AS (
                    SELECT DISTINCT 
                        DATE_TRUNC('month', fw.end_date) + INTERVAL '1 month' - INTERVAL '1 day' as month_end,
                        fw.end_date
                    FROM forecast_windows fw
                    JOIN ensemble_stats es ON fw.window_id = es.window_id
                    WHERE es.horizon = ?
                        AND fw.index_id = ?
                        AND fw.end_date >= '2005-01-14'
                        AND fw.end_date <= '2022-07-05'
                )
                SELECT end_date as date
                FROM monthly_dates
                ORDER BY end_date
            """, [horizon, index_id]).df()
            
            logger.info(f"""
            Filtered dates for {horizon}:
            Count: {len(step_check)}
            First few dates: {step_check['date'].head().tolist() if not step_check.empty else []}
            Last few dates: {step_check['date'].tail().tolist() if not step_check.empty else []}
            """)
            
            # Verify step size with more tolerance for monthly data
            dates = step_check['date'].tolist()
            if len(dates) < 2:
                raise ValueError(f"Not enough dates to calculate step size for {horizon}")
            
            actual_steps = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            median_step = float(np.median(actual_steps))  # Convert to float explicitly
            
            # Allow more tolerance for monthly steps (21 trading days ≈ 28-31 calendar days)
            if self.dev_mode and not (25 <= median_step <= 33):  # More flexible range
                raise ValueError(f"""
                Step size mismatch:
                Expected ~21 trading days (dev_mode={self.dev_mode})
                Found {median_step:.1f} calendar days in ensemble stats
                Note: Monthly steps can vary from 28-31 calendar days
                """)
            elif not self.dev_mode and median_step > 2:  # Allow some variance for daily steps
                raise ValueError(f"""
                Step size mismatch:
                Expected 1 day (dev_mode={self.dev_mode})
                Found {median_step:.1f} days in ensemble stats
                """)
            
            logger.info(f"""
            Checking ensemble_stats for {horizon}:
            Count: {len(dates)}
            Window range: {dates[0]} to {dates[-1]}
            Step size: {median_step:.1f} days (≈ {median_step/7:.1f} weeks)
            """)
            
            # Diagnostic checks for data availability
            ensemble_check = self.conn.execute("""
                SELECT COUNT(*) as count, MIN(window_id) as min_window, MAX(window_id) as max_window
                FROM ensemble_stats
                WHERE horizon = ?
            """, [horizon]).df()
            
            logger.info(f"""
            Checking ensemble_stats for {horizon}:
            Count: {ensemble_check['count'].iloc[0]}
            Window range: {ensemble_check['min_window'].iloc[0]} to {ensemble_check['max_window'].iloc[0]}
            Step size: {median_step} days
            """)
            
            # Check residuals
            residual_check = self.conn.execute("""
                SELECT COUNT(*) as count,
                       MIN(date) as min_date,
                       MAX(date) as max_date
                FROM expanding_window_residuals
                WHERE index_id = ? AND tenor = ?
            """, [index_id, tenor]).df()
            
            logger.info(f"""
            Checking residuals for {index_id} {tenor}:
            Count: {residual_check['count'].iloc[0]}
            Date range: {residual_check['min_date'].iloc[0]} to {residual_check['max_date'].iloc[0]}
            """)
            
            # Modified query to use contemporaneous windows
            query = """
                WITH contemporaneous_windows AS (
                    -- Select windows that are truly contemporaneous
                    SELECT 
                        r.date,
                        r.iv_actual,
                        r.residual as expanding_residual,
                        fw.window_id,
                        fw.start_date,
                        fw.end_date,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.date 
                            ORDER BY fw.end_date ASC
                        ) as rn
                    FROM expanding_window_residuals r
                    JOIN forecast_windows fw 
                        ON r.date BETWEEN fw.start_date AND fw.end_date
                        AND r.index_id = fw.index_id
                        AND fw.end_date <= r.date + INTERVAL '1 month'  -- Ensure contemporaneous
                    WHERE r.index_id = ?
                        AND r.tenor = ?
                ),
                valid_ensemble_data AS (
                    -- Get ensemble stats with proper temporal alignment
                    SELECT 
                        cw.date,
                        cw.iv_actual,
                        cw.expanding_residual,
                        cw.end_date as window_end,
                        es.gev as GEV,
                        es.evoev as EVOEV,
                        es.dev as DEV,
                        es.kev as KEV,
                        es.sevts as SEVTS,
                        ROW_NUMBER() OVER (ORDER BY cw.date) as row_num
                    FROM contemporaneous_windows cw
                    JOIN ensemble_stats es 
                        ON cw.window_id = es.window_id
                        AND es.horizon = ?
                    WHERE cw.rn = 1
                        AND cw.end_date <= cw.date  -- Strict temporal ordering
                )
                SELECT 
                    d1.date,
                    d1.window_end,
                    d1.iv_actual - d2.iv_actual as delta_iv,
                    d1.GEV - d2.GEV as delta_gev,
                    d1.EVOEV - d2.EVOEV as delta_evoev,
                    d1.DEV - d2.DEV as delta_dev,
                    d1.KEV - d2.KEV as delta_kev,
                    d1.SEVTS - d2.SEVTS as delta_sevts,
                    d2.expanding_residual as error_correction
                FROM valid_ensemble_data d1
                JOIN valid_ensemble_data d2 
                    ON d1.row_num = d2.row_num + ?
                    AND d2.window_end < d1.date  -- Strict inequality for error correction
                ORDER BY d1.date
            """
            
            data = self.conn.execute(query, [
                index_id, 
                tenor,
                horizon,
                self.step_size
            ]).df()
            
            # Validate temporal consistency before proceeding
            self.validate_window_sequence(data)
            
            logger.info(f"""
            Prepared differences for {index_id} {tenor}:
            Observations: {len(data)}
            Date range: {data['date'].min()} to {data['date'].max()}
            Mean IV change: {data['delta_iv'].mean():.6f}
            Std IV change: {data['delta_iv'].std():.6f}
            """)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing differences: {str(e)}")
            raise
            
    def estimate(self,
                index_id: str,
                tenor: str,
                data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Estimate error correction model
        """
        try:
            # Get or prepare data
            if data is None:
                data = self.prepare_differences(index_id, tenor)
                
            # Prepare regression variables
            Y = data['delta_iv']
            X = data[[
                'delta_gev',
                'delta_evoev',
                'delta_dev',
                'delta_kev',
                'delta_sevts',
                'error_correction'
            ]]
            X = sm.add_constant(X)
            
            # Run regression
            model = sm.OLS(Y, X)
            results = model.fit()
            
            # Store results
            self.store_results(
                data=data,
                results=results,
                index_id=index_id,
                tenor=tenor
            )
            
            # Format coefficients for analysis
            coefficients = {}
            for var in results.params.index:
                coefficients[var] = {
                    'coefficient': results.params[var],
                    't_stat': results.tvalues[var],
                    'p_value': results.pvalues[var]
                }
            
            return {
                'coefficients': coefficients,  # Now properly structured
                'r_squared': results.rsquared,
                'adj_r_squared': results.rsquared_adj,
                'nobs': results.nobs,
                'start_date': data['date'].min(),
                'end_date': data['date'].max()
            }
            
        except Exception as e:
            logger.error(f"Error estimating error correction model: {str(e)}")
            raise
            
    def store_results(self,
                     data: pd.DataFrame,
                     results: sm.regression.linear_model.RegressionResultsWrapper,
                     index_id: str,
                     tenor: str):
        """Store error correction model results"""
        try:
            # Prepare results data
            fitted_changes = results.fittedvalues
            residuals = results.resid
            
            # Store main results
            for idx, row in data.iterrows():
                # Replace NaN/inf with NULL in SQL and ensure float type
                self.conn.execute("""
                    INSERT OR REPLACE INTO error_correction_results
                    (date, index_id, tenor, delta_iv, delta_gev, delta_evoev,
                     delta_dev, delta_kev, delta_sevts, error_correction,
                     fitted_change, residual, r_squared)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    row['date'],
                    index_id,
                    tenor,
                    None if np.isnan(row['delta_iv']) else float(row['delta_iv']),
                    None if np.isnan(row['delta_gev']) else float(row['delta_gev']),
                    None if np.isnan(row['delta_evoev']) else float(row['delta_evoev']),
                    None if np.isnan(row['delta_dev']) else float(row['delta_dev']),
                    None if np.isnan(row['delta_kev']) else float(row['delta_kev']),
                    None if np.isnan(row['delta_sevts']) else float(row['delta_sevts']),
                    None if np.isnan(row['error_correction']) else float(row['error_correction']),
                    None if np.isnan(fitted_changes[idx]) else float(fitted_changes[idx]),
                    None if np.isnan(residuals[idx]) else float(residuals[idx]),
                    None if np.isnan(results.rsquared) else float(results.rsquared)
                ])
            
            # Store coefficients
            for var, coef in results.params.items():
                self.conn.execute("""
                    INSERT OR REPLACE INTO error_correction_coefficients
                    (date, index_id, tenor, variable, coefficient, t_stat, p_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    data['date'].max(),  # Use end date for coefficients
                    index_id,
                    tenor,
                    var,
                    None if np.isnan(coef) else float(coef),
                    None if np.isnan(results.tvalues[var]) else float(results.tvalues[var]),
                    None if np.isnan(results.pvalues[var]) else float(results.pvalues[var])
                ])
            
            logger.info(f"""
            Stored error correction results for {index_id} {tenor}:
            Observations: {len(data)}
            R-squared: {results.rsquared:.4f}
            Error correction coef: {results.params['error_correction']:.4f}
            Error correction t-stat: {results.tvalues['error_correction']:.2f}
            """)
            
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
            raise
            
    def analyze_error_correction(self,
                               index_id: str,
                               tenor: str) -> Dict[str, Any]:
        """Analyze error correction model results"""
        try:
            # Get stored results
            results = self.conn.execute("""
                WITH model_stats AS (
                    SELECT 
                        COUNT(*) as n_obs,
                        AVG(r_squared) as r_squared,
                        CORR(fitted_change, delta_iv) as fit_correlation,
                        AVG(ABS(residual)) as mae,
                        SQRT(AVG(residual * residual)) as rmse
                    FROM error_correction_results
                    WHERE index_id = ? AND tenor = ?
                ),
                error_correction_effect AS (
                    SELECT 
                        coefficient as ec_coefficient,
                        t_stat as ec_tstat,
                        p_value as ec_pvalue
                    FROM error_correction_coefficients
                    WHERE index_id = ?
                        AND tenor = ?
                        AND variable = 'error_correction'
                ),
                residual_stats AS (
                    SELECT 
                        CORR(residual, LAG(residual) OVER (ORDER BY date)) as resid_autocorr,
                        SKEW(residual) as resid_skew,
                        KURTOSIS(residual) as resid_kurt
                    FROM error_correction_results
                    WHERE index_id = ? AND tenor = ?
                )
                SELECT * FROM model_stats, error_correction_effect, residual_stats
            """, [index_id, tenor, index_id, tenor, index_id, tenor]).df()
            
            if len(results) == 0:
                raise ValueError(f"No results found for {index_id} {tenor}")
            
            # Get coefficient significance
            coef_sig = self.conn.execute("""
                SELECT 
                    variable,
                    coefficient,
                    t_stat,
                    p_value,
                    CASE 
                        WHEN ABS(t_stat) > 1.96 THEN 1 
                        ELSE 0 
                    END as is_significant
                FROM error_correction_coefficients
                WHERE index_id = ? AND tenor = ?
                ORDER BY variable
            """, [index_id, tenor]).df()
            
            return {
                'model_stats': results.iloc[0].to_dict(),
                'coefficients': coef_sig.to_dict('records'),
                'step_size': self.step_size,
                'dev_mode': self.dev_mode
            }
            
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            raise
            
    def get_trading_signals(self,
                           index_id: str,
                           tenor: str,
                           z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Generate trading signals from error correction model
        
        Parameters:
        - z_threshold: Z-score threshold for signals
        """
        try:
            signals = self.conn.execute("""
                WITH signal_data AS (
                    SELECT 
                        date,
                        delta_iv,
                        fitted_change,
                        error_correction,
                        residual,
                        residual / STDDEV(residual) OVER () as z_score
                    FROM error_correction_results
                    WHERE index_id = ? AND tenor = ?
                )
                SELECT 
                    date,
                    delta_iv,
                    fitted_change,
                    error_correction,
                    z_score,
                    CASE 
                        WHEN error_correction <= -? THEN 1    -- Buy signal
                        WHEN error_correction >= ? THEN -1    -- Sell signal
                        ELSE 0 
                    END as signal
                FROM signal_data
                ORDER BY date
            """, [index_id, tenor, z_threshold, z_threshold]).df()
            
            logger.info(f"""
            Generated trading signals for {index_id} {tenor}:
            Total observations: {len(signals)}
            Buy signals: {(signals['signal'] == 1).sum()}
            Sell signals: {(signals['signal'] == -1).sum()}
            No signals: {(signals['signal'] == 0).sum()}
            """)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

    def find_referencing_tables(self, window_id: int) -> None:
        """Find all tables referencing a specific window_id directly or indirectly"""
        try:
            # Get all tables in the database
            tables = self.conn.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table'
            """).df()['name'].tolist()

            references = {}
            for table in tables:
                if table != 'forecast_windows':
                    try:
                        cols = self.conn.execute(f"""
                            SELECT * 
                            FROM {table} 
                            LIMIT 0
                        """).df().columns

                        if 'window_id' in cols:
                            refs = self.conn.execute(f"""
                                SELECT COUNT(*) as ref_count 
                                FROM {table} 
                                WHERE window_id = ?
                            """, [window_id]).df()['ref_count'].iloc[0]

                            if refs > 0:
                                references[table] = {
                                    'direct_refs': refs,
                                    'indirect_refs': 0,
                                    'via': None
                                }

                    except Exception as e:
                        logger.debug(f"Skipping table {table}: {str(e)}")
                        continue

            if references:
                logger.info(f"References to window_id {window_id}: {references}")
            else:
                logger.info(f"No references found to window_id {window_id}")

            return references

        except Exception as e:
            logger.error(f"Error finding references: {str(e)}")
            raise

    def setup_forecast_windows(self):
        """
        Create and/or rebuild forecast windows while systematically removing references
        to window_id=7 before deleting it, including logging for multi-col FKs and
        enumerating all "window" columns as a fallback.
        """
        try:
            # Check if forecast_windows exists
            table_exists = self.conn.execute("""
                SELECT COUNT(*) AS count
                FROM sqlite_master
                WHERE type='table'
                  AND name='forecast_windows'
            """).df()['count'].iloc[0] > 0

            if not table_exists:
                # Nothing to do
                logger.info("forecast_windows table does not exist. Skipping setup.")
                return

            self.conn.execute("BEGIN TRANSACTION")
            try:
                logger.info("Deleting rows referencing window_id=7 in child tables...")

                # ─────────────────────────────────────────────────────────
                # A) Enumerate all FOREIGN KEY constraints referencing forecast_windows
                # ─────────────────────────────────────────────────────────
                fk_constraints = self.conn.execute("""
                    SELECT table_name, constraint_text
                    FROM duckdb_constraints
                    WHERE constraint_type = 'FOREIGN KEY'
                      AND UPPER(referenced_table) = 'FORECAST_WINDOWS'
                """).df()

                if fk_constraints.empty:
                    logger.info("No foreign keys referencing forecast_windows found in duckdb_constraints.")
                else:
                    logger.info("Found the following foreign keys referencing forecast_windows:")
                    for idx, row in fk_constraints.iterrows():
                        logger.info(f"  Table={row['table_name']}, Constraint Text={row['constraint_text']}")

                for idx, row in fk_constraints.iterrows():
                    child_table = row["table_name"]
                    ctext = row["constraint_text"].upper()

                    # Example ctext: "FOREIGN KEY(window_id, index_id) REFERENCES forecast_windows(window_id, index_id)"
                    # or "FOREIGN KEY(window_id) REFERENCES forecast_windows(window_id)"

                    # Parse out the (child columns)
                    start_fk = ctext.find("FOREIGN KEY") + len("FOREIGN KEY")
                    col_start = ctext.find("(", start_fk)
                    col_end = ctext.find(")", col_start)
                    if col_start == -1 or col_end == -1:
                        logger.warning(f"Cannot parse referencing columns from {ctext}. Skipping.")
                        continue

                    referencing_cols_part = ctext[col_start+1:col_end].strip()  # e.g. "window_id, index_id"
                    referencing_cols = [x.strip().strip('"') for x in referencing_cols_part.split(",")]

                    if len(referencing_cols) > 1:
                        # Multi-column foreign key. We either skip or try partial cleanup if "window_id" is among them.
                        logger.warning(f"Multi-col FK in {child_table}: {referencing_cols}. Attempt partial cleanup if 'window_id' is included.")
                        if "WINDOW_ID" in [x.upper() for x in referencing_cols]:
                            # We'll do a partial DELETE in case it unblocks the constraint
                            # But if the other column also participates in the constraint, you might still get an error
                            try:
                                before_cnt = self.conn.execute(f"""
                                    SELECT COUNT(*) AS cnt
                                    FROM {child_table}
                                    WHERE window_id = 7
                                """).df()['cnt'].iloc[0]
                                if before_cnt > 0:
                                    self.conn.execute(f"DELETE FROM {child_table} WHERE window_id = 7")
                                    logger.info(f"Deleted {before_cnt} rows from {child_table} on window_id=7 (multi-col constraint).")
                            except Exception as e:
                                logger.error(f"Partial cleanup for multi-col FK in {child_table} failed: {str(e)}")
                        else:
                            logger.warning(f"No 'window_id' col in multi-col FK. You may need custom handling for {child_table}.")
                    else:
                        # Single-column foreign key references (typical case)
                        referencing_col = referencing_cols[0]
                        # Check that child_table actually has referencing_col
                        try:
                            table_cols = self.conn.execute(f"SELECT * FROM {child_table} LIMIT 0").df().columns
                        except Exception as e:
                            logger.warning(f"Could not read cols from {child_table}. Error: {str(e)}. Skipping.")
                            continue

                        if referencing_col.upper() not in [c.upper() for c in table_cols]:
                            logger.warning(f"{child_table} does not contain {referencing_col}? Skipping.")
                            continue

                        # Delete child rows referencing window_id=7
                        before_cnt = self.conn.execute(f"""
                            SELECT COUNT(*) AS cnt
                            FROM {child_table}
                            WHERE {referencing_col} = 7
                        """).df()['cnt'].iloc[0]
                        if before_cnt > 0:
                            self.conn.execute(f"DELETE FROM {child_table} WHERE {referencing_col} = 7")
                            logger.info(f"Deleted {before_cnt} rows from {child_table}.{referencing_col}=7.")

                # ─────────────────────────────────────────────────────────
                # B) Fallback: Enumerate all tables whose columns contain "window"
                #    (to catch columns not literally named "window_id")
                # ─────────────────────────────────────────────────────────
                logger.info("Fallback: checking all tables for columns containing 'window'.")
                all_tables = self.conn.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                """).df()['name'].tolist()

                for tbl in all_tables:
                    if tbl.lower() == 'forecast_windows':
                        continue

                    try:
                        tbl_cols = self.conn.execute(f"SELECT * FROM {tbl} LIMIT 0").df().columns
                    except Exception as e:
                        logger.debug(f"Skipping table {tbl}, can't SELECT * LIMIT 0: {str(e)}")
                        continue

                    # If any col includes 'window' in its name, try a DELETE
                    for c in tbl_cols:
                        if 'window' in c.lower() and c.lower() != 'forecast_windows':
                            # Attempt to delete if numeric
                            # We'll guess that "window_id" or "windowID" or "my_window_col" might hold the ID
                            try:
                                before_cnt = self.conn.execute(f"""
                                    SELECT COUNT(*) AS cnt
                                    FROM {tbl}
                                    WHERE {c} = 7
                                """).df()['cnt'].iloc[0]
                                if before_cnt > 0:
                                    self.conn.execute(f"DELETE FROM {tbl} WHERE {c} = 7")
                                    logger.info(f"[Fallback] Deleted {before_cnt} rows from {tbl}.{c}=7.")
                            except Exception as e:
                                logger.debug(f"[Fallback] Could not DELETE FROM {tbl} WHERE {c}=7. Possibly not numeric. Error: {str(e)}")

                # ─────────────────────────────────────────────────────────
                # C) Attempt final DELETE from forecast_windows
                # ─────────────────────────────────────────────────────────
                logger.info("Finally, deleting window_id=7 from forecast_windows.")
                self.conn.execute("""
                    DELETE FROM forecast_windows
                    WHERE window_id = 7
                """)
                self.conn.execute("COMMIT")

            except Exception as inner_e:
                self.conn.execute("ROLLBACK")
                logger.error(f"Error while clearing forecast_windows references: {str(inner_e)}")
                raise

        except Exception as outer_e:
            logger.error(f"Error rebuilding forecast windows: {str(outer_e)}")
            raise

    def verify_no_gaps_in_windows(self) -> bool:
        """Verify there are no gaps in window_id sequence in forecast_windows."""
        gaps = self.conn.execute("""
            WITH RECURSIVE numbers AS (
                SELECT MIN(window_id) as n FROM forecast_windows
                UNION ALL
                SELECT n + 1 
                FROM numbers 
                WHERE n < (SELECT MAX(window_id) FROM forecast_windows)
            )
            SELECT COUNT(*) as missing_count
            FROM numbers n
            LEFT JOIN forecast_windows fw ON n.n = fw.window_id
            WHERE fw.window_id IS NULL
        """).df()
        return gaps['missing_count'].iloc[0] == 0

    def analyze_window_structure(self, data: pd.DataFrame) -> None:
        """Detailed analysis of window structure and temporal alignment"""
        try:
            # Analyze window growth
            window_growth = data.groupby(pd.Grouper(key='date', freq='Y')).agg({
                'window_end': ['min', 'max', 'count'],
                'date': 'count'
            }).round(2)
            
            # Analyze step sizes
            step_sizes = pd.Series([
                (data['date'].iloc[i] - data['date'].iloc[i-1]).days
                for i in range(1, len(data))
            ])
            
            # Check for seasonality in window selection
            monthly_counts = data.groupby(
                data['date'].dt.to_period('M')
            ).size()
            
            logger.info(f"""
            Detailed Window Analysis:
            
            1. Annual Window Growth:
            {window_growth.to_string()}
            
            2. Step Size Statistics:
            Mean: {step_sizes.mean():.2f} days
            Median: {step_sizes.median():.2f} days
            Std Dev: {step_sizes.std():.2f} days
            Range: {step_sizes.min():.0f} to {step_sizes.max():.0f} days
            
            3. Monthly Distribution:
            {monthly_counts.describe().to_string()}
            
            4. Temporal Gaps:
            Max gap: {step_sizes.max():.0f} days
            Gap locations: {
                [data['date'].iloc[i].strftime('%Y-%m-%d') 
                 for i in range(1, len(data)) 
                 if (data['date'].iloc[i] - data['date'].iloc[i-1]).days > 35]
            }
            """)
            
        except Exception as e:
            logger.error(f"Error analyzing window structure: {str(e)}")
            raise

    def verify_window_sequence(self) -> bool:
        """Verify proper expanding window sequence"""
        try:
            sequence_check = self.conn.execute("""
                WITH window_checks AS (
                    SELECT 
                        window_id,
                        start_date,
                        end_date,
                        LEAD(start_date) OVER (ORDER BY window_id) as next_start,
                        LEAD(end_date) OVER (ORDER BY window_id) as next_end
                    FROM forecast_windows
                    WHERE index_id = 'SPX'
                )
                SELECT 
                    COUNT(*) as total_windows,
                    SUM(CASE 
                        WHEN next_start < start_date THEN 1 
                        ELSE 0 
                    END) as start_date_violations,
                    SUM(CASE 
                        WHEN next_end <= end_date THEN 1 
                        ELSE 0 
                    END) as end_date_violations
                FROM window_checks
            """).df()
            
            is_valid = (
                sequence_check['start_date_violations'].iloc[0] == 0 and
                sequence_check['end_date_violations'].iloc[0] == 0
            )
            
            logger.info(f"""
            Window sequence verification:
            Total windows: {sequence_check['total_windows'].iloc[0]}
            Start date violations: {sequence_check['start_date_violations'].iloc[0]}
            End date violations: {sequence_check['end_date_violations'].iloc[0]}
            Sequence valid: {is_valid}
            """)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying window sequence: {str(e)}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        result = self.conn.execute("""
            SELECT COUNT(*) as count 
            FROM sqlite_master 
            WHERE type='table' AND name=?
        """, [table_name]).df()['count'].iloc[0] > 0
        return result
"""Test script for expanding window regression analysis"""

import logging
from pathlib import Path
import sys
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.level_regression import LevelRegression
from regression.expanding_window import ExpandingWindowRegression

def validate_coefficient_stability(results: dict) -> pd.DataFrame:
    """Analyze stability of regression coefficients"""
    coef_df = results['coefficients']
    
    # Calculate stability metrics for each variable
    stability = []
    for var in coef_df['variable'].unique():
        var_data = coef_df[coef_df['variable'] == var]
        
        stability.append({
            'variable': var,
            'mean': var_data['coefficient'].mean(),
            'std': var_data['coefficient'].std(),
            'min': var_data['coefficient'].min(),
            'max': var_data['coefficient'].max(),
            'significant_pct': (var_data['p_value'] < 0.05).mean() * 100,
            'n_windows': len(var_data)
        })
    
    return pd.DataFrame(stability)

def validate_residual_properties(results: dict) -> pd.DataFrame:
    """Analyze residual properties across windows"""
    residuals = results['residuals']
    
    # Group by window and calculate properties
    properties = []
    for window_end in residuals['window_end_date'].unique():
        window_data = residuals[residuals['window_end_date'] == window_end]
        
        properties.append({
            'window_end': window_end,
            'mean_residual': window_data['residual'].mean(),
            'std_residual': window_data['residual'].std(),
            'skewness': window_data['residual'].skew(),
            'kurtosis': window_data['residual'].kurtosis(),
            'autocorr_lag1': window_data['residual'].autocorr(lag=1),
            'n_obs': len(window_data)
        })
    
    return pd.DataFrame(properties)

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('expanding_window_test')
    
    try:
        # Connect to database
        results_db = Path("results/historical/historical_results.db")
        conn = duckdb.connect(str(results_db))
        
        # First, check if we have level regression results
        logger.info("Checking for level regression results...")
        level_reg = LevelRegression(conn)
        
        # Initialize expanding window regression with smaller minimum window
        exp_reg = ExpandingWindowRegression(
            conn,
            min_window_size=126  # 6 months of data for initial window
        )
        
        # Test parameters
        index_id = 'SPX'
        tenors = ['1M', '2M', '3M', '6M', '12M']
        
        # Store results for analysis
        all_results = {}
        stability_metrics = {}
        residual_properties = {}
        
        for tenor in tenors:
            logger.info(f"\nProcessing {index_id} {tenor}")
            
            try:
                # First run level regression if needed
                logger.info("Running level regression...")
                level_results = level_reg.estimate(index_id, tenor)
                
                logger.info(f"""
                Level regression complete:
                R-squared: {level_results['r_squared']:.4f}
                Observations: {level_results['nobs']}
                """)
                
                # Now run expanding window analysis
                logger.info("Running expanding window analysis...")
                exp_reg.run_expanding_windows(
                    index_id=index_id,
                    tenor=tenor,
                    step_size=21  # Monthly steps in DEV_MODE
                )
                
                # Get results
                results = exp_reg.get_window_results(index_id, tenor)
                all_results[(index_id, tenor)] = results
                
                # Analyze coefficient stability
                stability = validate_coefficient_stability(results)
                stability_metrics[(index_id, tenor)] = stability
                
                logger.info("\nCoefficient Stability:")
                logger.info(stability)
                
                # Analyze residual properties
                properties = validate_residual_properties(results)
                residual_properties[(index_id, tenor)] = properties
                
                logger.info("\nResidual Properties Summary:")
                logger.info(properties.describe())
                
            except Exception as e:
                logger.error(f"Error processing {index_id} {tenor}: {str(e)}")
                continue
        
        # Only save results if we have any
        if stability_metrics and residual_properties:
            # Save analysis results
            output_dir = Path("results/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save stability metrics
            stability_file = output_dir / "coefficient_stability.csv"
            stability_df = pd.concat([
                df.assign(index_id=idx[0], tenor=idx[1])
                for idx, df in stability_metrics.items()
            ])
            stability_df.to_csv(stability_file, index=False)
            
            # Save residual properties
            properties_file = output_dir / "residual_properties.csv"
            properties_df = pd.concat([
                df.assign(index_id=idx[0], tenor=idx[1])
                for idx, df in residual_properties.items()
            ])
            properties_df.to_csv(properties_file, index=False)
            
            logger.info(f"\nAnalysis results saved to {output_dir}")
        else:
            logger.warning("No results to save")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
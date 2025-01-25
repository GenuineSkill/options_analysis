"""Test script for error correction model with proper sequencing"""

import logging
from pathlib import Path
import sys
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.expanding_window import ExpandingWindowRegression
from regression.error_correction import ErrorCorrectionModel
from regression.validation import verify_residual_storage, verify_temporal_consistency
from workflows.run_regression_sequence import run_regression_sequence

def analyze_model_stability(results: dict) -> pd.DataFrame:
    """Analyze stability of error correction model"""
    # Convert coefficients to proper DataFrame format
    coef_data = []
    for variable, values in results['coefficients'].items():
        coef_data.append({
            'variable': variable,
            'coefficient': values['coefficient'],
            't_stat': values['t_stat'],
            'p_value': values['p_value']
        })
    
    # Create DataFrame
    stability = pd.DataFrame(coef_data)
    stability['abs_effect'] = abs(stability['coefficient'])
    stability['significant'] = stability['p_value'] < 0.05
    
    # Sort by absolute effect size
    stability = stability.sort_values('abs_effect', ascending=False)
    
    return stability

def analyze_signal_distribution(signals: pd.DataFrame) -> dict:
    """Analyze distribution of trading signals"""
    analysis = {
        'total_obs': len(signals),
        'buy_signals': (signals['signal'] == 1).sum(),
        'sell_signals': (signals['signal'] == -1).sum(),
        'no_signals': (signals['signal'] == 0).sum(),
        'mean_change': signals.groupby('signal')['delta_iv'].mean().to_dict(),
        'success_rate': {
            'buy': np.nan,
            'sell': np.nan
        }
    }
    
    # Calculate success rates
    buy_signals = signals[signals['signal'] == 1]
    if len(buy_signals) > 0:
        analysis['success_rate']['buy'] = (buy_signals['delta_iv'] > 0).mean()
        
    sell_signals = signals[signals['signal'] == -1]
    if len(sell_signals) > 0:
        analysis['success_rate']['sell'] = (sell_signals['delta_iv'] < 0).mean()
    
    return analysis

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('error_correction_test')
    
    try:
        # Connect to database
        results_db = Path("results/historical/historical_results.db")
        conn = duckdb.connect(str(results_db))
        
        # Test parameters
        index_id = 'SPX'
        tenors = ['1M', '2M', '3M', '6M', '12M']
        dev_mode = True  # Explicitly set dev_mode to match ensemble stats
        step_size = 21 if dev_mode else 1
        
        logger.info(f"""
        Running tests with:
        dev_mode: {dev_mode}
        step_size: {step_size} days
        """)
        
        # Create model with explicit dev_mode
        ec_model = ErrorCorrectionModel(conn, dev_mode=dev_mode)
        
        # Store results for analysis
        all_results = {}
        
        for tenor in tenors:
            logger.info(f"\nProcessing {index_id} {tenor}")
            
            try:
                # Run complete regression sequence
                results = run_regression_sequence(
                    conn=conn,
                    index_id=index_id,
                    tenor=tenor,
                    min_window_size=126,
                    step_size=21
                )
                
                all_results[(index_id, tenor)] = results
                
                # Analyze results
                analyze_model_stability(results['error_correction'])
                signals = analyze_signal_distribution(
                    results['error_correction']['signals']
                )
                
                logger.info(f"""
                Analysis complete for {index_id} {tenor}:
                Expanding Window R²: {results['expanding_window']['r_squared']:.4f}
                Error Correction R²: {results['error_correction']['r_squared']:.4f}
                EC Coefficient: {results['error_correction']['coefficients']['error_correction']['coefficient']:.4f}
                EC t-stat: {results['error_correction']['coefficients']['error_correction']['t_stat']:.2f}
                """)
                
            except Exception as e:
                logger.error(f"Error processing {index_id} {tenor}: {str(e)}")
                continue
                
        # Save results if successful
        if all_results:
            save_analysis_results(all_results)
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main() 
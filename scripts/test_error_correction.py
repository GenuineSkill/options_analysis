"""Test script for error correction model"""

import logging
from pathlib import Path
import sys
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from regression.error_correction import ErrorCorrectionModel
from regression.level_regression import LevelRegression

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
        
        # Initialize models
        level_reg = LevelRegression(conn)
        ec_model = ErrorCorrectionModel(conn, dev_mode=True)
        
        # Test parameters
        index_id = 'SPX'
        tenors = ['1M', '2M', '3M', '6M', '12M']
        
        # Store results for analysis
        model_results = {}
        signal_results = {}
        stability_metrics = {}
        
        for tenor in tenors:
            logger.info(f"\nProcessing {index_id} {tenor}")
            
            try:
                # Ensure level regression results exist
                level_results = level_reg.estimate(index_id, tenor)
                
                logger.info(f"""
                Level regression complete:
                R-squared: {level_results['r_squared']:.4f}
                Observations: {level_results['nobs']}
                """)
                
                # Run error correction model
                ec_results = ec_model.estimate(index_id, tenor)
                model_results[(index_id, tenor)] = ec_results
                
                logger.info(f"""
                Error correction complete:
                R-squared: {ec_results['r_squared']:.4f}
                Observations: {ec_results['nobs']}
                """)
                
                # Analyze model stability
                stability = analyze_model_stability(ec_results)
                stability_metrics[(index_id, tenor)] = stability
                
                logger.info("\nModel Stability:")
                logger.info(stability)
                
                # Generate and analyze trading signals
                signals = ec_model.get_trading_signals(
                    index_id=index_id,
                    tenor=tenor,
                    z_threshold=2.0
                )
                
                signal_analysis = analyze_signal_distribution(signals)
                signal_results[(index_id, tenor)] = signal_analysis
                
                logger.info("\nSignal Analysis:")
                logger.info(f"""
                Total signals: {signal_analysis['total_obs']}
                Buy signals: {signal_analysis['buy_signals']} 
                    (Success: {signal_analysis['success_rate']['buy']:.2%})
                Sell signals: {signal_analysis['sell_signals']}
                    (Success: {signal_analysis['success_rate']['sell']:.2%})
                """)
                
            except Exception as e:
                logger.error(f"Error processing {index_id} {tenor}: {str(e)}")
                continue
        
        # Save analysis results
        if model_results and signal_results:
            output_dir = Path("results/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model stability metrics
            stability_file = output_dir / "error_correction_stability.csv"
            stability_df = pd.concat([
                df.assign(index_id=idx[0], tenor=idx[1])
                for idx, df in stability_metrics.items()
            ])
            stability_df.to_csv(stability_file, index=False)
            
            # Save signal analysis
            signal_file = output_dir / "error_correction_signals.csv"
            signal_df = pd.DataFrame([
                {
                    'index_id': idx[0],
                    'tenor': idx[1],
                    **analysis
                }
                for idx, analysis in signal_results.items()
            ])
            signal_df.to_csv(signal_file, index=False)
            
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
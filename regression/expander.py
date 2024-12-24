from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, date
import pickle

from data_manager.database import GARCHDatabase
from garch.forecaster import GARCHForecaster, ForecastWindow
from garch.estimator import GARCHEstimator
from utils.db_manager import DatabaseManager
from utils.progress import ProgressMonitor

logger = logging.getLogger(__name__)

class ExpandingWindowAnalyzer:
    """Manages expanding window analysis for implied volatility modeling"""
    
    def __init__(self, forecaster, db_manager: Optional[DatabaseManager] = None):
        """Initialize the analyzer with a forecaster and optional database manager"""
        self.forecaster = forecaster
        self.db_manager = db_manager
        self.logger = logging.getLogger('regression.expander')

    def save_results(self, results: Dict, output_dir: Path):
        """Save analysis results to disk"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_file = output_dir / f"{results['index_id']}_results.pkl"
            self.logger.info(f"Saving results to {results_file}")
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

    def process_window(self, window: np.ndarray, implied_vols: np.ndarray, 
                      date: pd.Timestamp, model_type: str = 'GARCH') -> Dict:
        """Process a single window of data and save to database"""
        self.logger.debug(f"Processing window ending {date}")
        
        try:
            # Run GARCH estimation
            garch_results = self.forecaster.estimator.estimate(
                returns=window,
                date=date
            )
            
            # Generate forecasts
            forecasts = self.forecaster.generate_forecasts(
                window=window,
                garch_results=garch_results
            )
            
            # Calculate ensemble statistics
            ensemble_stats = self.forecaster.calculate_ensemble_stats(
                forecasts=forecasts,
                implied_vols=implied_vols
            )
            
            # Package results
            results = {
                'window_size': len(window),
                'garch_results': garch_results,
                'forecasts': forecasts,
                'ensemble_stats': ensemble_stats
            }
            
            # Save to database if available
            if self.db_manager:
                try:
                    # Save GARCH parameters
                    for model, params in garch_results.items():
                        self.db_manager.save_garch_results(
                            date=date,
                            model_type=model,
                            params=params['parameters'],
                            converged=bool(params['converged']),
                            loglik=float(params['loglikelihood'])
                        )
                    
                    # Save forecasts
                    for model, forecast in forecasts.items():
                        self.db_manager.save_forecasts(
                            estimation_date=date,
                            model_type=model,
                            forecasts=forecast
                        )
                    
                    # Save ensemble stats
                    self.db_manager.save_ensemble_stats(
                        date=date,
                        stats=ensemble_stats
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error saving to database: {str(e)}")
                    # Continue processing even if database save fails
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing window ending {date}: {str(e)}")
            raise

    def process_index(self, index_id: str, dates: List[date], returns: np.ndarray, 
                     implied_vols: np.ndarray, output_dir: Path, 
                     monitor: Optional[ProgressMonitor] = None) -> Dict:
        """Process expanding windows for a given index"""
        try:
            self.logger.info(f"Starting new analysis for {index_id}")
            self.logger.info(f"Analysis date range: {dates[0]} to {dates[-1]}")
            
            # Process all data
            windows = self.forecaster.generate_expanding_windows(
                returns=returns,
                start_idx=0
            )
            
            # Process each window
            results = []
            for i, window in enumerate(windows):
                result = self.process_window(
                    window=window,
                    implied_vols=implied_vols[i:i+len(window)],
                    date=dates[i+len(window)-1]
                )
                results.append(result)
                
                if monitor:
                    monitor.update(1)
                    
            # Package results
            analysis_results = {
                'index_id': index_id,
                'dates': dates,
                'results': results
            }
            
            # Save results
            self.save_results(analysis_results, output_dir)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error processing {index_id}: {str(e)}")
            raise
    
    def _calculate_error_correction(self, 
                              dates: np.ndarray,
                              returns: np.ndarray,
                              implied_vols: pd.DataFrame,
                              forecast_windows: List[ForecastWindow]) -> pd.DataFrame:
        """Calculate error correction terms
        
        Note on statistic availability:
        - SEVTS becomes valid immediately (requires only endpoint forecasts)
        - GEV, EVOEV, DEV, and KEV require full forecast sequences
        - Expect ~3 windows before all statistics become valid
        - Some windows may have NaN values due to forecast availability
        
        Parameters:
        -----------
        dates : array-like
            Dates corresponding to returns
        returns : array-like
            Return series to analyze
        implied_vols : DataFrame
            Implied volatility data
        forecast_windows : List[ForecastWindow]
            Windows containing GARCH results
        
        Returns:
        --------
        DataFrame
            Results including all statistics, with expected NaN pattern:
            - SEVTS: Valid from first window
            - Other stats: Valid from ~window 3
            - Occasional NaN windows may occur
        """
        if not forecast_windows:
            logger.warning("No forecast windows provided")
            return pd.DataFrame()
        
        print(f"\nAnalyzing {len(forecast_windows)} forecast windows...")
        
        results = []
        transitions = {}
        
        for i, window in enumerate(forecast_windows):
            try:
                start_date = pd.to_datetime(window.start_date)
                end_date = pd.date_range(start=start_date, periods=len(window.returns), freq='B')[-1]
                
                result = {
                    'start_date': start_date,
                    'end_date': end_date
                }
                
                if hasattr(window, 'ensemble_stats'):
                    stats = window.ensemble_stats
                    
                    # Track first valid appearance of each statistic
                    for stat, value in stats.items():
                        if pd.notna(value) and stat not in transitions:
                            transitions[stat] = {
                                'window': i,
                                'start_date': start_date,
                                'end_date': end_date,
                                'value': value
                            }
                            print(f"\n{stat} first valid in window {i}:")
                            print(f"  Start: {start_date.date()}")
                            print(f"  End: {end_date.date()}")
                            print(f"  Value: {value:.6f}")
                            
                            # Log GARCH details at transition
                            if hasattr(window, 'garch_results'):
                                n_models = len(window.garch_results)
                                print(f"  GARCH models: {n_models}")
                                if n_models > 0 and hasattr(window.garch_results[0], 'forecasts_annualized'):
                                    print(f"  Forecast length: {len(window.garch_results[0].forecasts_annualized)}")
                
                    result.update(stats)
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Error processing window {i}: {str(e)}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Print transition summary
        print("\nTransition Summary:")
        for stat in ['GEV', 'EVOEV', 'DEV', 'KEV', 'SEVTS']:
            if stat in transitions:
                info = transitions[stat]
                print(f"\n{stat}:")
                print(f"  First valid: Window {info['window']}")
                print(f"  Period: {info['start_date'].date()} to {info['end_date'].date()}")
                
                # Show pattern of values after transition
                if stat in results_df.columns:
                    next_values = results_df[stat].iloc[info['window']:info['window']+3]
                    print(f"  Next three values: {next_values.tolist()}")
            else:
                print(f"\n{stat}: Never became valid")
        
        return results_df
    
    def process_multiple_indices(self,
                              data_dict: Dict[str, Dict],
                              forecast_horizon: int = 252) -> Dict[str, pd.DataFrame]:
        """
        Process multiple indices
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary of index data containing returns, dates, and implied vols
        forecast_horizon : int
            Forecast horizon in days
            
        Returns:
        --------
        dict
            Results for each index
        """
        results = {}
        for index_id, data in data_dict.items():
            try:
                results[index_id] = self.process_index(
                    index_id=index_id,
                    returns=data['returns'],
                    dates=data['dates'],
                    implied_vols=data['implied_vols'],
                    forecast_horizon=forecast_horizon
                )
            except Exception as e:
                logger.error(f"Error processing {index_id}: {str(e)}")
                results[index_id] = pd.DataFrame()
                
        return results
    
    def close(self):
        """Close database connection"""
        self.db.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Example usage
if __name__ == '__main__':
    # Example data
    data = {
        'SPX': {
            'returns': np.random.normal(0, 1, 2000) * 0.01,
            'dates': pd.date_range('2015-01-01', periods=2000),
            'implied_vols': pd.DataFrame(
                np.random.normal(15, 2, (2000, 5)),
                columns=['1M', '2M', '3M', '6M', '12M']
            )
        }
    }
    
    # Initialize analyzer
    analyzer = ExpandingWindowAnalyzer('garch_results.db')
    
    # Process indices
    results = analyzer.process_multiple_indices(data)
    
    # Print summary
    for index_id, df in results.items():
        print(f"\nResults for {index_id}:")
        print(df.describe())
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from data_manager.database import GARCHDatabase
from garch.forecaster import GARCHForecaster, ForecastWindow
from garch.estimator import GARCHEstimator

logger = logging.getLogger(__name__)

class ExpandingWindowAnalyzer:
    """Manages expanding window analysis for implied volatility modeling"""
    
    def __init__(self,
                db_path: Union[str, Path],
                min_window: int = 1260,
                step_size: int = 21,
                forecaster: Optional[GARCHForecaster] = None):
        """
        Initialize analyzer
        
        Parameters:
        -----------
        db_path : str or Path
            Path to database for storing results
        min_window : int
            Minimum window size (default 5 years)
        step_size : int
            Step size for expanding windows (default 21 days)
        forecaster : GARCHForecaster, optional
            Pre-configured forecaster instance (creates new one if None)
        """
        if min_window <= 0:
            raise ValueError("min_window must be positive")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        
        self.db = GARCHDatabase(db_path)
        self.forecaster = forecaster or GARCHForecaster(
            min_window=min_window,
            step_size=step_size
        )
        
    def process_index(self,
                    index_id: str,
                    returns: np.ndarray,
                    dates: np.ndarray,
                    implied_vols: pd.DataFrame,
                    forecast_horizon: int = 252) -> pd.DataFrame:
        """
        Process expanding windows for a single index
        
        Parameters:
        -----------
        index_id : str
            Identifier for the equity index
        returns : array-like
            Return series to analyze
        dates : array-like
            Dates corresponding to returns
        implied_vols : DataFrame
            Implied volatility data with columns for different tenors
        forecast_horizon : int
            Horizon for forecasts in days
            
        Returns:
        --------
        DataFrame
            Results including error correction terms
        """
        # Validate inputs
        if len(returns) != len(dates):
            raise ValueError("Length mismatch between returns and dates")
        if len(dates) != len(implied_vols):
            raise ValueError("Length mismatch between dates and implied_vols")
        
        try:
            # Get latest window from database
            latest = self.db.get_latest_window(index_id)
            
            # Determine start point
            if latest is not None:
                start_idx = np.where(dates > latest['end_date'])[0][0]
                logger.info(f"Continuing from {dates[start_idx]} for {index_id}")
            else:
                start_idx = 0
                logger.info(f"Starting new analysis for {index_id}")
            
            # Generate expanding windows
            windows = self.forecaster.generate_expanding_windows(
                returns=returns[start_idx:],
                dates=dates[start_idx:],
                forecast_horizon=forecast_horizon
            )
            
            # Store results
            for window in windows:
                self.db.store_forecast_window(window, index_id)
            
            # Calculate error correction terms
            results_df = self._calculate_error_correction(
                dates=dates,
                returns=returns,
                implied_vols=implied_vols,
                forecast_windows=windows
            )
            
            # Debug check before accessing end_date
            if 'end_date' not in results_df.columns:
                logger.error(f"Available columns: {results_df.columns.tolist()}")
                raise KeyError("Missing end_date column")
            
            # Add date column explicitly, only if we have results
            if not results_df.empty:
                results_df['date'] = pd.to_datetime(results_df['end_date'])
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing {index_id}: {str(e)}")
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
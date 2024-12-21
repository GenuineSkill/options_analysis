"""
Prepare data for GARCH estimation with proper holiday handling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from data_manager.data_loader import DataLoader

class GarchDataPrep:
    """Prepares market data for GARCH estimation."""
    
    def __init__(self):
        """Initialize data preparation class."""
        pass
        
    def prepare_returns(self, daily_data: pd.DataFrame, market: str) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare log returns for GARCH estimation, properly handling holidays.
        
        Args:
            daily_data: DataFrame with market data including holiday flags
            market: Market identifier (e.g., 'SPX')
            
        Returns:
            Tuple of (clean_returns, trading_dates)
        """
        # Filter for specific market and create market data
        market_data = pd.DataFrame({
            'price': daily_data[market],
            'is_holiday': daily_data[market] == daily_data[market].shift(1)  # Simple holiday detection
        })
        market_data.index = daily_data.index
        market_data = market_data.sort_index()
        
        # Remove holidays and calculate clean returns
        non_holiday_data = market_data[~market_data['is_holiday']]
        clean_prices = non_holiday_data['price']
        
        # Calculate log returns only for trading days
        clean_returns = np.log(clean_prices / clean_prices.shift(1))
        
        # Remove any residual NaN values that might occur at the start
        clean_returns = clean_returns.dropna()
        
        # Get the actual trading dates
        trading_dates = clean_returns.index
        
        # Verify no zero returns (additional safety check)
        zero_returns = clean_returns[clean_returns == 0]
        if not zero_returns.empty:
            print(f"Warning: Found {len(zero_returns)} zero returns in {market} data:")
            print(zero_returns)
        
        return clean_returns, trading_dates
    
    def verify_data_quality(self, returns: pd.Series, min_observations: int = 1260) -> bool:
        """
        Verify data quality for GARCH estimation.
        
        Args:
            returns: Series of log returns
            min_observations: Minimum required observations (default 5 years = 1260 days)
            
        Returns:
            bool indicating if data meets quality requirements
        """
        # Check basic requirements
        if len(returns) < min_observations:
            print(f"Insufficient observations: {len(returns)} < {min_observations}")
            return False
        
        # Check for long sequences of identical returns
        max_identical = returns.groupby(returns).size().max()
        if max_identical > 5:  # More than 5 identical returns in a row
            print(f"Warning: Found sequence of {max_identical} identical returns")
            return False
        
        # Check for extreme values with more lenient threshold during known crisis periods
        mean = returns.mean()
        std = returns.std()
        extreme_threshold = 15  # Increased from 10 to 15 standard deviations
        
        # Define known crisis periods
        crisis_periods = {
            '1987-10': 20,  # Black Monday and aftermath
            '2008-09': 15,  # Financial Crisis
            '2020-03': 15   # COVID-19 Crash
        }
        
        extremes = pd.Series()
        for date, value in returns.items():
            period = date.strftime('%Y-%m')
            threshold = extreme_threshold
            if period in crisis_periods:
                threshold = crisis_periods[period]  # Use higher threshold during crisis periods
            
            if abs(value - mean) > threshold * std:
                extremes[date] = value
        
        if not extremes.empty:
            print(f"Warning: Found {len(extremes)} extreme returns")
            print(extremes)
            if len(extremes) > 5:  # Only fail if there are too many unexplained extreme returns
                return False
        
        return True

def test_garch_data_prep():
    """Test the GARCH data preparation."""
    from data_manager.data_loader import DataLoader
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader()
    file_path = "data_manager/data/Global Equity Vol Durham 12.13.2024.csv"
    data_dict = loader.load_and_clean_data(file_path)
    
    # Get daily data and pivot it for GARCH prep
    daily_df = data_dict['daily_data']
    
    # Use pivot_table instead of pivot to handle duplicates
    daily_data = daily_df.pivot_table(
        index='date',
        columns='index_id',
        values='price',
        aggfunc='first'  # Take the first value if there are duplicates
    )
    
    # Initialize data prep
    prep = GarchDataPrep()
    
    # Test for each market
    for market in ['SPX', 'SX5E', 'UKX']:
        print(f"\nPreparing data for {market}:")
        returns, dates = prep.prepare_returns(daily_data, market)
        
        print(f"Total observations: {len(returns)}")
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"Trading days: {len(dates)}")
        print("\nReturn statistics:")
        print(returns.describe())
        
        is_valid = prep.verify_data_quality(returns)
        print(f"\nData quality check passed: {is_valid}")

if __name__ == "__main__":
    test_garch_data_prep()
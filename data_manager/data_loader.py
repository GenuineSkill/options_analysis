"""
Data loader implementation with validation for Durham (2022) Options Analysis.
"""

import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
from data_manager.data_validator import DataValidator

class DataLoader:
    def __init__(self, db_path: str = "data_manager/data/market_data.db"):
        """Initialize data loader with database path."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self.validator = DataValidator()
        
        # Load holiday calendar
        holiday_file = Path(__file__).parent / "data/market_holidays_1987_2027.csv"
        self.holiday_calendar = pd.read_csv(holiday_file)
        self.holiday_calendar['date'] = pd.to_datetime(self.holiday_calendar['date'])
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        # Daily market data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_data (
                date DATE,
                index_id VARCHAR,
                price DECIMAL,
                rf_rate DECIMAL,
                div_yield DECIMAL,
                is_holiday BOOLEAN,
                is_ntrade BOOLEAN,
                returns DECIMAL,
                PRIMARY KEY (date, index_id)
            )
        """)

        # Implied volatility data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS implied_vols (
                date DATE,
                index_id VARCHAR,
                tenor VARCHAR,
                iv DECIMAL,
                PRIMARY KEY (date, index_id, tenor)
            )
        """)

    def load_and_clean_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load and clean data from CSV file with validation."""
        print("\nLoading and validating data...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop('Unnamed: 0', axis=1)
        
        # Merge with holiday calendar
        df = pd.merge(df, self.holiday_calendar, on='date', how='left')
        
        # Create daily data DataFrame with holiday handling
        daily_data = []
        for index in ['SPX', 'SX5E', 'UKX']:
            holiday_col = f'{index}_holiday'
            ntrade_col = f'{index}_NTRADE'
            
            index_data = pd.DataFrame({
                'date': df['date'],
                'index_id': index,
                'price': df[index],
                'rf_rate': df[f'{index}_RF'],
                'div_yield': df[f'{index}_DY'],
                'is_holiday': df[holiday_col],
                'is_ntrade': df[ntrade_col]  # Keep NTRADE as is for later analysis
            })
            
            # Calculate returns
            index_data['returns'] = index_data['price'].pct_change()
            
            # Set returns to 0 on holidays
            index_data.loc[index_data['is_holiday'] == 1, 'returns'] = 0.0
            
            daily_data.append(index_data)
            
        daily_df = pd.concat(daily_data, ignore_index=True)
        
        # Create implied volatility DataFrame (include all days)
        iv_data = []
        tenors = ['1M', '2M', '3M', '6M', '12M']
        for index in ['SPX', 'SX5E', 'UKX']:
            for tenor in tenors:
                col_name = f'{index}_{tenor}'
                if col_name in df.columns:
                    tenor_data = pd.DataFrame({
                        'date': df['date'],
                        'index_id': index,
                        'tenor': tenor,
                        'iv': df[col_name]
                    })
                    iv_data.append(tenor_data)
                    
        iv_df = pd.concat(iv_data, ignore_index=True)
        
        # Print data quality summary
        self._print_data_quality_summary(daily_df, iv_df)
        
        return {
            'daily_data': daily_df,
            'implied_vols': iv_df
        }

    def _print_data_quality_summary(self, daily_df: pd.DataFrame, iv_df: pd.DataFrame):
        """Print summary of data quality metrics."""
        print("\nData Quality Summary:")
        print("-" * 50)
        
        # Daily data summary
        print("\nDaily Data:")
        print(f"Total rows: {len(daily_df):,}")
        print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        
        # Trading day summary by market
        print("\nTrading Day Summary:")
        for market in ['SPX', 'SX5E', 'UKX']:
            market_data = daily_df[daily_df['index_id'] == market]
            total_days = len(market_data)
            ntrade_days = market_data['is_ntrade'].sum()
            print(f"{market}:")
            print(f"  Trading days: {total_days:,}")
            print(f"  NTRADE days: {ntrade_days:,} ({ntrade_days/total_days*100:.1f}%)")
            
        # IV data summary with trading day alignment
        print("\nImplied Volatility Data (Trading Days Only):")
        print(f"Total rows: {len(iv_df):,}")
        
        # IV coverage by market and tenor
        print("\nIV Coverage by Market and Tenor:")
        for market in ['SPX', 'SX5E', 'UKX']:
            market_data = iv_df[iv_df['index_id'] == market]
            print(f"\n{market}:")
            for tenor in ['1M', '2M', '3M', '6M', '12M']:
                tenor_data = market_data[market_data['tenor'] == tenor]
                valid_count = tenor_data['iv'].notna().sum()
                total_count = len(tenor_data)
                coverage_pct = (valid_count / total_count) * 100
                print(f"  {tenor}: {valid_count:,} valid values ({coverage_pct:.1f}% coverage)")

    def save_to_database(self, data_dict: Dict[str, pd.DataFrame]):
        """Save cleaned data to DuckDB."""
        # Save daily data
        if 'daily_data' in data_dict:
            self.conn.register('daily_data_df', data_dict['daily_data'])
            self.conn.execute("""
                INSERT OR REPLACE INTO daily_data 
                SELECT 
                    date, 
                    index_id, 
                    price, 
                    rf_rate, 
                    div_yield, 
                    is_holiday,
                    is_ntrade,
                    returns
                FROM daily_data_df
            """)

        # Save implied volatility data
        if 'implied_vols' in data_dict:
            self.conn.register('implied_vols_df', data_dict['implied_vols'])
            self.conn.execute("""
                INSERT OR REPLACE INTO implied_vols
                SELECT 
                    date, 
                    index_id, 
                    tenor, 
                    iv 
                FROM implied_vols_df
            """)

    def close(self):
        """Close database connection."""
        self.conn.close()
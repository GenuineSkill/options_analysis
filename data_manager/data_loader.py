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
        """
        Load and clean data from CSV file with validation.
        
        Args:
            file_path: Path to data file
        Returns:
            Dict with cleaned DataFrames for daily_data and implied_vols
        """
        print("\nLoading and validating data...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop('Unnamed: 0', axis=1)
        
        # Validate the data
        is_valid, issues, holiday_dict = self.validator.validate_market_data(df)
        
        # Report any validation issues
        if not is_valid:
            print("\nValidation Issues Found:")
            print("-" * 50)
            for issue in issues:
                print(f"- {issue}")
            print("\nProceeding with data loading despite issues...")
        
        # Create daily data DataFrame
        daily_data = []
        for index in ['SPX', 'SX5E', 'UKX']:
            index_data = pd.DataFrame({
                'date': df['date'],
                'index_id': index,
                'price': df[index],
                'rf_rate': df[f'{index}_RF'],
                'div_yield': df[f'{index}_DY'],
                'is_holiday': holiday_dict[f'{index}_holiday']
            })
            daily_data.append(index_data)
        daily_df = pd.concat(daily_data, ignore_index=True)
        
        # Create implied volatility DataFrame
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
        self._print_data_quality_summary(daily_df, iv_df, holiday_dict)
        
        return {
            'daily_data': daily_df,
            'implied_vols': iv_df
        }

    def _print_data_quality_summary(self, daily_df: pd.DataFrame, 
                                  iv_df: pd.DataFrame, 
                                  holiday_dict: Dict[str, pd.Series]):
        """Print summary of data quality metrics."""
        print("\nData Quality Summary:")
        print("-" * 50)
        
        # Daily data summary
        print("\nDaily Data:")
        print(f"Total rows: {len(daily_df):,}")
        print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        
        # Holiday summary by market
        print("\nHoliday Summary:")
        for market in ['SPX', 'SX5E', 'UKX']:
            holidays = holiday_dict[f'{market}_holiday']
            holiday_count = holidays.sum()
            holiday_pct = (holiday_count / len(holidays)) * 100
            print(f"{market}: {holiday_count:,} holidays ({holiday_pct:.1f}%)")
        
        # IV data summary
        print("\nImplied Volatility Data:")
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
                    is_holiday
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
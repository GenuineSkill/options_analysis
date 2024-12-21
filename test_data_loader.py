"""
Test script for data_loader.py with enhanced error handling and debugging
"""

from data_manager.data_loader import DataLoader
import pandas as pd
import os
from typing import Dict
import duckdb

def test_data_loading():
    print("Starting data loader test...")
    
    # Initialize data loader
    print("\n1. Initializing DataLoader...")
    loader = DataLoader()
    
    try:
        # Define file path
        file_path = "data_manager/data/Global Equity Vol Durham 12.13.2024.csv"
        print(f"\n2. Loading data from: {file_path}")
        
        # First, let's examine the raw data
        print("\n3. Examining raw data structure...")
        raw_df = pd.read_csv(file_path, nrows=5)
        print("\nFirst 5 rows of raw data:")
        print(raw_df)
        print("\nColumns in raw data:", raw_df.columns.tolist())
        
        # Load and clean data
        print("\n4. Loading and cleaning data...")
        data_dict = loader.load_and_clean_data(file_path)
        
        # Print basic information about loaded data
        print("\n5. Data loaded successfully!")
        if 'daily_data' in data_dict:
            print("\nDaily Data Summary:")
            print("-" * 50)
            daily_df = data_dict['daily_data']
            print(f"Shape: {daily_df.shape}")
            print("\nFirst few rows of daily data:")
            print(daily_df.head())
            print("\nColumns:", daily_df.columns.tolist())
        
        if 'implied_vols' in data_dict:
            print("\nImplied Volatility Data Summary:")
            print("-" * 50)
            iv_df = data_dict['implied_vols']
            print(f"Shape: {iv_df.shape}")
            print("\nFirst few rows of IV data:")
            print(iv_df.head())
            print("\nColumns:", iv_df.columns.tolist())
        
        # Save to database
        print("\n6. Saving to database...")
        loader.save_to_database(data_dict)
        print("Data saved successfully!")
        
        # Verify data in database
        print("\n7. Verifying database contents...")
        daily_count = loader.conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
        iv_count = loader.conn.execute("SELECT COUNT(*) FROM implied_vols").fetchone()[0]
        print(f"Records in daily_data table: {daily_count}")
        print(f"Records in implied_vols table: {iv_count}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
    finally:
        print("\n8. Closing database connection...")
        loader.close()
        print("\nTest completed!")

class DataLoader:
    def __init__(self, db_path: str = "data_manager/data/market_data.db"):
        """Initialize data loader with database path."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
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

    def load_and_clean_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Load and clean data from CSV file."""
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['Unnamed: 0'])
        df = df.drop('Unnamed: 0', axis=1)
        
        # Create daily data DataFrame
        daily_data = []
        for index in ['SPX', 'SX5E', 'UKX']:
            index_data = pd.DataFrame({
                'date': df['date'],
                'index_id': index,
                'price': df[index],
                'rf_rate': df[f'{index}_RF'],
                'div_yield': df[f'{index}_DY'],
                'is_holiday': df[index].diff() == 0  # Simple holiday detection
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
        
        return {
            'daily_data': daily_df,
            'implied_vols': iv_df
        }

if __name__ == "__main__":
    test_data_loading()
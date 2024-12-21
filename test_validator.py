"""
Test script for validated data loader
"""

from data_manager.data_loader import DataLoader
import pandas as pd
import os

def test_validated_data_loading():
    print("Starting validated data loader test...")
    
    # Initialize data loader
    print("\n1. Initializing DataLoader...")
    loader = DataLoader()
    
    try:
        # Define file path
        file_path = "data_manager/data/Global Equity Vol Durham 12.13.2024.csv"
        print(f"\n2. Loading data from: {file_path}")
        
        # Load and clean data (validation happens inside)
        data_dict = loader.load_and_clean_data(file_path)
        
        # Verify database storage
        print("\n3. Verifying database storage...")
        daily_count = loader.conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
        iv_count = loader.conn.execute("SELECT COUNT(*) FROM implied_vols").fetchone()[0]
        print(f"Records in daily_data table: {daily_count:,}")
        print(f"Records in implied_vols table: {iv_count:,}")
        
        # Sample checks
        print("\n4. Running sample database queries...")
        
        # Check holiday distribution
        print("\nHoliday Distribution by Market:")
        holiday_query = """
            SELECT 
                index_id,
                COUNT(*) as total_days,
                SUM(CASE WHEN is_holiday THEN 1 ELSE 0 END) as holiday_count,
                ROUND(SUM(CASE WHEN is_holiday THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as holiday_pct
            FROM daily_data
            GROUP BY index_id
        """
        print(loader.conn.execute(holiday_query).fetchdf().to_string(index=False))
        
        # Check IV coverage
        print("\nIV Coverage by Market and Tenor:")
        iv_query = """
            SELECT 
                index_id,
                tenor,
                COUNT(*) as total_records,
                SUM(CASE WHEN iv IS NOT NULL THEN 1 ELSE 0 END) as valid_records,
                ROUND(SUM(CASE WHEN iv IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as coverage_pct
            FROM implied_vols
            GROUP BY index_id, tenor
            ORDER BY index_id, tenor
        """
        print(loader.conn.execute(iv_query).fetchdf().to_string(index=False))
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())
    finally:
        print("\n5. Closing database connection...")
        loader.close()
        print("\nTest completed!")

if __name__ == "__main__":
    test_validated_data_loading()
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from pathlib import Path

def create_holiday_calendars(start_date, end_date, output_file):
    """
    Create a combined holiday calendar with dummy variables for all markets/indices.
    
    Parameters:
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - output_file (str): Path to save the CSV file.
    
    Returns:
    - DataFrame: A DataFrame with weekdays, holiday dummies, and NTRADE dummies.
    """
    # Get market calendars
    nyse_cal = mcal.get_calendar('NYSE')
    lse_cal = mcal.get_calendar('LSE')
    eurex_cal = mcal.get_calendar('EUREX')
    
    # Generate schedules for each market
    nyse_schedule = nyse_cal.schedule(start_date=start_date, end_date=end_date)
    lse_schedule = lse_cal.schedule(start_date=start_date, end_date=end_date)
    eurex_schedule = eurex_cal.schedule(start_date=start_date, end_date=end_date)
    
    # Generate 5-day weekday calendar
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Create base DataFrame
    calendar_df = pd.DataFrame({'date': all_days})
    
    # Add day of week
    calendar_df['weekday'] = calendar_df['date'].dt.dayofweek  # Monday = 0
    
    # Add holiday indicators for each market
    calendar_df['SPX_holiday'] = (~calendar_df['date'].isin(nyse_schedule.index)).astype(int)
    calendar_df['UKX_holiday'] = (~calendar_df['date'].isin(lse_schedule.index)).astype(int)
    calendar_df['SX5E_holiday'] = (~calendar_df['date'].isin(eurex_schedule.index)).astype(int)
    
    # Create NTRADE indicators
    for index in ['SPX', 'UKX', 'SX5E']:
        # Initialize NTRADE as 0
        calendar_df[f'{index}_NTRADE'] = 0
        
        holiday_col = f'{index}_holiday'
        calendar_df[f'{index}_NTRADE'] = calendar_df.apply(
            lambda row: 1 if (
                # Current day is not a holiday
                row[holiday_col] == 0 and
                # Previous day exists
                row.name > 0 and
                (
                    # Previous day was Friday (for regular weekends)
                    calendar_df.loc[row.name - 1, 'weekday'] == 4 or
                    # Previous day was a holiday (for mid-week holidays or long weekends)
                    calendar_df.loc[row.name - 1, holiday_col] == 1
                )
            ) else 0,
            axis=1
        )
    
    # Drop weekday column as it was only needed for calculation
    calendar_df = calendar_df.drop('weekday', axis=1)
    
    # Save the DataFrame to a CSV file
    calendar_df.to_csv(output_file, index=False)
    print(f"Saved combined calendar to {output_file}")
    
    # Display first few rows as a sample
    print("\nFirst few rows of the calendar:")
    print(calendar_df.head())
    
    # Display summary statistics
    print("\nSummary of holiday and NTRADE days:")
    for index in ['SPX', 'UKX', 'SX5E']:
        n_holidays = calendar_df[f'{index}_holiday'].sum()
        n_ntrade = calendar_df[f'{index}_NTRADE'].sum()
        print(f"\n{index}:")
        print(f"  Holiday days: {n_holidays}")
        print(f"  NTRADE days: {n_ntrade}")
    
    return calendar_df

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Define the date range
start_date = "1987-01-01"
end_date = "2027-12-31"

# Generate and save combined calendar
output_file = SCRIPT_DIR / "market_holidays_1987_2027.csv"
calendar_df = create_holiday_calendars(start_date, end_date, output_file)

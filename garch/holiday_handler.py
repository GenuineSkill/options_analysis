"""Handles market holiday detection and date validation"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HolidayHandler:
    def __init__(self):
        """Initialize holiday handler with market calendar"""
        # Load holiday calendar
        holiday_file = Path(__file__).parent.parent / "data_manager/data/market_holidays_1987_2027.csv"
        self.calendar = pd.read_csv(holiday_file)
        self.calendar['date'] = pd.to_datetime(self.calendar['date'])
        
        # Set date range
        self.start_date = self.calendar['date'].min()
        self.end_date = self.calendar['date'].max()
        
        logger.info(
            f"Initialized holiday calendar from {self.start_date:%Y-%m-%d} "
            f"to {self.end_date:%Y-%m-%d}"
        )

    def validate_date(self, date: datetime) -> bool:
        """Check if date is within valid range"""
        return self.start_date <= pd.Timestamp(date) <= self.end_date

    def is_trading_day(self, date: datetime, market: str = 'SPX') -> bool:
        """Check if given date is a trading day for market"""
        date = pd.Timestamp(date)
        if not self.validate_date(date):
            return False
            
        holiday_col = f'{market}_holiday'
        return not self.calendar[
            self.calendar['date'] == date
        ][holiday_col].iloc[0]

    def get_trading_days(self, start: datetime, end: datetime, market: str = 'SPX') -> pd.DatetimeIndex:
        """Get trading days between start and end for market"""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        holiday_col = f'{market}_holiday'
        
        trading_days = self.calendar[
            (self.calendar['date'] >= start) &
            (self.calendar['date'] <= end) &
            (self.calendar[holiday_col] == 0)
        ]['date']
        
        return pd.DatetimeIndex(trading_days) 
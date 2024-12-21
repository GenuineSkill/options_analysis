"""
Data validation and holiday detection for equity market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class DataValidator:
    """Validates market data and detects holidays for different markets."""
    
    def __init__(self):
        # Define expected columns for validation
        self.expected_columns = {
            'price_cols': ['SPX', 'SX5E', 'UKX'],
            'rf_cols': ['SPX_RF', 'SX5E_RF', 'UKX_RF'],
            'dy_cols': ['SPX_DY', 'SX5E_DY', 'UKX_DY'],
            'iv_tenors': ['1M', '2M', '3M', '6M', '12M']
        }
        
        # Define reasonable bounds for data validation
        self.validation_bounds = {
            'price': {'min': 0, 'max': 100000},
            'rf_rate': {'min': -5, 'max': 25},  # Percent
            'div_yield': {'min': 0, 'max': 20},  # Percent
            'iv': {'min': 0, 'max': 150}  # Percent
        }
        
        # Known universal market holidays (month, day)
        self.universal_holidays = [
            (1, 1),   # New Year's Day
            (12, 25), # Christmas
        ]
        
        # Market-specific holidays
        self.market_holidays = {
            'SPX': [
                (1, -3),   # MLK Day (third Monday in January)
                (2, -3),   # Presidents Day (third Monday in February)
                (7, 4),    # Independence Day
                (-1, -1),  # Last Monday in May (Memorial Day)
                (9, -1),   # First Monday in September (Labor Day)
                (11, -4),  # Fourth Thursday in November (Thanksgiving)
            ],
            'SX5E': [
                (5, 1),    # Labor Day
                (12, 26),  # Boxing Day
                (5, -4),   # Ascension Day (40 days after Easter)
            ],
            'UKX': [
                (5, -1),   # Early May Bank Holiday
                (5, -4),   # Spring Bank Holiday
                (8, -1),   # Summer Bank Holiday
                (12, 26),  # Boxing Day
            ]
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates the market data DataFrame.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        required_cols = (
            ['date'] + 
            self.expected_columns['price_cols'] + 
            self.expected_columns['rf_cols'] + 
            self.expected_columns['dy_cols']
        )
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check for data completeness
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues.append(f"Column {col} has {missing_count} missing values")
        
        # Validate data bounds
        for index in self.expected_columns['price_cols']:
            # Price validation
            price_issues = self._validate_bounds(
                df[index],
                self.validation_bounds['price']['min'],
                self.validation_bounds['price']['max'],
                f"{index} price"
            )
            issues.extend(price_issues)
            
            # RF rate validation
            rf_issues = self._validate_bounds(
                df[f"{index}_RF"],
                self.validation_bounds['rf_rate']['min'],
                self.validation_bounds['rf_rate']['max'],
                f"{index} risk-free rate"
            )
            issues.extend(rf_issues)
            
            # Dividend yield validation
            dy_issues = self._validate_bounds(
                df[f"{index}_DY"],
                self.validation_bounds['div_yield']['min'],
                self.validation_bounds['div_yield']['max'],
                f"{index} dividend yield"
            )
            issues.extend(dy_issues)
            
            # IV validation for each tenor
            for tenor in self.expected_columns['iv_tenors']:
                col_name = f"{index}_{tenor}"
                if col_name in df.columns:
                    iv_issues = self._validate_bounds(
                        df[col_name],
                        self.validation_bounds['iv']['min'],
                        self.validation_bounds['iv']['max'],
                        f"{index} {tenor} implied volatility"
                    )
                    issues.extend(iv_issues)
        
        return len(issues) == 0, issues

    def _validate_bounds(self, series: pd.Series, min_val: float, max_val: float, name: str) -> List[str]:
        """Validates that values fall within expected bounds."""
        issues = []
        
        # Check for values below minimum
        below_min = series[series < min_val]
        if not below_min.empty:
            issues.append(
                f"{name}: {len(below_min)} values below minimum of {min_val} "
                f"(first occurrence at index {below_min.index[0]})"
            )
        
        # Check for values above maximum
        above_max = series[series > max_val]
        if not above_max.empty:
            issues.append(
                f"{name}: {len(above_max)} values above maximum of {max_val} "
                f"(first occurrence at index {above_max.index[0]})"
            )
        
        return issues

    def detect_holidays(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detects holidays for each market using multiple methods:
        1. Price unchanged from previous day
        2. Known holidays
        3. Weekends
        4. Anomalous trading volumes or price changes
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dict with holiday boolean Series for each market
        """
        # Ensure we have a DateTimeIndex
        if 'date' in df.columns:
            df = df.set_index('date')
        
        holidays = {}
        
        for market in self.expected_columns['price_cols']:
            # Initialize holiday series
            holidays[f"{market}_holiday"] = pd.Series(False, index=df.index)
            
            # Method 1: Detect unchanged prices
            unchanged_prices = df[market] == df[market].shift(1)
            
            # Method 2: Weekend detection (now works because index is datetime)
            weekends = df.index.dayofweek.isin([5, 6])  # Saturday = 5, Sunday = 6
            
            # Method 3: Known holidays
            known_holidays = self._detect_known_holidays(df.index, market)
            
            # Method 4: Detect anomalous price changes
            # Calculate rolling standard deviation of returns
            returns = df[market].pct_change()
            rolling_std = returns.rolling(window=21).std()
            mean_std = rolling_std.mean()
            
            # Flag days with suspiciously low price changes
            suspicious_changes = (abs(returns) < mean_std * 0.01) & (returns != 0)
            
            # Combine all holiday detection methods
            holidays[f"{market}_holiday"] = (
                unchanged_prices | 
                weekends | 
                known_holidays | 
                suspicious_changes
            )
            
            # Clean up potential false positives
            holidays[f"{market}_holiday"] = self._clean_holiday_detection(
                holidays[f"{market}_holiday"],
                df.index
            )
        
        return holidays

    def _detect_known_holidays(self, dates: pd.DatetimeIndex, market: str) -> pd.Series:
        """Detects known holidays for a specific market."""
        is_holiday = pd.Series(False, index=dates)
        
        for date in dates:
            # Check universal holidays
            if (date.month, date.day) in self.universal_holidays:
                is_holiday[date] = True
                continue
            
            # Check market-specific holidays
            for month, day in self.market_holidays.get(market, []):
                if day < 0:
                    # Handle relative holidays (e.g., "last Monday")
                    if self._is_relative_holiday(date, month, day):
                        is_holiday[date] = True
                elif date.month == month and date.day == day:
                    is_holiday[date] = True
        
        return is_holiday

    def _is_relative_holiday(self, date: datetime, month: int, day: int) -> bool:
        """Checks if a date is a relative holiday (e.g., "third Monday")."""
        if date.month != month:
            return False
        
        if day < 0:  # Negative day means counting from the end of the month
            # Get the first day of next month and subtract one day
            if month == 12:
                next_month = date.replace(year=date.year + 1, month=1, day=1)
            else:
                next_month = date.replace(month=month + 1, day=1)
            last_day = next_month - timedelta(days=1)
            
            # Count backwards to find the nth last occurrence of each weekday
            current = last_day
            weekday_count = 1
            
            while current.month == month:
                if current.date() == date.date():
                    return weekday_count == abs(day)
                if current.weekday() == date.weekday():
                    weekday_count += 1
                current -= timedelta(days=1)
        
        return False

    def _clean_holiday_detection(self, holidays: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        """
        Cleans up holiday detection by:
        1. Removing isolated holidays (likely false positives)
        2. Adding missed holidays between consecutive holidays
        """
        cleaned = holidays.copy()
        
        # Remove isolated holidays
        isolated = (
            (cleaned != cleaned.shift(1)) & 
            (cleaned != cleaned.shift(-1)) & 
            cleaned
        )
        cleaned[isolated] = False
        
        # Fill gaps between consecutive holidays
        holiday_groups = cleaned.astype(int).diff()
        start_holidays = dates[holiday_groups == 1]
        end_holidays = dates[holiday_groups == -1]
        
        for start, end in zip(start_holidays, end_holidays):
            if (end - start).days <= 4:  # Fill gaps of up to 4 days
                cleaned[start:end] = True
        
        return cleaned

    def validate_market_data(self, df: pd.DataFrame) -> Tuple[bool, List[str], Dict[str, pd.Series]]:
        """
        Validates market data and detects holidays.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Tuple of (is_valid, list_of_issues, holiday_dict)
        """
        is_valid, issues = self.validate_data(df)
        holidays = self.detect_holidays(df)
        
        return is_valid, issues, holidays
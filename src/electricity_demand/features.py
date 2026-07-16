# src/electricity_demand/features.py

import pandas as pd
import numpy as np
from .config import *

class HolidayFeatures:
    """Generate holiday features for German public holidays."""

    @staticmethod
    def get_german_holidays(start_date, end_date):
        """
        Generate German public holiday indicators.
        Includes both fixed and variable date holidays.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame: Weekly holiday indicators
        """
        print("\n" + "="*70)
        print("HOLIDAY FEATURES")
        print("="*70)

        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        holidays = pd.DataFrame(index=dates)
        holidays['holiday'] = 0

        # Fixed date holidays (ignoring weekend adjustments for simplicity)
        fixed_holidays = [
            (1, 1),   # New Year's Day
            (5, 1),   # Labour Day
            (10, 3),  # German Unity Day
            (12, 25), # Christmas Day
            (12, 26)  # Boxing Day
        ]

        # Variable date holidays (approximated)
        for year in range(start_date.year, end_date.year + 1):
            # Simple approximation for Easter Sunday
            easter_date = HolidayFeatures._calculate_easter(year)

            # Good Friday (2 days before Easter)
            good_friday = easter_date - pd.Timedelta(days=2)

            # Easter Monday (1 day after Easter)
            easter_monday = easter_date + pd.Timedelta(days=1)

            # Ascension Day (39 days after Easter)
            ascension = easter_date + pd.Timedelta(days=39)

            # Whit Monday (50 days after Easter)
            whit_monday = easter_date + pd.Timedelta(days=50)

            # Add to holidays
            for date in [good_friday, easter_date, easter_monday, ascension, whit_monday]:
                if start_date <= date <= end_date:
                    if date in holidays.index:
                        holidays.loc[date, 'holiday'] = 1

        # Add fixed holidays
        for month, day in fixed_holidays:
            for year in range(start_date.year, end_date.year + 1):
                date = pd.Timestamp(f'{year}-{month:02d}-{day:02d}')
                if date in holidays.index:
                    holidays.loc[date, 'holiday'] = 1

        # Resample daily holidays to weekly sum
        weekly_holidays = holidays[['holiday']].resample('W').sum()
        weekly_holidays['has_holiday'] = (weekly_holidays['holiday'] > 0).astype(int)
        weekly_holidays['holiday_days'] = weekly_holidays['holiday']
        weekly_holidays = weekly_holidays.drop(columns='holiday')

        # Ensure the index is naive (no timezone)
        weekly_holidays.index = weekly_holidays.index.tz_localize(None)

        print(f"Generated {len(weekly_holidays)} weeks of holiday data")
        print(f"Holiday weeks: {weekly_holidays['has_holiday'].sum()} out of {len(weekly_holidays)} weeks")

        return weekly_holidays

    @staticmethod
    def _calculate_easter(year):
        """Calculate Easter Sunday date using the Anonymous Gregorian algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return pd.Timestamp(f'{year}-{month:02d}-{day:02d}')

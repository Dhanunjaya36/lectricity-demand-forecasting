# tests/test_features.py

import pandas as pd
from src.electricity_demand.features import HolidayFeatures

def test_holiday_generation():
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-12-31")
    holidays = HolidayFeatures.get_german_holidays(start, end)
    assert len(holidays) > 0
    assert 'has_holiday' in holidays.columns
    assert 'holiday_days' in holidays.columns
    # Check that New Year's Day is a holiday
    assert holidays.loc['2020-01-05']['has_holiday'] == 1  # week containing Jan 1

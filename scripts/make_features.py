# scripts/make_features.py

from src.electricity_demand.data import DataLoader
from src.electricity_demand.features import HolidayFeatures
from src.electricity_demand.config import DATA_PROCESSED
import os

def main():
    data = DataLoader.load_electricity_data()
    weekly = data['weekly']

    start = weekly.index[0]
    end = weekly.index[-1]
    holidays = HolidayFeatures.get_german_holidays(start, end)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    weekly.to_csv(os.path.join(DATA_PROCESSED, "weekly_load.csv"))
    holidays.to_csv(os.path.join(DATA_PROCESSED, "holidays.csv"))
    print("Features saved to", DATA_PROCESSED)

if __name__ == "__main__":
    main()

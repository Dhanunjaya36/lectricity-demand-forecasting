# scripts/download_data.py

from src.electricity_demand.data import DataLoader
from src.electricity_demand.config import DATA_RAW
import os

def main():
    print("Downloading electricity demand data...")
    data = DataLoader.load_electricity_data()

    os.makedirs(DATA_RAW, exist_ok=True)
    for freq, df in data.items():
        df.to_csv(os.path.join(DATA_RAW, f"{freq}_load.csv"))
        print(f"Saved {freq} data to {DATA_RAW}")

if __name__ == "__main__":
    main()

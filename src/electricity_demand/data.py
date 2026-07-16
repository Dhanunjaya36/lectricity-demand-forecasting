# src/electricity_demand/data.py

import pandas as pd
import requests
from .config import *

class DataLoader:
    """Handle data loading and preprocessing."""

    @staticmethod
    def load_electricity_data():
        """
        Download and preprocess German electricity demand data.

        Returns:
            dict: Dictionary containing hourly, daily, and weekly data
        """
        print("\n" + "="*70)
        print("DATA COLLECTION")
        print("="*70)

        # Download data
        url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

        print("Downloading electricity demand data...")
        df = pd.read_csv(url, index_col=0, parse_dates=True)

        # Identify German load columns
        de_load_cols = [col for col in df.columns if 'DE' in col and 'load' in col.lower()]

        if not de_load_cols:
            de_load_cols = [col for col in df.columns if col.startswith('DE_')]

        print(f"Found {len(de_load_cols)} German load columns")

        # Use first available load column
        load_col = de_load_cols[0]
        df_de = df[[load_col]].copy()
        df_de.columns = ['load']

        # Filter to 2015-2020
        df_de = df_de.loc['2015-01-01':'2020-10-31']
        df_de = df_de.dropna()

        # IMPORTANT: Remove timezone info to make it naive (matches electricity data)
        df_de.index = df_de.index.tz_localize(None)

        # Resample to different frequencies
        data = {
            'hourly': df_de.resample('h').mean(),
            'daily': df_de.resample('D').mean(),
            'weekly': df_de.resample('W').mean()
        }

        print(f"Data shapes:")
        print(f"  Hourly: {len(data['hourly'])} observations")
        print(f"  Daily: {len(data['daily'])} observations")
        print(f"  Weekly: {len(data['weekly'])} observations")

        return data

    @staticmethod
    def load_temperature_data(start_date, end_date, lat=52.52, lon=13.405):
        """
        Retrieve temperature data from Open-Meteo API.

        Args:
            start_date: Start date
            end_date: End date
            lat: Latitude (Berlin default)
            lon: Longitude (Berlin default)

        Returns:
            pd.Series: Weekly temperature data with naive datetime index
        """
        print("\nRetrieving temperature data...")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": "temperature_2m_mean",
            "timezone": "Europe/Berlin"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            dates = pd.to_datetime(data['daily']['time'])
            temps = data['daily']['temperature_2m_mean']

            temp_series = pd.Series(temps, index=dates, name='temperature')

            # IMPORTANT: Remove timezone info to make it naive
            temp_series.index = temp_series.index.tz_localize(None)

            # Resample to weekly
            weekly_temp = temp_series.resample('W').mean()

            print(f"Temperature data retrieved: {len(weekly_temp)} weeks")
            return weekly_temp

        except Exception as e:
            print(f"Warning: Could not retrieve temperature data: {e}")
            # Generate synthetic temperature with realistic pattern
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
            # Remove timezone from synthetic dates
            dates = dates.tz_localize(None)
            t = np.arange(len(dates))
            temp = 10 + 15 * np.sin(2 * np.pi * t / 52 - np.pi/2) + np.random.normal(0, 3, len(dates))
            return pd.Series(temp, index=dates, name='temperature')

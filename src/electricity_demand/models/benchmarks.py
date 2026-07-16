# src/electricity_demand/models/benchmarks.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..config import OUTPUT_FIGURES
import os

class BenchmarkModels:
    """Implement benchmark forecasting models."""

    @staticmethod
    def forecast(data, forecast_horizon=104):
        """
        Generate forecasts using benchmark models.

        Args:
            data: Weekly time series
            forecast_horizon: Number of weeks to forecast

        Returns:
            dict: Dictionary of forecasts and evaluation metrics
        """
        print("\n" + "="*70)
        print("BENCHMARK MODELS")
        print("="*70)

        # Split data
        train_data = data.iloc[:-forecast_horizon]
        test_data = data.iloc[-forecast_horizon:]

        print(f"Training period: {train_data.index[0].date()} to {train_data.index[-1].date()}")
        print(f"Test period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
        print(f"Forecast horizon: {forecast_horizon} weeks (2 years)")

        models = {}

        # 1. Mean model
        mean_value = train_data['load'].mean()
        mean_forecast = pd.Series([mean_value] * forecast_horizon, index=test_data.index)
        models['Mean'] = mean_forecast

        # 2. Naïve model
        last_value = train_data['load'].iloc[-1]
        naive_forecast = pd.Series([last_value] * forecast_horizon, index=test_data.index)
        models['Naïve'] = naive_forecast

        # 3. Seasonal Naïve (52 weeks = 1 year)
        seasonal_pattern = train_data['load'].iloc[-52:].values
        seasonal_forecast = np.tile(seasonal_pattern, forecast_horizon // 52 + 1)[:forecast_horizon]
        models['Seasonal Naïve'] = pd.Series(seasonal_forecast, index=test_data.index)

        # 4. Drift model
        drift = (train_data['load'].iloc[-1] - train_data['load'].iloc[0]) / len(train_data)
        drift_forecast = [train_data['load'].iloc[-1] + drift * i for i in range(1, forecast_horizon + 1)]
        models['Drift'] = pd.Series(drift_forecast, index=test_data.index)

        # Evaluate models
        results = {}
        print("\n" + "-"*50)
        print("MODEL PERFORMANCE")
        print("-"*50)

        for name, forecast in models.items():
            rmse = np.sqrt(mean_squared_error(test_data['load'], forecast))
            mae = mean_absolute_error(test_data['load'], forecast)
            mape = mean_absolute_percentage_error(test_data['load'], forecast) * 100

            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }

            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.2f} MW")
            print(f"  MAE:  {mae:.2f} MW")
            print(f"  MAPE: {mape:.2f}%")

        # Plot forecasts
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()

        for idx, (name, forecast) in enumerate(models.items()):
            context_train = train_data.iloc[-52:]

            axes[idx].plot(context_train.index, context_train['load'],
                          label='Training (last year)', alpha=0.7, linewidth=2)
            axes[idx].plot(test_data.index, test_data['load'],
                          label='Actual', alpha=0.7, linewidth=2)
            axes[idx].plot(test_data.index, forecast,
                          label='Forecast', linestyle='--', linewidth=2, color='red')
            axes[idx].set_title(f'{name} Model')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Demand (MW)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, 'benchmark_forecasts.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return models, results

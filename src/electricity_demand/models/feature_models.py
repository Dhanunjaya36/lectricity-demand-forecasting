# src/electricity_demand/models/feature_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from ..config import *
import os

class FeatureBasedModel:
    """Implement feature-based regression models."""

    @staticmethod
    def create_features(data, temperature, holidays=None, lags=[1, 2, 3, 4, 52]):
        data_copy = data.copy()
        temp_copy = temperature.copy()

        if data_copy.index.tz is not None:
            data_copy.index = data_copy.index.tz_localize(None)
        if temp_copy.index.tz is not None:
            temp_copy.index = temp_copy.index.tz_localize(None)

        combined = pd.DataFrame({
            'load': data_copy['load'],
            'temperature': temp_copy
        })

        if holidays is not None:
            holiday_copy = holidays.copy()
            if holiday_copy.index.tz is not None:
                holiday_copy.index = holiday_copy.index.tz_localize(None)
            combined = combined.join(holiday_copy[['has_holiday', 'holiday_days']], how='left')
            combined['has_holiday'] = combined['has_holiday'].fillna(0)
            combined['holiday_days'] = combined['holiday_days'].fillna(0)

        combined = combined.dropna()

        features = pd.DataFrame(index=combined.index)

        # Temporal features
        features['year'] = combined.index.year
        features['month'] = combined.index.month
        features['week'] = combined.index.isocalendar().week
        features['day_of_week'] = combined.index.dayofweek
        features['day_of_year'] = combined.index.dayofyear
        features['quarter'] = combined.index.quarter

        # Cyclical features
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        features['week_sin'] = np.sin(2 * np.pi * features['week'] / 52)
        features['week_cos'] = np.cos(2 * np.pi * features['week'] / 52)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # Temperature
        features['temperature'] = combined['temperature']

        # Holiday features
        if holidays is not None:
            features['has_holiday'] = combined['has_holiday']
            features['holiday_days'] = combined['holiday_days']

        # Lag features
        for lag in lags:
            features[f'lag_{lag}'] = combined['load'].shift(lag)

        # Rolling statistics
        for window in [4, 8, 13, 26, 52]:
            features[f'rolling_mean_{window}'] = combined['load'].rolling(window).mean()
            features[f'rolling_std_{window}'] = combined['load'].rolling(window).std()

        features = features.dropna()
        return features, combined.loc[features.index]

    @staticmethod
    def train_predict(data, temperature, holidays=None, forecast_horizon=104, model_type='random_forest'):
        print("\n" + "="*70)
        print(f"FEATURE-BASED MODEL - {model_type.upper()}")
        print("="*70)

        features, combined = FeatureBasedModel.create_features(data, temperature, holidays)

        if len(features) == 0:
            print("⚠ No features created. Returning NaN results.")
            metrics = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
            return None, None, metrics, None

        target = combined['load']
        train_size = len(features) - forecast_horizon
        if train_size <= 0:
            train_size = int(len(features) * 0.8)

        X_train = features.iloc[:train_size]
        X_test = features.iloc[train_size:]
        y_train = target.iloc[:train_size]
        y_test = target.iloc[train_size:]

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {X_train.columns.tolist()}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            tscv = TimeSeriesSplit(n_splits=3)
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(model, param_grid, cv=tscv,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            print(f"\nBest parameters: {grid_search.best_params_}")
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [2, 5]
            }
            tscv = TimeSeriesSplit(n_splits=3)
            model = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=tscv,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=-1, verbose=0)
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            print(f"\nBest parameters: {grid_search.best_params_}")

        predictions = best_model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100

        print("\nForecast Performance:")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MAE:  {mae:.2f} MW")
        print(f"  MAPE: {mape:.2f}%")

        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        print(feature_importance.head(10).to_string(index=False))

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax1 = axes[0]
        context_train = combined.iloc[max(0, train_size-104):train_size]
        ax1.plot(context_train.index, context_train['load'],
                label='Training', alpha=0.7, linewidth=2)
        ax1.plot(y_test.index, y_test,
                label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(y_test.index, predictions,
                label=f'{model_type.replace("_", " ").title()} Forecast',
                color='purple', linewidth=2)
        ax1.set_title(f'{model_type.replace("_", " ").title()} Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Demand (MW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        top_features = feature_importance.head(10)
        ax2.barh(top_features['feature'], top_features['importance'])
        ax2.set_title('Top 10 Feature Importances')
        ax2.set_xlabel('Importance')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, f'{model_type}_forecast.png'), dpi=300, bbox_inches='tight')
        plt.show()

        metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        return best_model, predictions, metrics, feature_importance

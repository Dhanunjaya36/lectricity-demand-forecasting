# src/electricity_demand/models/neural.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from ..config import *
import os
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    """
    LSTM model for hourly demand forecasting with recursive multi-step forecasting.
    """
    def __init__(self, lookback=None):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback = lookback or 72  # default 72 hours (3 days)

    def prepare_data(self, data, target_col='load'):
        """
        Prepare sequences from a DataFrame.
        Returns train/test split (80/20) for model training.
        """
        values = data[target_col].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)

        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i-self.lookback:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X).reshape(-1, self.lookback, 1)
        y = np.array(y)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train(self, data, epochs=10, batch_size=32, lookback=None):
        """
        Train the LSTM model on the given data.
        The data should be the training portion (e.g., all hours before the test horizon).
        """
        if lookback:
            self.lookback = lookback

        X_train, X_test, y_train, y_test = self.prepare_data(data, 'load')

        self.model = self.build_model((self.lookback, 1))

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        print(f"\nTraining LSTM with {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stopping]
        )

        # Evaluate on the test split (for monitoring only)
        pred_scaled = self.model.predict(X_test, verbose=0)
        predictions = self.scaler.inverse_transform(pred_scaled)
        actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        print(f"  (Test split) RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Small sample of predictions vs actual
        plot_len = min(1000, len(predictions))
        axes[0, 1].plot(actual[-plot_len:], label='Actual', alpha=0.7)
        axes[0, 1].plot(predictions[-plot_len:], label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Sample Predictions vs Actual')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Demand (MW)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        residuals = actual.flatten() - predictions.flatten()
        axes[1, 0].plot(residuals)
        axes[1, 0].axhline(0, color='red', linestyle='--')
        axes[1, 0].set_title('Residuals')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(0, color='red', linestyle='--')
        axes[1, 1].set_title('Distribution of Residuals')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, 'lstm_training.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return history

    def recursive_forecast(self, data, horizon, lookback=None):
        """
        Generate a multi-step recursive forecast.

        Args:
            data: The training DataFrame (used to initialise the last sequence).
            horizon: Number of hours to forecast (e.g., 17520).
            lookback: Lookback window (if None, uses self.lookback).

        Returns:
            pd.Series: Forecast with datetime index starting after the last data point.
        """
        if lookback:
            self.lookback = lookback

        # Use the last 'lookback' values from the training data as the initial seed
        last_sequence = data['load'].values[-self.lookback:].reshape(1, self.lookback, 1)
        # Scale the sequence using the already fitted scaler
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, self.lookback, 1)

        predictions_scaled = []
        for _ in range(horizon):
            next_scaled = self.model.predict(last_sequence_scaled, verbose=0)[0, 0]
            predictions_scaled.append(next_scaled)
            # Shift the sequence: drop first, append new prediction
            last_sequence_scaled = np.roll(last_sequence_scaled, -1, axis=1)
            last_sequence_scaled[0, -1, 0] = next_scaled

        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).ravel()

        # Create index (hourly frequency)
        last_date = data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='h')

        return pd.Series(predictions, index=forecast_index, name='LSTM_forecast')

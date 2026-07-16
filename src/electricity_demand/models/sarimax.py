# src/electricity_demand/models/sarimax.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from ..config import *
import os

class SARIMAModel:
    """Implement SARIMA modelling with comprehensive grid search."""

    @staticmethod
    def grid_search_full(data, p_range=range(0, 4), d_range=range(0, 2),
                         q_range=range(0, 4), P_range=range(0, 2),
                         D_range=range(0, 2), Q_range=range(0, 2),
                         seasonal_period=52, verbose=True):
        """
        Perform SARIMA grid search with progress updates.
        """
        print("\n" + "="*70)
        print("SARIMA GRID SEARCH")
        print("="*70)

        train_data = data['load'].values

        best_aic = np.inf
        best_bic = np.inf
        best_params = None
        best_model = None
        results = []

        # Build all parameter combinations
        param_combinations = []
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                param_combinations.append((p,d,q,P,D,Q))

        total = len(param_combinations)
        print(f"Testing {total} combinations...")

        success = 0
        failed = 0

        for idx, (p,d,q,P,D,Q) in enumerate(param_combinations):
            try:
                model = SARIMAX(train_data,
                               order=(p,d,q),
                               seasonal_order=(P,D,Q,seasonal_period),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                fitted = model.fit(disp=False, method='lbfgs', maxiter=500)
                aic = fitted.aic
                bic = fitted.bic
                results.append({'p':p,'d':d,'q':q,'P':P,'D':D,'Q':Q,'AIC':aic,'BIC':bic})
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_params = (p,d,q,P,D,Q)
                    best_model = fitted
                success += 1
            except:
                failed += 1

            # --- Progress update every 5 iterations ---
            if verbose and ((idx + 1) % 5 == 0 or idx + 1 == total):
                best_str = f"AIC={best_aic:.2f}" if best_params is not None else "None"
                print(f"Processed {idx+1}/{total} | S:{success} F:{failed} | Best: {best_str} {best_params if best_params else ''}", end='\r')

        print()  # newline after progress

        # If none succeeded, fallback
        if best_params is None:
            print("\n⚠ No models converged. Using fallback ARIMA(0,1,1).")
            try:
                model = SARIMAX(train_data, order=(0,1,1), seasonal_order=(0,0,0,0))
                best_model = model.fit(disp=False)
                best_params = (0,1,1,0,0,0)
                best_aic = best_model.aic
            except:
                try:
                    model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(0,0,0,0))
                    best_model = model.fit(disp=False)
                    best_params = (1,1,1,0,0,0)
                    best_aic = best_model.aic
                except:
                    print("CRITICAL: All models failed.")
                    return None, None, pd.DataFrame()

        results_df = pd.DataFrame(results).sort_values('AIC') if results else pd.DataFrame()
        print(f"\n✓ {success} succeeded, ✗ {failed} failed")
        print(f"Best: SARIMA{best_params[:3]}{best_params[3:]} AIC={best_aic:.2f}")
        return best_model, best_params, results_df

    @staticmethod
    def forecast(train_data, test_data, params, seasonal_period=52):
        if params is None:
            params = (1,1,1,0,0,0)
        p,d,q,P,D,Q = params
        print("\n" + "="*70)
        print("SARIMA FORECASTING")
        print("="*70)
        model = SARIMAX(train_data['load'],
                       order=(p,d,q),
                       seasonal_order=(P,D,Q,seasonal_period),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        try:
            fitted = model.fit(disp=False, method='lbfgs', maxiter=500)
        except:
            fitted = model.fit(disp=False, method='nm', maxiter=1000)
        print(fitted.summary())
        forecast = fitted.get_forecast(steps=len(test_data))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=0.05)

        rmse = np.sqrt(mean_squared_error(test_data['load'], forecast_mean))
        mae = mean_absolute_error(test_data['load'], forecast_mean)
        mape = mean_absolute_percentage_error(test_data['load'], forecast_mean)*100
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

        residuals = fitted.resid
        try:
            lb = acorr_ljungbox(residuals, lags=20, return_df=True)
            print(f"Ljung-Box p-value: {lb['lb_pvalue'].iloc[-1]:.4f}")
        except:
            pass

        # Diagnostic plots
        fig, axes = plt.subplots(2,2,figsize=(14,10))
        axes[0,0].plot(residuals)
        axes[0,0].axhline(0,color='red',linestyle='--')
        axes[0,0].set_title('Residuals')
        plot_acf(residuals, ax=axes[0,1], lags=40)
        stats.probplot(residuals, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot')
        axes[1,1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1,1].axvline(0,color='red',linestyle='--')
        axes[1,1].set_title('Histogram')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, 'sarima_diagnostics.png'))
        plt.show()

        # Forecast plot
        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(train_data.index[-104:], train_data['load'][-104:], label='Training')
        ax.plot(test_data.index, test_data['load'], label='Actual')
        ax.plot(test_data.index, forecast_mean, label='SARIMA Forecast', color='red')
        ax.fill_between(test_data.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1],
                        color='red', alpha=0.2, label='95% CI')
        ax.legend(); ax.grid(True); ax.set_title('SARIMA Forecast')
        plt.savefig(os.path.join(OUTPUT_FIGURES, 'sarima_forecast.png'))
        plt.show()

        metrics = {'RMSE':rmse, 'MAE':mae, 'MAPE':mape}
        return fitted, forecast_mean, forecast_ci, metrics


class SARIMAXModel:
    """Implement SARIMAX with exogenous variables including temperature and holidays."""

    @staticmethod
    def forecast(data, temperature, holidays=None, forecast_horizon=104, sarima_params=None):
        print("\n" + "="*70)
        print("SARIMAX WITH TEMPERATURE AND HOLIDAY COVARIATES")
        print("="*70)

        if sarima_params:
            p, d, q, P, D, Q = sarima_params
        else:
            p, d, q, P, D, Q = (1, 1, 1, 1, 0, 1)

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

        if len(combined) == 0:
            print("⚠ Warning: No data after merging! Using fallback...")
            common_idx = data_copy.index.intersection(temp_copy.index)
            if len(common_idx) == 0:
                print("No common indices found. Aligning by interpolation...")
                temp_reindexed = temp_copy.reindex(data_copy.index, method='ffill')
                combined = pd.DataFrame({
                    'load': data_copy['load'],
                    'temperature': temp_reindexed
                }).dropna()

        train_data = combined.iloc[:-forecast_horizon]
        test_data = combined.iloc[-forecast_horizon:]

        print(f"Combined data shape: {combined.shape}")
        print(f"Training shape: {train_data.shape}")
        print(f"Test shape: {test_data.shape}")

        exog_cols = ['temperature']
        if holidays is not None:
            exog_cols.extend(['has_holiday', 'holiday_days'])

        try:
            model = SARIMAX(train_data['load'],
                           exog=train_data[exog_cols],
                           order=(p, d, q),
                           seasonal_order=(P, D, Q, 52),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            fitted = model.fit(disp=False, method='lbfgs', maxiter=500)
        except Exception as e:
            print(f"SARIMAX fitting failed: {e}")
            print("Using alternative SARIMAX fitting...")
            try:
                model = SARIMAX(train_data['load'],
                               exog=train_data[['temperature']],
                               order=(1, 1, 1),
                               seasonal_order=(0, 0, 0, 52))
                fitted = model.fit(disp=False, method='nm', maxiter=1000)
            except:
                print("⚠ SARIMAX completely failed. Returning NaN results.")
                metrics = {
                    'RMSE': np.nan,
                    'MAE': np.nan,
                    'MAPE': np.nan,
                    'temperature_coef': np.nan,
                    'holiday_coef': np.nan
                }
                return None, None, metrics

        print("\nModel Summary:")
        print(fitted.summary())

        try:
            forecast = fitted.get_forecast(steps=forecast_horizon,
                                         exog=test_data[exog_cols])
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
        except Exception as e:
            print(f"Forecast generation failed: {e}")
            forecast = fitted.get_forecast(steps=forecast_horizon)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)

        rmse = np.sqrt(mean_squared_error(test_data['load'], forecast_mean))
        mae = mean_absolute_error(test_data['load'], forecast_mean)
        mape = mean_absolute_percentage_error(test_data['load'], forecast_mean) * 100

        print("\nForecast Performance:")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MAE:  {mae:.2f} MW")
        print(f"  MAPE: {mape:.2f}%")

        temp_coef = fitted.params.get('temperature', np.nan)
        holiday_coef = fitted.params.get('has_holiday', np.nan)
        print(f"  Temperature coefficient: {temp_coef:.4f}")
        if holidays is not None and not np.isnan(holiday_coef):
            print(f"  Holiday coefficient: {holiday_coef:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))

        context_train = train_data.iloc[-104:]
        ax.plot(context_train.index, context_train['load'],
                label='Training', alpha=0.7, linewidth=2)
        ax.plot(test_data.index, test_data['load'],
                label='Actual', alpha=0.7, linewidth=2)
        ax.plot(test_data.index, forecast_mean,
                label='SARIMAX Forecast', color='green', linewidth=2)
        ax.fill_between(test_data.index,
                       forecast_ci.iloc[:, 0],
                       forecast_ci.iloc[:, 1],
                       color='green', alpha=0.2, label='95% CI')

        ax2 = ax.twinx()
        ax2.plot(test_data.index, test_data['temperature'],
                color='orange', alpha=0.5, linestyle='--', linewidth=2)
        ax2.set_ylabel('Temperature (°C)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        if holidays is not None and 'has_holiday' in test_data.columns:
            holiday_dates = test_data[test_data['has_holiday'] == 1].index
            for date in holiday_dates:
                ax.axvline(x=date, color='purple', alpha=0.1, linewidth=1)

        ax.set_title('SARIMAX Forecast with Temperature and Holiday Covariates', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand (MW)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, 'sarimax_forecast.png'), dpi=300, bbox_inches='tight')
        plt.show()

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'temperature_coef': temp_coef,
            'holiday_coef': holiday_coef
        }

        return fitted, forecast_mean, metrics

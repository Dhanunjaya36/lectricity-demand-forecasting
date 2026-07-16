# src/electricity_demand/pipeline.py

from .data import DataLoader
from .features import HolidayFeatures
from .eda import EDA
from .models.benchmarks import BenchmarkModels
from .models.sarimax import SARIMAModel, SARIMAXModel
from .models.feature_models import FeatureBasedModel
from .models.neural import LSTMModel
from .models.bayesian import BayesianModel
from .evaluation import ModelComparison, ResidualDiagnostics, DataLeakagePrevention, DiscussionQuestions
from .config import *
import os

def run_pipeline():
    print("\n" + "="*70)
    print("TIME SERIES MODELLING CASE STUDY")
    print("German Electricity Demand Forecasting")
    print("="*70)
    print("\nThis script implements ALL assignment tasks.")
    print("Results will be saved in the 'outputs/' directory.\n")

    # ==================================================================
    # TASK 1: DATA COLLECTION AND EDA
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 1: DATA COLLECTION AND EXPLORATORY DATA ANALYSIS")
    print("="*70)

    data_loader = DataLoader()
    data = data_loader.load_electricity_data()

    eda = EDA()
    eda.comprehensive_eda(data['weekly'], 'weekly')
    eda.comprehensive_eda(data['daily'], 'daily')

    # ==================================================================
    # TASK 2: BENCHMARK MODELS
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 2: BENCHMARK FORECASTING MODELS")
    print("="*70)

    benchmark_models = BenchmarkModels()
    benchmark_forecasts, benchmark_results = benchmark_models.forecast(data['weekly'])

    # ==================================================================
    # TASK 3: SARIMA MODEL
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 3: SARIMA MODEL WITH GRID SEARCH")
    print("="*70)

    sarima = SARIMAModel()
    best_model, best_params, grid_results = sarima.grid_search_full(
        data['weekly'],
        p_range=range(0, 4),
        d_range=range(0, 2),
        q_range=range(0, 4),
        P_range=range(0, 2),
        D_range=range(0, 2),
        Q_range=range(0, 2),
        seasonal_period=52          # fixed: now an integer
    )

    train_data = data['weekly'].iloc[:-104]
    test_data = data['weekly'].iloc[-104:]

    sarima_model, sarima_forecast, sarima_ci, sarima_results = sarima.forecast(
        train_data, test_data, best_params
    )

    residual_diag = ResidualDiagnostics()
    residual_diag.full_diagnostics(
        sarima_model, sarima_model.fittedvalues, sarima_model.resid, "SARIMA"
    )

    # ==================================================================
    # TASK 4: SARIMAX WITH TEMPERATURE AND HOLIDAYS
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 4: SARIMAX WITH EXOGENOUS VARIABLES")
    print("="*70)

    temperature = data_loader.load_temperature_data(
        data['weekly'].index[0],
        data['weekly'].index[-1]
    )
    holidays = HolidayFeatures.get_german_holidays(
        data['weekly'].index[0],
        data['weekly'].index[-1]
    )
    sarimax = SARIMAXModel()
    sarimax_model, sarimax_forecast, sarimax_results = sarimax.forecast(
        data['weekly'],
        temperature,
        holidays=holidays,
        sarima_params=best_params
    )

    # ==================================================================
    # TASK 5: FEATURE-BASED MODELS
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 5: FEATURE-BASED MACHINE LEARNING MODELS")
    print("="*70)

    feature_model = FeatureBasedModel()
    try:
        rf_model, rf_forecast, rf_results, rf_importance = feature_model.train_predict(
            data['weekly'], temperature, holidays=holidays, model_type='random_forest'
        )
    except Exception as e:
        print(f"Random Forest failed: {e}")
        rf_results = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        rf_forecast = None

    try:
        gb_model, gb_forecast, gb_results, gb_importance = feature_model.train_predict(
            data['weekly'], temperature, holidays=holidays, model_type='gradient_boosting'
        )
    except Exception as e:
        print(f"Gradient Boosting failed: {e}")
        gb_results = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        gb_forecast = None

    # ==================================================================
    # TASK 6: LSTM NEURAL NETWORK (HOURLY DATA)
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 6: LSTM NEURAL NETWORK (HOURLY DATA)")
    print("="*70)

    try:
        # Define the test horizon: last 2 years = 17520 hours
        TEST_HORIZON = 104 * 24 * 7  # 17520

        hourly_data = data['hourly']
        train_hourly = hourly_data.iloc[:-TEST_HORIZON]
        test_hourly = hourly_data.iloc[-TEST_HORIZON:]

        print(f"Training on {len(train_hourly)} hours, forecasting {TEST_HORIZON} hours.")

        # Create and train the LSTM
        lstm = LSTMModel(lookback=72)
        lstm.train(train_hourly, epochs=10, batch_size=32, lookback=72)

        # Generate recursive forecast for the test horizon
        lstm_forecast = lstm.recursive_forecast(
            data=train_hourly,
            horizon=TEST_HORIZON,
            lookback=72
        )

        # Evaluate on actual test data
        actual = test_hourly['load'].values
        predicted = lstm_forecast.values
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        lstm_results = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
        print(f"\nLSTM Performance (recursive forecast, horizon={TEST_HORIZON} hours):")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MAE:  {mae:.2f} MW")
        print(f"  MAPE: {mape:.2f}%")

        # Save the forecast (optional)
        lstm_forecast.to_csv(os.path.join(OUTPUT_FORECASTS, 'lstm_hourly_forecast.csv'))

    except Exception as e:
        print(f"LSTM failed: {e}")
        import traceback
        traceback.print_exc()
        lstm_results = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
        lstm_forecast = None

    # ==================================================================
    # TASK 7: BAYESIAN MODEL (OPTIONAL)
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 7: BAYESIAN MODEL (OPTIONAL)")
    print("="*70)

    try:
        bayesian = BayesianModel()
        bayes_model, bayes_X = bayesian.build_model(data['weekly'], temperature, holidays)
        bayes_results = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}
    except Exception as e:
        print(f"Bayesian model skipped: {e}")
        bayes_results = {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

    # ==================================================================
    # TASK 8: MODEL COMPARISON AND EVALUATION
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 8: MODEL COMPARISON AND EVALUATION")
    print("="*70)

    all_results = {
        'Mean': benchmark_results['Mean'],
        'Naïve': benchmark_results['Naïve'],
        'Seasonal Naïve': benchmark_results['Seasonal Naïve'],
        'Drift': benchmark_results['Drift'],
        'SARIMA': sarima_results,
        'SARIMAX': sarimax_results,
        'Random Forest': rf_results,
        'Gradient Boosting': gb_results,
        'LSTM': lstm_results,
        'Bayesian': bayes_results
    }
    comparison = ModelComparison()
    comparison_df = comparison.compare_models(all_results)

    # ==================================================================
    # TASK 9: DISCUSSION QUESTIONS
    # ==================================================================
    print("\n" + "="*70)
    print("TASK 9: DISCUSSION QUESTIONS")
    print("="*70)

    discussion = DiscussionQuestions()
    discussion_answers = discussion.answer_questions(comparison_df)

    # ==================================================================
    # SAVE ALL OUTPUTS
    # ==================================================================
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    comparison_df.to_csv(os.path.join(OUTPUT_METRICS, 'model_comparison.csv'))
    if grid_results is not None:
        grid_results.to_csv(os.path.join(OUTPUT_METRICS, 'sarima_grid_results.csv'))
    with open(os.path.join(OUTPUT_METRICS, 'best_params.txt'), 'w') as f:
        f.write(f"Best SARIMA Parameters: {best_params}\n")

    # Save weekly forecasts
    forecast_df = pd.DataFrame({'actual': test_data['load']})
    if sarima_forecast is not None:
        forecast_df['sarima'] = sarima_forecast
    if sarimax_forecast is not None:
        forecast_df['sarimax'] = sarimax_forecast
    if rf_forecast is not None:
        forecast_df['random_forest'] = rf_forecast
    if gb_forecast is not None:
        forecast_df['gradient_boosting'] = gb_forecast
    forecast_df.to_csv(os.path.join(OUTPUT_FORECASTS, 'weekly_forecasts.csv'))

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll figures saved in: {OUTPUT_FIGURES}")
    print(f"All metrics saved in: {OUTPUT_METRICS}")
    print(f"Forecasts saved in: {OUTPUT_FORECASTS}")

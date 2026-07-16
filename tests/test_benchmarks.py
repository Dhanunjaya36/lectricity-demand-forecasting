# tests/test_benchmarks.py

import pandas as pd
import numpy as np
from src.electricity_demand.models.benchmarks import BenchmarkModels

def test_benchmark_forecast_length():
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    data = pd.DataFrame({'load': np.random.randn(200)}, index=dates)
    models, results = BenchmarkModels.forecast(data, forecast_horizon=52)
    for name, fc in models.items():
        assert len(fc) == 52
        assert isinstance(fc, pd.Series)

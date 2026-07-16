# tests/test_evaluation.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

import numpy as np
from src.electricity_demand.evaluation import ModelComparison

def test_comparison_structure():
    dummy_results = {
        'ModelA': {'RMSE': 1.0, 'MAE': 0.8, 'MAPE': 5.0},
        'ModelB': {'RMSE': 1.2, 'MAE': 0.9, 'MAPE': 6.0}
    }
    df = ModelComparison.compare_models(dummy_results, plot=False)
    assert 'RMSE' in df.columns
    assert 'MAE' in df.columns
    assert 'MAPE' in df.columns
    assert df.loc['ModelA', 'RMSE'] == 1.0

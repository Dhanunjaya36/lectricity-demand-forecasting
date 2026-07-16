# src/electricity_demand/models/bayesian.py

import numpy as np
from ..config import *

class BayesianModel:
    """Implement a Bayesian time series model using PyMC."""

    @staticmethod
    def build_model(data, temperature=None, holidays=None, seasonality=52):
        print("\n" + "="*70)
        print("BAYESIAN MODEL")
        print("="*70)

        try:
            import pymc as pm
            import arviz as az
        except ImportError:
            print("⚠ PyMC not installed. Skipping Bayesian model.")
            print("  To install: pip install pymc arviz")
            return None, None

        y = data['load'].values
        t = np.arange(len(y))

        n_fourier = 4
        fourier_features = []
        for k in range(1, n_fourier + 1):
            fourier_features.append(np.sin(2 * np.pi * k * t / seasonality))
            fourier_features.append(np.cos(2 * np.pi * k * t / seasonality))

        X = np.column_stack([np.ones_like(y)] + fourier_features)
        feature_names = ['intercept'] + [f'sin_{k}' for k in range(1, n_fourier + 1)] + [f'cos_{k}' for k in range(1, n_fourier + 1)]

        if temperature is not None:
            temp_aligned = temperature.reindex(data.index).values
            if not np.all(np.isnan(temp_aligned)):
                X = np.column_stack([X, temp_aligned])
                feature_names.append('temperature')

        if holidays is not None:
            holiday_aligned = holidays['has_holiday'].reindex(data.index).fillna(0).values
            X = np.column_stack([X, holiday_aligned])
            feature_names.append('has_holiday')

        print(f"Building Bayesian model with {len(feature_names)} features")

        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = pm.math.dot(X, beta)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        print("✓ Bayesian model built successfully")
        print(f"  Features: {feature_names}")

        return model, X

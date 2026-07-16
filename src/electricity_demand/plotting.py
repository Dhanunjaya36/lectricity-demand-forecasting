# src/electricity_demand/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_forecast(train, test, forecast, title="Forecast", save_path=None):
    """Plot a single forecast against training and test data."""
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Training', alpha=0.7)
    plt.plot(test.index, test, label='Actual', alpha=0.7)
    plt.plot(test.index, forecast, label='Forecast', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(residuals, title="Residuals", save_path=None):
    """Plot residuals over time and their distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(residuals)
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_title('Residuals over time')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Residual')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_title('Distribution of residuals')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_model_comparison(models_dict, test_actual, save_path=None):
    """Plot multiple forecasts on the same axis for comparison."""
    plt.figure(figsize=(14, 7))
    plt.plot(test_actual.index, test_actual, label='Actual', linewidth=2, color='black')
    for name, forecast in models_dict.items():
        plt.plot(test_actual.index, forecast, label=name, linestyle='--')
    plt.title('Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Demand (MW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

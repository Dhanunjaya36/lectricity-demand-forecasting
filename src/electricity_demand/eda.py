# src/electricity_demand/eda.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from .config import OUTPUT_FIGURES
import os

class EDA:
    """Perform exploratory data analysis."""

    @staticmethod
    def comprehensive_eda(data, frequency='weekly'):
        print("\n" + "="*70)
        print(f"EXPLORATORY DATA ANALYSIS - {frequency.upper()}")
        print("="*70)

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # 1. Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data.index, data['load'], linewidth=1)
        ax1.set_title(f'{frequency.capitalize()} Electricity Demand (2015-2020)', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Demand (MW)')
        ax1.grid(True, alpha=0.3)

        # 2. Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(data['load'], bins=40, edgecolor='black', alpha=0.7)
        ax2.axvline(data['load'].mean(), color='red', linestyle='--', label=f'Mean: {data["load"].mean():.0f}')
        ax2.set_title('Distribution of Demand')
        ax2.set_xlabel('Demand (MW)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Box plot by year
        ax3 = fig.add_subplot(gs[1, 1])
        data_by_year = data.groupby(data.index.year)
        years = [str(y) for y in data_by_year.groups.keys()]
        box_data = [group['load'].values for _, group in data_by_year]
        bp = ax3.boxplot(box_data, patch_artist=True)
        ax3.set_xticklabels(years)
        ax3.set_title('Annual Distribution')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Demand (MW)')
        ax3.grid(True, alpha=0.3)

        # 4. Box plot by month
        ax4 = fig.add_subplot(gs[1, 2])
        data_by_month = data.groupby(data.index.month)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        box_data_month = [group['load'].values for _, group in data_by_month]
        ax4.boxplot(box_data_month, patch_artist=True)
        ax4.set_xticklabels(months)
        ax4.set_title('Monthly Distribution')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Demand (MW)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Weekly pattern (if daily or hourly)
        if frequency in ['daily', 'hourly']:
            ax5 = fig.add_subplot(gs[2, 0])
            dow = data.groupby(data.index.dayofweek).mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax5.bar(days, dow['load'])
            ax5.set_title('Weekly Pattern')
            ax5.set_xlabel('Day of Week')
            ax5.set_ylabel('Mean Demand (MW)')
            ax5.grid(True, alpha=0.3)

        # 6. Seasonal decomposition
        if len(data) > 52:
            ax6 = fig.add_subplot(gs[2, 1])
            ax7 = fig.add_subplot(gs[2, 2])

            try:
                period = 52 if frequency == 'weekly' else 365 if frequency == 'daily' else 24*7
                decomp = seasonal_decompose(data['load'], model='additive', period=period)

                ax6.plot(decomp.seasonal)
                ax6.set_title('Seasonal Component')
                ax6.set_xlabel('Date')
                ax6.set_ylabel('Seasonal')
                ax6.grid(True, alpha=0.3)

                ax7.plot(decomp.resid)
                ax7.set_title('Residual Component')
                ax7.set_xlabel('Date')
                ax7.set_ylabel('Residual')
                ax7.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Could not perform decomposition: {e}")
                ax6.text(0.5, 0.5, 'Decomposition Failed',
                        ha='center', va='center', transform=ax6.transAxes)
                ax7.text(0.5, 0.5, 'Insufficient Data',
                        ha='center', va='center', transform=ax7.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, f'eda_{frequency}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # Statistical summary
        print("\nStatistical Summary:")
        print(data['load'].describe())

        # Stationarity tests
        EDA.test_stationarity(data['load'], frequency)

    @staticmethod
    def test_stationarity(series, frequency='weekly'):
        print("\n" + "-"*50)
        print(f"STATIONARITY TESTS - {frequency.upper()}")
        print("-"*50)

        adf_result = adfuller(series, autolag='AIC')
        print("\nADF Test (Augmented Dickey-Fuller):")
        print(f"  Statistic: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Critical values:")
        for key, value in adf_result[4].items():
            print(f"    {key}: {value:.6f}")

        kpss_result = kpss(series, regression='c', nlags='auto')
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  Critical values:")
        for key, value in kpss_result[3].items():
            print(f"    {key}: {value:.6f}")

        is_stationary_adf = adf_result[1] <= 0.05
        is_stationary_kpss = kpss_result[1] >= 0.05

        print(f"\nInterpretation:")
        print(f"  ADF: {'✓ Series is stationary' if is_stationary_adf else '✗ Series is non-stationary'}")
        print(f"  KPSS: {'✓ Series is stationary' if is_stationary_kpss else '✗ Series is non-stationary'}")

        if is_stationary_adf and is_stationary_kpss:
            print("  ✓ Both tests confirm stationarity")
        elif not is_stationary_adf and not is_stationary_kpss:
            print("  ✗ Both tests indicate non-stationarity")
        else:
            print("  ⚠ Mixed results - series may be trend-stationary or have structural breaks")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        plot_acf(series, ax=ax1, lags=40, title=f'ACF - {frequency.capitalize()} Data')
        plot_pacf(series, ax=ax2, lags=40, title=f'PACF - {frequency.capitalize()} Data')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FIGURES, f'stationarity_{frequency}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'adf_pvalue': adf_result[1],
            'kpss_pvalue': kpss_result[1],
            'is_stationary': is_stationary_adf and is_stationary_kpss
        }

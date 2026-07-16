# src/electricity_demand/config.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# Machine learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Additional
import requests
from scipy import stats

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# Output paths – these are used throughout the code
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
OUTPUT_FIGURES = os.path.join(OUTPUT_DIR, 'figures')
OUTPUT_FORECASTS = os.path.join(OUTPUT_DIR, 'forecasts')
OUTPUT_METRICS = os.path.join(OUTPUT_DIR, 'metrics')
OUTPUT_MODELS = os.path.join(OUTPUT_DIR, 'model_objects')

# Create all output directories if they don't exist
for d in [OUTPUT_FIGURES, OUTPUT_FORECASTS, OUTPUT_METRICS, OUTPUT_MODELS]:
    os.makedirs(d, exist_ok=True)

# Data paths (optional)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

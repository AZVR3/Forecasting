import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import pmdarima as pm
from pmdarima.model_selection import train_test_split
print(f"Using pmdarima {pm.__version__}")
# Using pmdarima 1.5.2

df = pd.read_csv('south-korea-gathered-data.csv') 
print(df.head())

# Determine the size of the training set
dataSize = len(df)
train_size = int(0.7 * dataSize)

# Split the data into training and testing sets
y_train = df['New_cases'][:train_size]
y_test = df['New_cases'][train_size:]

from pmdarima.utils import tsdisplay
from pmdarima.preprocessing import LogEndogTransformer

y_train_log, _ = LogEndogTransformer(lmbda=1e-6).fit_transform(y_train)
tsdisplay(y_train_log, lag_max=100)

from scipy.stats import normaltest
print(normaltest(y_train_log)[1])

from pmdarima.pipeline import Pipeline

fit3 = Pipeline([
    ('log', LogEndogTransformer(lmbda=1e-6)),
    ('arima', pm.AutoARIMA(trace=True,
                         suppress_warnings=True,
                         m=7, # set the seasonal period
                         seasonal=True, # enable the seasonal component
                         seasonal_test='ocsb', # use the OCSB test to determine D
                         ))
])

fit3.fit(y_train)
print(fit3.summary())

from sklearn.metrics import mean_squared_error as mse

def plot_forecasts(forecasts, title, figsize=(8, 12)):
    x = np.arange(y_train.shape[0] + forecasts.shape[0])

    fig, axes = plt.subplots(2, 1, sharex=False, figsize=figsize)

    # Plot the forecasts
    axes[0].plot(x[:y_train.shape[0]], y_train, c='b')
    axes[0].plot(x[y_train.shape[0]:], forecasts, c='g')
    axes[0].set_xlabel(f'New Cases (RMSE={np.sqrt(mse(y_test, forecasts)):.3f})')
    axes[0].set_title(title)

    # Plot the residuals
    resid = y_test - forecasts
    _, p = normaltest(resid)
    axes[1].hist(resid, bins=15)
    axes[1].axvline(0, linestyle='--', c='r')
    axes[1].set_title(f'Residuals (p={p:.3f})')

    plt.tight_layout()
    plt.show()

# Generate forecasts
forecasts_log = fit3.predict(y_test.shape[0])

# Plot the forecasts
plot_forecasts(forecasts_log, title='Log transformed ARIMA')

# Import the error functions
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import median_absolute_error as mdae
from pmdarima.metrics import smape

# Calculate the errors
mae_value_log = mae(y_test, forecasts_log)
mdape_value_log = mdae(y_test, forecasts_log) / np.median(y_test)
smape_value_log = smape(y_test, forecasts_log)
mape_value_log = mape(y_test, forecasts_log)
rmse_value_log = np.sqrt(mse(y_test, forecasts_log))

# Print the errors
print(f'MAE: {mae_value_log:.3f}')
print(f'MdAPE: {mdape_value_log:.3f}')
print(f'SMAPE: {smape_value_log:.3f}')
print(f'MAPE: {mape_value_log:.3f}')
print(f'RMSE: {rmse_value_log:.3f}')
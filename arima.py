import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import pmdarima as pm
from pmdarima.datasets import load_sunspots
from pmdarima.model_selection import train_test_split
print(f"Using pmdarima {pm.__version__}")
# Using pmdarima 1.5.2

df = pd.read_csv('south-korea-new-cases.csv') 
print(df.head())

# Determine the size of the training set
dataSize = len(df)
train_size = int(0.8 * dataSize)

# Split the data into training and testing sets
y_train = df['new_cases'][:train_size]
y_test = df['new_cases'][train_size:]

from pmdarima.utils import tsdisplay

tsdisplay(y_train, lag_max=100)

# ----------------------------------------------------------------

from pmdarima.preprocessing import BoxCoxEndogTransformer

y_train_bc, _ = BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(y_train)
tsdisplay(y_train_bc, lag_max=100)

# ----------------------------------------------------------------

from scipy.stats import normaltest
print(normaltest(y_train_bc)[1])
## New cases
# y_train: 5.941295417681244e-184
# y_train_bc: 0.000991565134002783
# y_train_log: 1.5127623302358342e-122

## Total cases
# y_train: 3.3936067176900496e-102
# y_train_bc: 3.0522077068491763e-06
# y_train_log: 2.273823980094518e-145

# ----------------------------------------------------------------

from pmdarima.pipeline import Pipeline

fit2 = Pipeline([
    ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),
    ('arima', pm.AutoARIMA(trace=True,
                           suppress_warnings=True,
                           m=12))
])

fit2.fit(y_train)
print(fit2.summary())

from sklearn.metrics import mean_squared_error as mse

# ----------------------------------------------------------------

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

# Generate forecasts # Added this line
forecasts = fit2.predict(y_test.shape[0]) # Added this line

# Plot the forecasts # Added this line
plot_forecasts(forecasts, title='Box-Cox transformed ARIMA') # Added this line
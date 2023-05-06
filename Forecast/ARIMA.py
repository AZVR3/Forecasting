# Import libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
import plotly.express as px
import itertools
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('south-korea-new-cases.csv', parse_dates=['Date_reported'], index_col='Date_reported')

# Filter data by date range
df = df.loc['2022-01-01':'2023-03-05']

# Add frequency information to the date index
df.index = pd.DatetimeIndex(df.index.values, freq='D')

# Define a list of ratios
ratios = [0.8, 0.7, 0.6]

# Loop over the ratios
for ratio in ratios:
    # Split data into training and forecasting sets
    train_size = int(len(df) * ratio)
    train = df.iloc[:train_size]
    forecast = df.iloc[train_size:]

    # Perform sixth order difference on training set
    train_diff = train.diff(6).dropna()

    # Choose p and q using grid search
    p = range(0, 5)
    q = range(0, 5)
    d = 6
    pdq = [(x[0], d, x[1]) for x in list(itertools.product(p, q))]
    aic = []
    best_aic = np.inf
    best_param = None
    for param in pdq:
        try:
            mod = ARIMA(train_diff['New_cases'], order=param)
            results = mod.fit()
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
            aic.append(results.aic)
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
        except ValueError as ve:
            print('ARIMA{} - ValueError: {}'.format(param, ve))
        except np.linalg.LinAlgError as lae:
            print('ARIMA{} - LinAlgError: {}'.format(param, lae))
        except Exception as e:
            print('ARIMA{} - Other Exception: {}'.format(param, e))

    # Print the best parameters for each ratio
    print('Best parameters for {}:{} ratio are {}'.format(int(ratio * 100), int((1 - ratio) * 100), best_param))

    # Define ARIMA model with best p and q
    mod = ARIMA(train_diff['New_cases'], order=best_param)

    # Fit model using default configuration
    results = mod.fit()

    # Make forecasts for forecasting set length
    forecasts = results.get_forecast(steps=len(forecast))

    # Get forecast mean and confidence intervals
    mean_forecast = forecasts.predicted_mean
    conf_int = forecasts.conf_int(alpha=0.05)

    # Plot forecasts along with original and differenced series for each ratio
    fig = px.line(df, x=df.index, y='New_cases',
                  title='South Korea New Cases with ARIMA Forecasts ({}:{} ratio)'.format(int(ratio * 100),
                                                                                          int((1 - ratio) * 100)))
    fig.add_scatter(x=train_diff.index, y=train_diff['New_cases'], name='Differenced Series')
    fig.add_scatter(x=mean_forecast.index, y=mean_forecast.values, name='Forecast')
    fig.add_scatter(x=conf_int.index, y=conf_int.iloc[:, 0], name='Lower CI', showlegend=False)
    fig.add_scatter(x=conf_int.index, y=conf_int.iloc[:, 1], name='Upper CI', showlegend=False)
    fig.show()

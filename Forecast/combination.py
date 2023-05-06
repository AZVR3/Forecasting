# Import libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss

# Read the data
df = pd.read_csv("south-korea-new-cases.csv")

# Calculate the first, second, third, fourth, fifth and sixth difference
df['First_diff'] = df['New_cases'].diff()
df['Second_diff'] = df['First_diff'].diff()
df['Third_diff'] = df['Second_diff'].diff()
df['Fourth_diff'] = df['Third_diff'].diff()
df['Fifth_diff'] = df['Fourth_diff'].diff()
df['Sixth_diff'] = df['Fifth_diff'].diff()

# Define a function to perform and print ADF and KPSS tests
def adf_kpss_test(series, name):
    print(f'Performing ADF Test on {name}')
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print(f'Performing KPSS Test on {name}')
    result = kpss(series.dropna())
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'\t{key}: {value}')
    print('\n')

# Perform ADF and KPSS tests on each series
adf_kpss_test(df['New_cases'], 'New cases')
adf_kpss_test(df['First_diff'], 'First difference')
adf_kpss_test(df['Second_diff'], 'Second difference')
adf_kpss_test(df['Third_diff'], 'Third difference')
adf_kpss_test(df['Fourth_diff'], 'Fourth difference')
adf_kpss_test(df['Fifth_diff'], 'Fifth difference')
adf_kpss_test(df['Sixth_diff'], 'Sixth difference')

# Create a figure with seven subplots
fig = make_subplots(rows=7, cols=1)

# Add traces for each subplot
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['New_cases'], name='New cases'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['First_diff'], name='First difference'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Second_diff'], name='Second difference'), row=3, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Third_diff'], name='Third difference'), row=4, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Fourth_diff'], name='Fourth difference'), row=5, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Fifth_diff'], name='Fifth difference'), row=6, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Sixth_diff'], name='Sixth difference'), row=7, col=1)

# Update the layout and title
fig.update_layout(title='Total Cases and Differences for South Korea')
fig.show()
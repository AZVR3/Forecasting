# Import libraries
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss

# Read the data
df = pd.read_csv("south-korea-new-cases.csv")

# Calculate the first, second and third difference
df['First_diff'] = df['New_cases'].diff()
df['Second_diff'] = df['First_diff'].diff()
df['Third_diff'] = df['Second_diff'].diff()


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

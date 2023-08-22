import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import pmdarima as pm
print(f"Using pmdarima {pm.__version__}")

df = pd.read_csv('south-korea-gathered-data.csv') 
print(df.head())

dataSize = len(df)
train_size = int(0.8 * dataSize)

y_train = df['New_cases'][:train_size]
y_test = df['New_cases'][train_size:]

from pmdarima.arima import ndiffs

kpss_diffs = ndiffs(y_train, alpha=0.01, test='kpss', max_d=100)
adf_diffs = ndiffs(y_train, alpha=0.01, test='adf', max_d=100)
pp_diffs = ndiffs(y_test, alpha=0.01, test='pp', max_d=100)
n_diffs = max(adf_diffs, kpss_diffs, pp_diffs)

from pmdarima.arima import nsdiffs

ocsb_diffs = nsdiffs(y_train, 7, max_D=20, test='ocsb')
ch_diffs = nsdiffs(y_train, 7, max_D=20, test='ch')
ns_diffs = max(ocsb_diffs, ch_diffs)

print(f"Estimated differencing term: {n_diffs}")
print(f"Estimate differencing term: {ns_diffs}")
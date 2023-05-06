# Import libraries
import pandas as pd

# Load data
df = pd.read_csv('south-korea-new-cases.csv')
df.rename(columns={'Date_reported': 'ds', 'New_cases': 'y'}, inplace=True)

# Define start date and end date
start_date = '2022-01-01'
end_date = '2023-03-05'

# Filter data by date range
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

# Check for missing values
print(df.isnull().sum())
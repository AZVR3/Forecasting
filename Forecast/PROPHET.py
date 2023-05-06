# Import libraries
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Load data
df = pd.read_csv('south-korea-new-cases.csv')
df.rename(columns={'Date_reported': 'ds', 'New_cases': 'y'}, inplace=True)

# Filter data by date range
df = df[(df['ds'] >= '2022-01-01') & (df['ds'] <= '2023-03-05')]

# Create and fit model
model = Prophet()
model.fit(df)

# Make future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Define ratios
ratios = [0.8, 0.7, 0.6]

# Loop over ratios and plot results
for ratio in ratios:
    # Split data into train and test sets
    train_size = int(len(df) * ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]

    # Fit model on train set and predict on test set
    model.fit(train_df)
    pred_df = model.predict(test_df)

    # Calculate mean absolute error (MAE)
    mae = abs(pred_df['yhat'] - test_df['y']).mean()

    # Plot actual vs predicted values
    fig = px.line(test_df, x='ds', y='y', title=f'Ratio: {ratio}, MAE: {mae:.2f}')
    fig.add_scatter(x=pred_df['ds'], y=pred_df['yhat'], mode='lines', name='Predicted')
    fig.add_scatter(x=pred_df['ds'], y=pred_df['yhat_lower'], mode='lines', name='Lower Bound')
    fig.add_scatter(x=pred_df['ds'], y=pred_df['yhat_upper'], mode='lines', name='Upper Bound')
    fig.show()

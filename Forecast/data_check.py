# Import libraries
import pandas as pd
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Read the data
df = pd.read_csv("south-korea-new-cases.csv")

# Calculate the first, second and third difference
df['First_diff'] = df['New_cases'].diff()
df['Second_diff'] = df['First_diff'].diff()
df['Third_diff'] = df['Second_diff'].diff()

# Create a figure with four subplots
fig = make_subplots(rows=2, cols=2)

# Add traces for each subplot
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['New_cases'], name='New cases'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['First_diff'], name='First difference'), row=1, col=2)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Second_diff'], name='Second difference'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date_reported'], y=df['Third_diff'], name='Third difference'), row=2, col=2)

# Update the layout and title
fig.update_layout(title='Total Cases and Differences for South Korea')
fig.show()

import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title('Sales Forecasting Platform')

# Load data
df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Input for forecasting period
periods = st.slider('Forecast periods (months)', 1, 12, 3)

# Model
model = ARIMA(df['Sales'], order=(1,1,1))
model_fit = model.fit()

forecast = model_fit.forecast(steps=periods)
forecast_dates = pd.date_range(start=df['Date'].max(), periods=periods+1, freq='M')[1:]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['Sales'], label='Historical')
ax.plot(forecast_dates, forecast, label='Forecast', color='red')
ax.set_title('Sales Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
st.pyplot(fig)

# Show forecast data
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Sales': forecast})
st.write(forecast_df)
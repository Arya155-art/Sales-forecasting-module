import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Data Preprocessing
df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()  # Handle missing values

# Step 2: Visualize Trends
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'])
plt.title('Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Step 3: Feature Engineering
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Step 4: Choose Model - ARIMA
# For simplicity, use ARIMA(1,1,1)
model = ARIMA(df['Sales'], order=(1,1,1))
model_fit = model.fit()

# Forecast next 3 months
forecast = model_fit.forecast(steps=3)
forecast_dates = pd.date_range(start=df['Date'].max(), periods=4, freq='M')[1:]

# Step 5: Visualize Forecast
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Sales'], label='Historical')
plt.plot(forecast_dates, forecast, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 6: Evaluate Model
# For evaluation, use in-sample prediction
predicted = model_fit.predict(start=1, end=len(df)-1)
actual = df['Sales'][1:]
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f'MAE: {mae}, RMSE: {rmse}')
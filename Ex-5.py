17)Perform Daily website visitors Time Series Analysis and apply the various visualization techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate synthetic daily visitors data for one year
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31')
visitors = np.random.poisson(lam=200, size=len(date_range)) + \
           10 * np.sin(2 * np.pi * date_range.dayofyear / 365)  # Add seasonality

data = pd.DataFrame({'Date': date_range, 'Visitors': visitors})
data.set_index('Date', inplace=True)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(data['Visitors'], color='blue', label='Daily Visitors')
plt.title('Daily Website Visitors')
plt.xlabel('Date')
plt.ylabel('Number of Visitors')
plt.legend()
plt.show()

# Calculate and plot moving average (30-day)
data['Moving_Avg'] = data['Visitors'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(data['Visitors'], label='Daily Visitors')
plt.plot(data['Moving_Avg'], color='orange', label='30-Day Moving Average')
plt.title('Daily Visitors with Moving Average')
plt.xlabel('Date')
plt.ylabel('Number of Visitors')
plt.legend()
plt.show()

# Decomposition
decomposition = seasonal_decompose(data['Visitors'], model='additive', period=30)
decomposition.plot()
plt.show()

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(data['Visitors'], lags=30)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(data['Visitors'], lags=30)
plt.show()

# Stationarity check with Augmented Dickey-Fuller Test
adf_result = adfuller(data['Visitors'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f'Critical Value ({key}): {value}')

# Visualization Summary
summary = {
    'ADF Statistic': adf_result[0],
    'p-value': adf_result[1],
    'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
}

summary
============================================================================================================================================================================================
18) Perform daily climate Time Series Analysis and apply the various visualization techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate synthetic daily climate data (temperature) for one year
np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31')
# Generate daily temperatures with a seasonal sinusoidal component and some noise
temperature = 15 + 10 * np.sin(2 * np.pi * date_range.dayofyear / 365) + np.random.normal(0, 2, len(date_range))

climate_data = pd.DataFrame({'Date': date_range, 'Temperature': temperature})
climate_data.set_index('Date', inplace=True)

# Line Plot for Temperature Time Series
plt.figure(figsize=(12, 6))
plt.plot(climate_data['Temperature'], color='blue', label='Daily Temperature')
plt.title('Daily Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Calculate and Plot Moving Average (30-day)
climate_data['Moving_Avg'] = climate_data['Temperature'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(climate_data['Temperature'], label='Daily Temperature')
plt.plot(climate_data['Moving_Avg'], color='orange', label='30-Day Moving Average')
plt.title('Daily Temperature with Moving Average')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Decomposition of Temperature Data (Trend, Seasonality, Residuals)
decomposition = seasonal_decompose(climate_data['Temperature'], model='additive', period=30)
decomposition.plot()
plt.show()

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(climate_data['Temperature'], lags=30)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(climate_data['Temperature'], lags=30)
plt.show()

# Stationarity check with Augmented Dickey-Fuller Test
adf_result = adfuller(climate_data['Temperature'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f'Critical Value ({key}): {value}')

# Summary
summary = {
    'ADF Statistic': adf_result[0],
    'p-value': adf_result[1],
    'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
}

summary
============================================================================================================================================================================================
19)Perform House property sales Time Series Analysis and apply the various visualization techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate synthetic monthly house sales data for three years
np.random.seed(10)
date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='M')
# Generate synthetic monthly sales data with seasonality and trend
sales = 50 + 2 * (np.arange(len(date_range))) + 10 * np.sin(2 * np.pi * date_range.month / 12) + np.random.normal(0, 3, len(date_range))

sales_data = pd.DataFrame({'Date': date_range, 'Sales': sales})
sales_data.set_index('Date', inplace=True)

# Line Plot for Sales Time Series
plt.figure(figsize=(12, 6))
plt.plot(sales_data['Sales'], color='blue', label='Monthly Sales')
plt.title('Monthly House Property Sales')
plt.xlabel('Date')
plt.ylabel('Sales Count')
plt.legend()
plt.show()

# Calculate and Plot Moving Average (3-month)
sales_data['Moving_Avg'] = sales_data['Sales'].rolling(window=3).mean()

plt.figure(figsize=(12, 6))
plt.plot(sales_data['Sales'], label='Monthly Sales')
plt.plot(sales_data['Moving_Avg'], color='orange', label='3-Month Moving Average')
plt.title('Monthly House Sales with Moving Average')
plt.xlabel('Date')
plt.ylabel('Sales Count')
plt.legend()
plt.show()

# Decomposition of Sales Data (Trend, Seasonality, Residuals)
decomposition = seasonal_decompose(sales_data['Sales'], model='additive', period=12)
decomposition.plot()
plt.show()

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(sales_data['Sales'], lags=24)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(sales_data['Sales'], lags=24)
plt.show()

# Stationarity check with Augmented Dickey-Fuller Test
adf_result = adfuller(sales_data['Sales'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f'Critical Value ({key}): {value}')

# Summary
summary = {
    'ADF Statistic': adf_result[0],
    'p-value': adf_result[1],
    'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
}

summary
===========================================================================================================================================================================================
20)Perform Stock exchange Time Series Analysis and apply the various visualization techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Generate synthetic daily stock prices data for one year
np.random.seed(42)
date_range = pd.date_range(start='2023-01-01', end='2023-12-31')
# Generate synthetic stock prices with a random walk, trend, and seasonal effect
stock_price = 100 + np.cumsum(np.random.normal(0, 1, len(date_range))) + \
              2 * np.sin(2 * np.pi * date_range.dayofyear / 365)  # Add some seasonality

stock_data = pd.DataFrame({'Date': date_range, 'Close': stock_price})
stock_data.set_index('Date', inplace=True)

# Line Plot for Stock Prices Time Series
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], color='blue', label='Daily Close Price')
plt.title('Daily Stock Close Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate and Plot Moving Average (30-day)
stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=30).mean()

plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'], label='Daily Close Price')
plt.plot(stock_data['Moving_Avg'], color='orange', label='30-Day Moving Average')
plt.title('Stock Prices with Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Decomposition of Stock Prices (Trend, Seasonality, Residuals)
decomposition = seasonal_decompose(stock_data['Close'], model='additive', period=30)
decomposition.plot()
plt.show()

# Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(12, 6))
plot_acf(stock_data['Close'], lags=30)
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(stock_data['Close'], lags=30)
plt.show()

# Stationarity check with Augmented Dickey-Fuller Test
adf_result = adfuller(stock_data['Close'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f'Critical Value ({key}): {value}')

# Summary
summary = {
    'ADF Statistic': adf_result[0],
    'p-value': adf_result[1],
    'Stationary': 'Yes' if adf_result[1] < 0.05 else 'No'
}

print(summary)
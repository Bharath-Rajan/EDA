### 4. Model a Daily website visitors Time Series Analysis and apply the various visualization techniques.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic daily visitors data
dates = pd.date_range(start='2024-10-01', periods=30)
visitors = np.random.randint(100, 500, size=len(dates))

visitors_df = pd.DataFrame({'Date': dates, 'Visitors': visitors})

# Plotting the time series
plt.plot(visitors_df['Date'], visitors_df['Visitors'], marker='o')
plt.title('Daily Website Visitors')
plt.xlabel('Date')
plt.ylabel('Number of Visitors')
plt.xticks(rotation=45)
plt.grid()
plt.show()



### 20. Design House property sales Time Series Analysis and apply the various visualization techniques.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic house sales data
dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
sales = np.random.randint(200000, 500000, size=len(dates))

sales_df = pd.DataFrame({'Date': dates, 'Sales': sales})

# Plotting the time series
plt.figure(figsize=(12, 6))
plt.plot(sales_df['Date'], sales_df['Sales'], marker='o')
plt.title('Monthly House Sales')
plt.xlabel('Date')
plt.ylabel('Sales Amount ($)')
plt.xticks(rotation=45)
plt.grid()
plt.show()



### 25. Develop a daily climate Time Series Analysis and apply the various visualization techniques.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic daily climate data
dates = pd.date_range(start='2024-10-01', periods=30)
temperature = np.random.randint(15, 30, size=len(dates))

climate_df = pd.DataFrame({'Date': dates, 'Temperature': temperature})

# Plotting the time series
plt.plot(climate_df['Date'], climate_df['Temperature'], marker='o', color='green')
plt.title('Daily Climate Analysis')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.grid()
plt.show()



### 27. Build a Stock exchange Time Series Analysis and apply the various visualization techniques.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic stock price data
dates = pd.date_range(start='2024-01-01', periods=60, freq='B')  # Business days
prices = np.random.randint(100, 200, size=len(dates))

stock_df = pd.DataFrame({'Date': dates, 'Price': prices})

# Plotting the time series
plt.plot(stock_df['Date'], stock_df['Price'], marker='o')
plt.title('Stock Price Analysis')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.xticks(rotation=45)
plt.grid()
plt.show()


### 32. Build a comprehensive time series analysis, utilizing appropriate visualization methods to explore trends, patterns, and anomalies within the data.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic time series data
dates = pd.date_range(start='2024-01-01', periods=100)
data = np.random.randn(len(dates)).cumsum()

ts_df = pd.DataFrame({'Date': dates, 'Value': data})

# Plotting the time series
plt.figure(figsize=(12, 6))
plt.plot(ts_df['Date'], ts_df['Value'], marker='o')
plt.title('Time Series Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid()
plt.show()

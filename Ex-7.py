25)Build a comprehensive time series analysis, utilizing appropriate visualization methods to explore trends, patterns, and anomalies within the dataimport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import IsolationForest

# Set the style for plots
sns.set(style='whitegrid')

# Step 1: Generate synthetic monthly time series data
# In practice, load your data using pd.read_csv() or pd.read_excel()
date_rng = pd.date_range(start='1/1/2010', end='12/31/2020', freq='M')
np.random.seed(42)
data = pd.DataFrame(date_rng, columns=['date'])
data['sales'] = np.random.poisson(lam=200, size=len(date_rng)) + \
                 np.linspace(0, 50, len(date_rng)) + \
                 np.sin(np.linspace(0, 10 * np.pi, len(date_rng))) * 10
data.set_index('date', inplace=True)

# Step 2: Visualize the Time Series Data
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['sales'], marker='o', linestyle='-', color='b')
plt.title('Monthly Sales Data (Synthetic)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Step 3: Decompose the Time Series
decomposition = seasonal_decompose(data['sales'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposition
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(data['sales'], label='Original', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='orange')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='red')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Step 4: Detect Anomalies using Isolation Forest
model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
data['anomaly'] = model.fit_predict(data[['sales']])

# Plot anomalies
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['sales'], marker='o', linestyle='-', color='b', label='Sales')
plt.scatter(data.index[data['anomaly'] == -1], data['sales'][data['anomaly'] == -1], color='red', label='Anomalies')
plt.title('Sales Data with Anomalies')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Step 5: Augmented Dickey-Fuller Test for stationarity
adf_result = adfuller(data['sales'])
print('ADF Statistic:', adf_result[0])
print('p-value:', adf_result[1])
print('Critical Values:', adf_result[4])

# Interpretation of ADF test
if adf_result[1] < 0.05:
    print("The time series is stationary (reject null hypothesis).")
else:
    print("The time series is non-stationary (fail to reject null hypothesis).")
===========================================================================================================================================================================================
26)Use a Telecommunication and Network performance data set and apply the various EDA and visualization techniques and present an analysis report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic telecommunication data
np.random.seed(42)
date_rng = pd.date_range(start='2020-01-01', end='2023-01-01', freq='H')
data = pd.DataFrame(date_rng, columns=['timestamp'])
data['network_latency'] = np.random.normal(loc=50, scale=10, size=len(date_rng))  # ms
data['packet_loss'] = np.random.uniform(0, 0.1, size=len(date_rng))  # 0% to 10%
data['throughput'] = np.random.normal(loc=100, scale=20, size=len(date_rng))  # Mbps
data.set_index('timestamp', inplace=True)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nSummary Statistics:")
print(data.describe())

# Step 3: Visualize Network Latency
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['network_latency'], label='Network Latency', color='blue', alpha=0.7)
plt.title('Network Latency Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Latency (ms)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Visualize Packet Loss
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['packet_loss'], label='Packet Loss', color='red', alpha=0.7)
plt.title('Packet Loss Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Packet Loss (%)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 5: Visualize Throughput
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['throughput'], label='Throughput', color='green', alpha=0.7)
plt.title('Throughput Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Throughput (Mbps)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 7: Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data['network_latency'], model='additive')
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()

# Step 8: Anomaly Detection using Z-score
data['z_score'] = (data['network_latency'] - data['network_latency'].mean()) / data['network_latency'].std()
data['anomaly'] = data['z_score'].apply(lambda x: x > 3 or x < -3)

# Step 9: Visualizing Anomalies
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['network_latency'], label='Network Latency', color='blue', alpha=0.5)
plt.scatter(data.index[data['anomaly']], data['network_latency'][data['anomaly']], color='red', label='Anomalies', zorder=5)
plt.title('Network Latency with Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Latency (ms)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

============================================================================================================================================================================================27)use a Agriculture and Crop yield data set and apply a various EDA and visualization techniques and present an analysis report.
27)use a Agriculture and Crop yield data set and apply a various EDA and visualization techniques and present an analysis report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic agriculture data
np.random.seed(42)
years = np.arange(2000, 2021)
crops = ['Wheat', 'Rice', 'Maize', 'Soybean', 'Barley']
data = {
    'Year': np.tile(years, len(crops)),
    'Crop': np.repeat(crops, len(years)),
    'Yield': np.random.normal(loc=200, scale=50, size=len(years) * len(crops)),  # Yield in kg/ha
    'Area': np.random.uniform(50, 150, size=len(years) * len(crops)),  # Area in ha
}
df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Total yield per crop
total_yield = df.groupby('Crop')['Yield'].sum().reset_index()

# Step 4: Visualize Total Yield per Crop
plt.figure(figsize=(10, 6))
sns.barplot(x='Crop', y='Yield', data=total_yield, palette='viridis')
plt.title('Total Crop Yield (2000-2020)')
plt.xlabel('Crop')
plt.ylabel('Total Yield (kg/ha)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 5: Yield Trends Over Years for Each Crop
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Year', y='Yield', hue='Crop', marker='o')
plt.title('Crop Yield Trends (2000-2020)')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.legend(title='Crop')
plt.grid()
plt.tight_layout()
plt.show()

# Step 6: Heatmap of Average Yield per Crop per Year
pivot_table = df.pivot("Year", "Crop", "Yield")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt=".1f", linewidths=.5)
plt.title('Average Yield per Crop per Year')
plt.xlabel('Crop')
plt.ylabel('Year')
plt.tight_layout()
plt.show()

# Step 7: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 8: Box Plot of Crop Yield by Crop Type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Crop', y='Yield', data=df, palette='Set2')
plt.title('Crop Yield Distribution by Crop Type')
plt.xlabel('Crop')
plt.ylabel('Yield (kg/ha)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 9: Analyzing Yield Outliers
outliers = df[(df['Yield'] > (df['Yield'].mean() + 3 * df['Yield'].std())) |
               (df['Yield'] < (df['Yield'].mean() - 3 * df['Yield'].std()))]

# Step 10: Visualize Outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Yield', hue='Crop', data=outliers, s=100, palette='deep', marker='D')
plt.title('Outliers in Crop Yield')
plt.xlabel('Year')
plt.ylabel('Yield (kg/ha)')
plt.grid()
plt.tight_layout()
plt.show()
============================================================================================================================================================================================
28)Use a Water resource management data set and apply a various EDA and visualization techniques and present an analysis report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic water resource management data
np.random.seed(42)
years = np.arange(2000, 2021)
regions = ['North', 'South', 'East', 'West']
data = {
    'Year': np.tile(years, len(regions)),
    'Region': np.repeat(regions, len(years)),
    'Water_Usage': np.random.normal(loc=300, scale=50, size=len(years) * len(regions)),  # Water usage in million m³
    'Water_Quality': np.random.normal(loc=80, scale=10, size=len(years) * len(regions)),  # Quality score from 0 to 100
    'Rainfall': np.random.normal(loc=600, scale=150, size=len(years) * len(regions)),  # Rainfall in mm
}
df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Water Usage Trends Over Years for Each Region
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Year', y='Water_Usage', hue='Region', marker='o')
plt.title('Water Usage Trends (2000-2020)')
plt.xlabel('Year')
plt.ylabel('Water Usage (million m³)')
plt.legend(title='Region')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Water Quality Trends Over Years for Each Region
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Year', y='Water_Quality', hue='Region', marker='o')
plt.title('Water Quality Trends (2000-2020)')
plt.xlabel('Year')
plt.ylabel('Water Quality Score')
plt.legend(title='Region')
plt.grid()
plt.tight_layout()
plt.show()

# Step 5: Rainfall Trends Over Years for Each Region
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Year', y='Rainfall', hue='Region', marker='o')
plt.title('Rainfall Trends (2000-2020)')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend(title='Region')
plt.grid()
plt.tight_layout()
plt.show()

# Step 6: Heatmap of Average Water Quality by Region and Year
pivot_quality = df.pivot("Year", "Region", "Water_Quality")
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_quality, annot=True, cmap='YlGnBu', fmt=".1f", linewidths=.5)
plt.title('Average Water Quality by Region and Year')
plt.xlabel('Region')
plt.ylabel('Year')
plt.tight_layout()
plt.show()

# Step 7: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 8: Box Plot of Water Usage by Region
plt.figure(figsize=(10, 6))
sns.boxplot(x='Region', y='Water_Usage', data=df, palette='Set2')
plt.title('Water Usage Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Water Usage (million m³)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 9: Analyzing Outliers in Water Usage
outliers = df[(df['Water_Usage'] > (df['Water_Usage'].mean() + 3 * df['Water_Usage'].std())) |
               (df['Water_Usage'] < (df['Water_Usage'].mean() - 3 * df['Water_Usage'].std()))]

# Step 10: Visualize Outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Water_Usage', hue='Region', data=outliers, s=100, palette='deep', marker='D')
plt.title('Outliers in Water Usage')
plt.xlabel('Year')
plt.ylabel('Water Usage (million m³)')
plt.grid()
plt.tight_layout()
plt.show()
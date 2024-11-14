### 30. Choose a Water resource management dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic water resource management dataset
data = {
    'Year': [2018, 2019, 2020, 2021],
    'WaterUsage': [300, 320, 350, 370],  # in million liters
}

water_df = pd.DataFrame(data)

# EDA
print(water_df.describe())

# Visualization
plt.plot(water_df['Year'], water_df['WaterUsage'], marker='o')
plt.title('Water Resource Management')
plt.xlabel('Year')
plt.ylabel('Water Usage (million liters)')
plt.grid()
plt.show()


### 18. Plan a Waste management and Recycling dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic waste management dataset
data = {
    'Month': ['January', 'February', 'March', 'April'],
    'RecycledWaste': [1500, 2000, 1800, 2200]
}

waste_df = pd.DataFrame(data)

# EDA
print(waste_df.describe())

# Visualization
plt.bar(waste_df['Month'], waste_df['RecycledWaste'], color='green')
plt.title('Monthly Recycled Waste')
plt.xlabel('Month')
plt.ylabel('Recycled Waste (kg)')
plt.show()

### 24. Plan an Agriculture and Crop yield dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic agriculture dataset
data = {
    'Crop': ['Wheat', 'Rice', 'Corn', 'Barley'],
    'Yield': [3.5, 4.0, 5.0, 2.5],  # in tons per hectare
    'Area': [100, 150, 200, 80]  # in hectares
}

agriculture_df = pd.DataFrame(data)

# EDA
print(agriculture_df.describe())

# Visualization
plt.bar(agriculture_df['Crop'], agriculture_df['Yield'], color='orange')
plt.title('Crop Yield')
plt.xlabel('Crop')
plt.ylabel('Yield (tons/ha)')
plt.show()
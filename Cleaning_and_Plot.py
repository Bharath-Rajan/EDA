### 15. Implement the data cleaning procedures and analyze the dataset using a scatter plot.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic dataset
data = {
    'Height': [150, 160, 170, None, 180],
    'Weight': [50, 60, 70, 80, None]
}

df = pd.DataFrame(data)

# Data cleaning
df.dropna(inplace=True)

# Scatter plot
plt.scatter(df['Height'], df['Weight'])
plt.title('Height vs Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.grid()
plt.show()

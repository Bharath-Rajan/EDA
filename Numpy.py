### 3. Utilize NumPy arrays and Pandas DataFrames for efficient data manipulation and analysis, coupled with Matplotlib for creating informative visualizations.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Creating a NumPy array and converting it to a DataFrame
data = np.array([[1, 100], [2, 200], [3, 300]])
df = pd.DataFrame(data, columns=['ID', 'Value'])

# Visualization
plt.plot(df['ID'], df['Value'], marker='o')
plt.title('ID vs Value')
plt.xlabel('ID')
plt.ylabel('Value')
plt.grid()
plt.show()


### 5. Construct a Plot line graph from a Numpy array for your own data.


import numpy as np
import matplotlib.pyplot as plt

# Creating a NumPy array
x = np.arange(1, 11)
y = np.random.randint(1, 100, size=10)

# Line graph
plt.plot(x, y, marker='o')
plt.title('Line Graph from NumPy Array')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.show()


### 7. Construct a Plot bar plot and box plot from a Numpy array for your own data.


import numpy as np
import matplotlib.pyplot as plt

# Creating a NumPy array
data = np.random.randint(1, 100, size=50)

# Bar plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(np.arange(1, 51), data)
plt.title('Bar Plot')
plt.xlabel('Index')
plt.ylabel('Values')

# Box plot
plt.subplot(1, 2, 2)
plt.boxplot(data)
plt.title('Box Plot')
plt.ylabel('Values')

plt.tight_layout()
plt.show()



### 9. Construct a Histogram from Numpy arrays for random numbers between 100.


import numpy as np
import matplotlib.pyplot as plt

# Generating random numbers
data = np.random.randint(1, 100, size=1000)

# Histogram
plt.hist(data, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Random Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

13)Perform EDA on various variable and row filters in R for cleaning data. Apply various plot features in R on sample data sets and visualize.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset for a class of students
data = {
    'StudentID': range(1, 16),
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah", "Ian", "Julia", "Ken", "Liam", "Mona", "Nina", "Oscar"],
    'MathScore': [88, 92, np.nan, 85, 90, 75, 68, 95, np.nan, 80, 82, 78, 87, 91, 85],
    'ScienceScore': [78, 85, 82, 90, np.nan, 70, 69, 94, 88, np.nan, 75, 80, 86, 91, 82],
    'AttendanceRate': [0.95, 0.89, 0.93, 0.87, 0.91, 0.85, 0.76, 0.94, 0.88, 0.92, 0.80, 0.83, 0.94, 0.89, 0.91]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isna().sum())

# Drop rows with missing values in MathScore or ScienceScore
df_cleaned = df.dropna(subset=['MathScore', 'ScienceScore'])

# Filter rows with AttendanceRate above 0.85
df_filtered = df_cleaned[df_cleaned['AttendanceRate'] > 0.85]

print("\nCleaned and Filtered DataFrame:")
print(df_filtered)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_filtered, x='MathScore', y='ScienceScore', color='blue', s=100)
plt.title("Scatter Plot of Math Score vs Science Score")
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df_filtered['AttendanceRate'], bins=5, color="skyblue", edgecolor="black")
plt.title("Histogram of Attendance Rates")
plt.xlabel("Attendance Rate")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered[['MathScore', 'ScienceScore']])
plt.title("Box Plot of Math and Science Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

14) Create different variable and row filters in R for data cleaning. Apply various plotting features in R to sample datasets and visualize the results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'StudentID': range(1, 16),
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah", "Ian", "Julia", "Ken", "Liam", "Mona", "Nina", "Oscar"],
    'MathScore': [88, 92, np.nan, 85, 90, 75, 68, 95, np.nan, 80, 82, 78, 87, 91, 85],
    'ScienceScore': [78, 85, 82, 90, np.nan, 70, 69, 94, 88, np.nan, 75, 80, 86, 91, 82],
    'AttendanceRate': [0.95, 0.89, 0.93, 0.87, 0.91, 0.85, 0.76, 0.94, 0.88, 0.92, 0.80, 0.83, 0.94, 0.89, 0.91]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
# Drop rows where MathScore or ScienceScore is NaN
df_cleaned = df.dropna(subset=['MathScore', 'ScienceScore'])
print("\nDataFrame after dropping rows with NaN values in MathScore or ScienceScore:")
print(df_cleaned)
# Keep only rows where AttendanceRate is above 0.85
df_filtered = df_cleaned[df_cleaned['AttendanceRate'] > 0.85]
print("\nDataFrame with AttendanceRate > 0.85:")
print(df_filtered)
# Select only columns of interest
df_selected = df_filtered[['StudentID', 'MathScore', 'ScienceScore', 'AttendanceRate']]
print("\nDataFrame with Selected Columns:")
print(df_selected)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_selected, x='MathScore', y='ScienceScore', color='blue', s=100)
plt.title("Scatter Plot of Math Score vs Science Score")
plt.xlabel("Math Score")
plt.ylabel("Science Score")
plt.grid(True)
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df_selected['MathScore'], bins=5, color="skyblue", edgecolor="black")
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


----------------------------------------------------------------------------------------------------------------------------------------------------------
15)Carry out data preprocessing by renaming columns and omitting NaN values in a user-defined dataset.

import pandas as pd
import numpy as np

# Sample user-defined dataset
data = {
    'Stu_ID': [1, 2, 3, 4, 5],
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva"],
    'Math_Score': [88, 92, np.nan, 85, 90],
    'Science_Score': [78, np.nan, 82, 85, 91],
    'Attendance_Rate': [0.95, 0.89, 0.93, np.nan, 0.91]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Step 1: Rename columns for consistency and readability
df = df.rename(columns={
    'Stu_ID': 'StudentID',
    'Math_Score': 'MathScore',
    'Science_Score': 'ScienceScore',
    'Attendance_Rate': 'AttendanceRate'
})
print("\nDataFrame with Renamed Columns:")
print(df)

# Step 2: Omit rows with any NaN values
df_cleaned = df.dropna()
print("\nDataFrame after Omitting Rows with NaN values:")
print(df_cleaned)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
16)Perform data preprocessing by eliminating NULL values in a dataset

import pandas as pd
import numpy as np

# Sample dataset with NULL values (NaN)
data = {
    'StudentID': [1, 2, 3, 4, 5],
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva"],
    'MathScore': [88, np.nan, 78, 85, np.nan],
    'ScienceScore': [92, 81, np.nan, 89, 76],
    'AttendanceRate': [0.95, 0.89, np.nan, 0.87, 0.91]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Option 1: Drop rows with any NULL values
df_dropped_rows = df.dropna()
print("\nDataFrame after Dropping Rows with NULL values:")
print(df_dropped_rows)

# Option 2: Drop columns with any NULL values
df_dropped_columns = df.dropna(axis=1)
print("\nDataFrame after Dropping Columns with NULL values:")
print(df_dropped_columns)

# Option 3: Fill NULL values with a specific value (e.g., 0 or the mean of the column)
df_filled = df.fillna({
    'MathScore': df['MathScore'].mean(),  # Fill with mean of 'MathScore'
    'ScienceScore': df['ScienceScore'].mean(),  # Fill with mean of 'ScienceScore'
    'AttendanceRate': 0.90  # Arbitrary value for 'AttendanceRate'
})
print("\nDataFrame after Filling NULL values:")
print(df_filled)

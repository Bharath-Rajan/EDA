5)Construct Histogram from Numpy arrays for random numbers between 100
import numpy as np
import matplotlib.pyplot as plt

# Generate random data between 0 and 100
data = np.random.randint(0, 100, size=1000)  # 1000 random integers between 0 and 100

# Create the histogram
plt.hist(data, bins=10, color='purple', edgecolor='black')
plt.title("Histogram of Random Numbers")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
-----------------------------------------------------------------------------------------------------------------------------------------------------------

6)Plot any two plots for school management dataset.

import numpy as np
import matplotlib.pyplot as plt

# Sample data for the school management dataset
subjects = ["Math", "Science", "English", "History", "Art"]
average_grades = np.random.randint(60, 100, size=len(subjects))  # Random average grades for each subject

activities = ["Sports", "Music", "Art Club", "Drama", "Science Club"]
students_in_activities = np.random.randint(10, 50, size=len(activities))  # Random number of students per activity

# Bar Plot for Average Grades per Subject
plt.figure(figsize=(12, 5))

# Plot 1: Bar plot for average grades per subject
plt.subplot(1, 2, 1)
plt.bar(subjects, average_grades, color='skyblue')
plt.title("Average Grades per Subject")
plt.xlabel("Subjects")
plt.ylabel("Average Grade")
plt.ylim(50, 100)  # Grade range

# Pie Chart for Distribution of Students in Activities
plt.subplot(1, 2, 2)
plt.pie(students_in_activities, labels=activities, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title("Distribution of Students in Extracurricular Activities")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

----------------------------------------------------------------------------------------------------------------------------------------------------------

7) Implement data preprocessing by performing dropping a columns in a dataset.
import pandas as pd

# Sample data for a school management dataset
data = {
    'StudentID': [1, 2, 3, 4, 5],
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva"],
    'Grade': [88, 92, 79, 85, 90],
    'Age': [15, 16, 15, 17, 16],
    'AttendanceRate': [0.95, 0.89, 0.93, 0.87, 0.91]
}

# Creating a DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Dropping the 'Age' column
df = df.drop(columns=['Age'])
print("\nDataFrame after dropping 'Age' column:")
print(df)
--------------------------------------------------------------------------------------------------------------------------------------------------------------

8)Implement the data cleaning procedures and analysis the dataset using scatter plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data for a school management dataset with some missing values and duplicates
data = {
    'StudentID': [1, 2, 3, 4, 5, 6, 6],
    'Name': ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Frank"],
    'Grade': [88, np.nan, 79, 85, 90, 91, 91],
    'Age': [15, 16, np.nan, 17, 16, 15, 15],
    'AttendanceRate': [0.95, 0.89, 0.93, np.nan, 0.91, 0.92, 0.92]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Data Cleaning Procedures

# 1. Drop duplicate rows
df = df.drop_duplicates()

# 2. Handle missing values by filling with the mean for numerical columns
df['Grade'].fillna(df['Grade'].mean(), inplace=True)
df['AttendanceRate'].fillna(df['AttendanceRate'].mean(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)

# 3. Convert data types if necessary (e.g., AttendanceRate to percentage if it makes sense)
# No type changes needed in this case.

print("\nCleaned DataFrame:")
print(df)

# Analysis: Scatter Plot of Attendance Rate vs. Grade
plt.figure(figsize=(8, 5))
plt.scatter(df['AttendanceRate'], df['Grade'], color='b', alpha=0.6)
plt.title("Scatter Plot of Attendance Rate vs. Grade")
plt.xlabel("Attendance Rate")
plt.ylabel("Grade")
plt.grid(True)
plt.show()

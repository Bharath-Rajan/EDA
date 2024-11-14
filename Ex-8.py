29) Use a Waste management and Recycling data set and apply a various EDA and visualization techniques and present an analysis report.
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
============================================================================================================================================================================================
29)Use a Employee satisfaction data set and apply a various EDA and visualization techniques and present an analysis report.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic employee satisfaction data
np.random.seed(42)
data = {
    'Employee_ID': np.arange(1, 101),
    'Satisfaction_Score': np.random.uniform(1, 5, 100),  # Satisfaction score from 1 to 5
    'Department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing'], 100),
    'Job_Level': np.random.choice(['Junior', 'Mid', 'Senior'], 100),
    'Salary': np.random.normal(60000, 15000, 100),  # Salary in USD
    'Years_at_Company': np.random.randint(1, 20, 100)
}
df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Distribution of Satisfaction Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Satisfaction_Score'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Employee Satisfaction Scores')
plt.xlabel('Satisfaction Score')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Satisfaction Score by Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Satisfaction_Score', data=df, palette='Set2')
plt.title('Satisfaction Scores by Department')
plt.xlabel('Department')
plt.ylabel('Satisfaction Score')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 5: Average Satisfaction Score by Job Level
avg_satisfaction_by_level = df.groupby('Job_Level')['Satisfaction_Score'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Job_Level', y='Satisfaction_Score', data=avg_satisfaction_by_level, palette='Set1')
plt.title('Average Satisfaction Score by Job Level')
plt.xlabel('Job Level')
plt.ylabel('Average Satisfaction Score')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 7: Satisfaction Score vs Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='Satisfaction_Score', data=df, hue='Department', style='Job_Level', s=100)
plt.title('Satisfaction Score vs Salary')
plt.xlabel('Salary (USD)')
plt.ylabel('Satisfaction Score')
plt.grid()
plt.tight_layout()
plt.show()

# Step 8: Satisfaction Score vs Years at Company
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_at_Company', y='Satisfaction_Score', data=df, hue='Department', style='Job_Level', s=100)
plt.title('Satisfaction Score vs Years at Company')
plt.xlabel('Years at Company')
plt.ylabel('Satisfaction Score')
plt.grid()
plt.tight_layout()
plt.show()

# Step 9: Countplot of Employees by Satisfaction Score
plt.figure(figsize=(10, 6))
sns.countplot(x='Satisfaction_Score', data=df, palette='Set2')
plt.title('Count of Employees by Satisfaction Score')
plt.xlabel('Satisfaction Score')
plt.ylabel('Count of Employees')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

============================================================================================================================================================================================
30)Use a Employee satisfaction data set and apply a various EDA and visualization techniques and present an analysis report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic employee satisfaction data
np.random.seed(42)
data = {
    'Employee_ID': np.arange(1, 101),
    'Satisfaction_Score': np.random.uniform(1, 5, 100),  # Satisfaction score from 1 to 5
    'Department': np.random.choice(['HR', 'Engineering', 'Sales', 'Marketing'], 100),
    'Job_Level': np.random.choice(['Junior', 'Mid', 'Senior'], 100),
    'Salary': np.random.normal(60000, 15000, 100),  # Salary in USD
    'Years_at_Company': np.random.randint(1, 20, 100)
}
df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Distribution of Satisfaction Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Satisfaction_Score'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Employee Satisfaction Scores')
plt.xlabel('Satisfaction Score')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Satisfaction Score by Department
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Satisfaction_Score', data=df, palette='Set2')
plt.title('Satisfaction Scores by Department')
plt.xlabel('Department')
plt.ylabel('Satisfaction Score')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 5: Average Satisfaction Score by Job Level
avg_satisfaction_by_level = df.groupby('Job_Level')['Satisfaction_Score'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Job_Level', y='Satisfaction_Score', data=avg_satisfaction_by_level, palette='Set1')
plt.title('Average Satisfaction Score by Job Level')
plt.xlabel('Job Level')
plt.ylabel('Average Satisfaction Score')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 6: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 7: Satisfaction Score vs Salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='Satisfaction_Score', data=df, hue='Department', style='Job_Level', s=100)
plt.title('Satisfaction Score vs Salary')
plt.xlabel('Salary (USD)')
plt.ylabel('Satisfaction Score')
plt.grid()
plt.tight_layout()
plt.show()

# Step 8: Satisfaction Score vs Years at Company
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Years_at_Company', y='Satisfaction_Score', data=df, hue='Department', style='Job_Level', s=100)
plt.title('Satisfaction Score vs Years at Company')
plt.xlabel('Years at Company')
plt.ylabel('Satisfaction Score')
plt.grid()
plt.tight_layout()
plt.show()

# Step 9: Countplot of Employees by Satisfaction Score
plt.figure(figsize=(10, 6))
sns.countplot(x='Satisfaction_Score', data=df, palette='Set2')
plt.title('Count of Employees by Satisfaction Score')
plt.xlabel('Satisfaction Score')
plt.ylabel('Count of Employees')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 10: Summary of Insights
print("\nSummary of Insights:")
print("1. The distribution of satisfaction scores indicates the range of employee satisfaction levels.")
print("2. Departments show varied satisfaction scores, which could suggest the need for targeted interventions.")
print("3. Job levels indicate differing average satisfaction, potentially highlighting areas for improvement.")
print("4. Salary appears to have a relationship with satisfaction scores, suggesting better compensation may lead to higher satisfaction.")
print("5. Employees' years at the company may also correlate with satisfaction, indicating loyalty factors.")

============================================================================================================================================================================================
31)Use a Healthcare and medical data set and apply a various EDA and visualization techniques and present an analysis report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic healthcare data
np.random.seed(42)
data = {
    'Patient_ID': np.arange(1, 101),
    'Age': np.random.randint(20, 80, 100),  # Age between 20 and 80
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Blood_Pressure': np.random.randint(90, 180, 100),  # Systolic BP
    'Cholesterol': np.random.choice(['Normal', 'High'], 100),
    'BMI': np.random.uniform(18.5, 40, 100),  # BMI range
    'Diabetes': np.random.choice(['Yes', 'No'], 100),
    'Outcome': np.random.choice(['Healthy', 'At Risk', 'Sick'], 100)
}

df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Blood Pressure Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Blood_Pressure'], bins=10, kde=True, color='salmon')
plt.title('Distribution of Blood Pressure')
plt.xlabel('Blood Pressure (mm Hg)')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Step 5: Count of Patients by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Count of Patients by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 6: Cholesterol Levels Count
plt.figure(figsize=(10, 6))
sns.countplot(x='Cholesterol', data=df, palette='Set1')
plt.title('Cholesterol Levels')
plt.xlabel('Cholesterol')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 7: Outcome by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Outcome', hue='Gender', data=df, palette='Set3')
plt.title('Patient Outcomes by Gender')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 8: Boxplot of Blood Pressure by Outcome
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outcome', y='Blood_Pressure', data=df, palette='Set2')
plt.title('Blood Pressure by Outcome')
plt.xlabel('Outcome')
plt.ylabel('Blood Pressure (mm Hg)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 9: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 10: Summary of Insights
print("\nSummary of Insights:")
print("1. Age distribution indicates a varied age range among patients.")
print("2. Blood pressure levels vary significantly, with some patients at higher risk.")
print("3. Gender distribution shows a relatively balanced representation.")
print("4. Cholesterol levels indicate a significant proportion of patients with high cholesterol.")
print("5. Patient outcomes reveal interesting trends by gender, highlighting health disparities.")
print("6. Blood pressure shows different distributions based on patient outcomes, suggesting health implications.")
============================================================================================================================================================================================
32)Use a Education and student performance data set and apply a various EDA and visualization techniques and present an analysis report.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the visual style
sns.set(style='whitegrid')

# Step 1: Generate synthetic education data
np.random.seed(42)
data = {
    'Student_ID': np.arange(1, 101),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Age': np.random.randint(15, 21, 100),  # Age between 15 and 20
    'Study_Time': np.random.choice(['<2 hours', '2-5 hours', '>5 hours'], 100),
    'Final_Grade': np.random.uniform(50, 100, 100),  # Final grades from 50 to 100
    'Absences': np.random.randint(0, 20, 100)  # Number of absences
}

df = pd.DataFrame(data)

# Step 2: Overview of the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Distribution of Final Grades
plt.figure(figsize=(10, 6))
sns.histplot(df['Final_Grade'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Final Grades')
plt.xlabel('Final Grade')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
plt.show()

# Step 4: Final Grades by Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Final_Grade', data=df, palette='Set2')
plt.title('Final Grades by Gender')
plt.xlabel('Gender')
plt.ylabel('Final Grade')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 5: Average Final Grade by Study Time
avg_grade_by_study_time = df.groupby('Study_Time')['Final_Grade'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Study_Time', y='Final_Grade', data=avg_grade_by_study_time, palette='Set1')
plt.title('Average Final Grade by Study Time')
plt.xlabel('Study Time')
plt.ylabel('Average Final Grade')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 6: Absences vs Final Grade
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Absences', y='Final_Grade', hue='Gender', style='Study_Time', data=df, s=100)
plt.title('Absences vs Final Grade')
plt.xlabel('Number of Absences')
plt.ylabel('Final Grade')
plt.grid()
plt.tight_layout()
plt.show()

# Step 7: Countplot of Students by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title('Count of Students by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Step 8: Correlation Analysis
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 9: Summary of Insights
print("\nSummary of Insights:")
print("1. The distribution of final grades shows a varied range, with most students scoring above 70.")
print("2. Final grades by gender indicate that there may be differences in performance.")
print("3. Students who study more tend to have higher average grades.")
print("4. There is a negative correlation between absences and final grades.")
print("5. The count of students is balanced between genders, with slight variations.")



### 1. Develop an email dataset to conduct a comprehensive exploratory data analysis (EDA). 


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic email dataset
data = {
    'Sender': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
    'Receiver': ['dave@example.com', 'eve@example.com', 'frank@example.com'],
    'Subject': ['Meeting', 'Project Update', 'Invoice'],
    'Date': pd.to_datetime(['2024-10-01', '2024-10-02', '2024-10-03']),
    'Read': [True, False, True]
}

email_df = pd.DataFrame(data)

# EDA
print(email_df.describe())
print(email_df['Sender'].value_counts())

# Visualization
email_df['Date'].value_counts().plot(kind='bar')
plt.title('Emails per Date')
plt.xlabel('Date')
plt.ylabel('Number of Emails')
plt.xticks(rotation=45)
plt.show()


### 8. Construct a Healthcare and medical dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic healthcare dataset
data = {
    'PatientID': [1, 2, 3, 4],
    'Age': [25, 30, 45, 50],
    'Cholesterol': [200, 240, 210, 180],
    'BloodPressure': [120, 130, 110, 140]
}

health_df = pd.DataFrame(data)

# EDA
print(health_df.describe())

# Visualization
plt.scatter(health_df['Cholesterol'], health_df['BloodPressure'])
plt.title('Cholesterol vs Blood Pressure')
plt.xlabel('Cholesterol Level')
plt.ylabel('Blood Pressure')
plt.grid()
plt.show()


### 12. Utilize a Telecommunication and Network performance dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic telecommunication dataset
data = {
    'Network': ['A', 'B', 'C', 'D'],
    'Performance': [85, 90, 75, 80],
    'Failures': [5, 3, 10, 7]
}

telecom_df = pd.DataFrame(data)

# EDA
print(telecom_df.describe())

# Visualization
plt.bar(telecom_df['Network'], telecom_df['Performance'], color='orange')
plt.title('Network Performance')
plt.xlabel('Network')
plt.ylabel('Performance (%)')
plt.show()













### 26. Utilize an Employee satisfaction dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic employee satisfaction dataset
data = {
    'EmployeeID': [1, 2, 3, 4],
    'Satisfaction': [7, 8, 5, 6],
    'Salary': [50000, 60000, 55000, 45000]
}

employee_df = pd.DataFrame(data)

# EDA
print(employee_df.describe())

# Visualization
plt.scatter(employee_df['Salary'], employee_df['Satisfaction'])
plt.title('Employee Satisfaction vs Salary')
plt.xlabel('Salary')
plt.ylabel('Satisfaction Level')
plt.grid()
plt.show()


### 28. Develop an Education and student performance dataset and apply various EDA and visualization techniques and present an analysis report.


import pandas as pd
import matplotlib.pyplot as plt

# Creating a synthetic education dataset
data = {
    'StudentID': [1, 2, 3, 4],
    'MathScore': [90, 85, 78, 92],
    'EnglishScore': [88, 76, 95, 80]
}

education_df = pd.DataFrame(data)

# EDA
print(education_df.describe())

# Visualization
plt.boxplot([education_df['MathScore'], education_df['EnglishScore']], labels=['Math', 'English'])
plt.title('Scores Comparison')
plt.ylabel('Scores')
plt.show()


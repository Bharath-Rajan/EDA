### 11. Construct a Plot of any two plots for a school management dataset.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a synthetic school management dataset
data = {
    'StudentID': [1, 2, 3, 4, 5],
    'Scores': np.random.randint(50, 100, size=5),
    'Attendance': np.random.randint(70, 100, size=5)
}

school_df = pd.DataFrame(data)

# Bar plot
plt.subplot(1, 2, 1)
plt.bar(school_df['StudentID'], school_df['Scores'], color='green')
plt.title('Student Scores')
plt.xlabel('Student ID')
plt.ylabel('Scores')

# Line plot
plt.subplot(1, 2, 2)
plt.plot(school_df['StudentID'], school_df['Attendance'], marker='o', color='blue')
plt.title('Student Attendance')
plt.xlabel('Student ID')
plt.ylabel('Attendance (%)')

plt.tight_layout()
plt.show()


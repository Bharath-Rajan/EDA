1)Develop an email dataset to conduct a comprehensive exploratory data analysis (EDA). Import the email data into a Pandas DataFrame, visualize key features using appropriate techniques, and extract valuable insights about the data



import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate sample data
n_samples = 1000
np.random.seed(0)

# Create a list of fake email addresses
senders = [f"sender{i}@example.com" for i in range(1, 51)]
receivers = [f"receiver{i}@example.com" for i in range(1, 51)]

# Create random dates within the last year
start_date = datetime.now() - timedelta(days=365)
timestamps = [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23), minutes=random.randint(0, 59)) for _ in range(n_samples)]

# Generate email data
data = {
    "sender": [random.choice(senders) for _ in range(n_samples)],
    "receiver": [random.choice(receivers) for _ in range(n_samples)],
    "timestamp": timestamps,
    "subject": ["Sample Subject " + str(i) for i in range(n_samples)],
    "body": ["This is a sample email body with some text." for _ in range(n_samples)],
    "email_length": [random.randint(50, 500) for _ in range(n_samples)],
    "num_attachments": [random.randint(0, 5) for _ in range(n_samples)],
    "is_spam": [random.choice([0, 1]) for _ in range(n_samples)],
    "keywords": [["keyword1", "keyword2", "keyword3"][random.randint(0, 2)] for _ in range(n_samples)]
}

# Create the DataFrame
email_df = pd.DataFrame(data)
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Generate sample data
n_samples = 1000
np.random.seed(0)

# Create a list of fake email addresses
senders = [f"sender{i}@example.com" for i in range(1, 51)]
receivers = [f"receiver{i}@example.com" for i in range(1, 51)]

# Create random dates within the last year
start_date = datetime.now() - timedelta(days=365)
timestamps = [start_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23), minutes=random.randint(0, 59)) for _ in range(n_samples)]

# Generate email data
data = {
    "sender": [random.choice(senders) for _ in range(n_samples)],
    "receiver": [random.choice(receivers) for _ in range(n_samples)],
    "timestamp": timestamps,
    "subject": ["Sample Subject " + str(i) for i in range(n_samples)],
    "body": ["This is a sample email body with some text." for _ in range(n_samples)],
    "email_length": [random.randint(50, 500) for _ in range(n_samples)],
    "num_attachments": [random.randint(0, 5) for _ in range(n_samples)],
    "is_spam": [random.choice([0, 1]) for _ in range(n_samples)],
    "keywords": [["keyword1", "keyword2", "keyword3"][random.randint(0, 2)] for _ in range(n_samples)]
}

# Create the DataFrame
email_df = pd.DataFrame(data)
# Display basic info
print(email_df.info())

# Display first few rows
print(email_df.head())

# Check for missing values
print(email_df.isnull().sum())
# Basic statistics
print(email_df[['email_length', 'num_attachments']].describe())

# Count of spam vs. non-spam emails
print(email_df['is_spam'].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns

# Plot email length distribution
plt.figure(figsize=(10, 5))
sns.histplot(email_df['email_length'], bins=30, kde=True)
plt.title('Distribution of Email Lengths')
plt.xlabel('Email Length')
plt.ylabel('Frequency')
plt.show()

# Number of attachments distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='num_attachments', data=email_df)
plt.title('Distribution of Number of Attachments')
plt.xlabel('Number of Attachments')
plt.ylabel('Count')
plt.show()

# Spam vs. non-spam email count
plt.figure(figsize=(10, 5))
sns.countplot(x='is_spam', data=email_df)
plt.title('Spam vs. Non-Spam Emails')
plt.xlabel('Is Spam')
plt.ylabel('Count')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------
2)Utilize NumPy arrays and Pandas DataFrames for efficient data manipulation and analysis, coupled with Matplotlib for creating informative visualizations.

import numpy as np

# Create a random dataset with 100 entries and 3 columns
data = np.random.rand(100, 3)
import pandas as pd

# Convert the NumPy array to a DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3'])
# Summary statistics
summary_stats = df.describe()
print(summary_stats)

# Filter entries where Feature1 is greater than 0.5
filtered_df = df[df['Feature1'] > 0.5]
import matplotlib.pyplot as plt

# Histogram for Feature1
plt.hist(df['Feature1'], bins=10, color='blue', alpha=0.7)
plt.title('Distribution of Feature1')
plt.xlabel('Feature1')
plt.ylabel('Frequency')
plt.show()

# Scatter plot to see correlation between Feature1 and Feature2
plt.scatter(df['Feature1'], df['Feature2'], color='green', alpha=0.5)
plt.title('Feature1 vs Feature2')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------

3)plot a line graph of your own data


import numpy as np
import matplotlib.pyplot as plt

# Generate example data
x = np.arange(0, 10, 0.5)  # x-axis values from 0 to 10 with a step of 0.5
y = np.sin(x)  # y-axis values using a sine function for some variation

# Create the line plot
plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2, markersize=5)
plt.title("Line Graph from Numpy Array")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
plt.show()

-----------------------------------------------------------------------------------------------------------------------------------------
4)Plot bar plot and box plot from Numpy array for your own data.

import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = np.random.randint(1, 20, size=10)  # 10 random integer values between 1 and 20

# Bar Plot
plt.figure(figsize=(12, 5))  # Set figure size for better layout

# Bar Plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.bar(np.arange(len(data)), data, color='skyblue')
plt.title("Bar Plot from Numpy Array")
plt.xlabel("Category")
plt.ylabel("Values")

# Box Plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.boxplot(data, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
plt.title("Box Plot from Numpy Array")
plt.ylabel("Values")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

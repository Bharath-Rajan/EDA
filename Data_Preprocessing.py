### 2. Implement data preprocessing by renaming columns and skipping NaN in a user-defined dataset.


import pandas as pd

# Creating a synthetic dataset
data = {
    'name': ['Alice', 'Bob', None],
    'age': [25, 30, None],
    'city': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

# Renaming columns
df.rename(columns={'name': 'Name', 'age': 'Age', 'city': 'City'}, inplace=True)

# Dropping rows with NaN values
df.dropna(inplace=True)
print(df)


### 13. Implement data preprocessing by performing dropping columns in a dataset.


import pandas as pd

# Creating a synthetic dataset
data = {
    'ID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com']
}

df = pd.DataFrame(data)

# Dropping the 'Email' column
df.drop(columns=['Email'], inplace=True)
print(df)

### 14. Implement data preprocessing by removing NULL values in a dataset.


import pandas as pd

# Creating a synthetic dataset with NaN values
data = {
    'ID': [1, 2, 3

, 4],
    'Name': ['Alice', None, 'Charlie', 'David'],
    'Score': [90, 85, None, 80]
}

df = pd.DataFrame(data)

# Removing NULL values
clean_df = df.dropna()
print(clean_df)


### 22. Develop data preprocessing by renaming columns and omitting NaN values in a user-defined dataset.


import pandas as pd

# Creating a synthetic dataset
data = {
    'sales': [100, 200, None],
    'date': ['2024-01-01', '2024-01-02', None]
}

sales_df = pd.DataFrame(data)

# Renaming columns
sales_df.rename(columns={'sales': 'Sales_Amount', 'date': 'Sales_Date'}, inplace=True)

# Dropping rows with NaN values
sales_df.dropna(inplace=True)
print(sales_df)


### 23. Build data preprocessing by eliminating NULL values in a dataset.


import pandas as pd

# Creating a synthetic dataset with NaN values
data = {
    'ID': [1, 2, None, 4],
    'Name': ['Alice', 'Bob', 'Charlie', None],
}

df = pd.DataFrame(data)

# Eliminating NULL values
clean_df = df.dropna()
print(clean_df)

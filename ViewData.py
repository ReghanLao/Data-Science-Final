import pandas as pd

# Load dataset
data = pd.read_json("Sports_and_Outdoors_5.json", lines=True)  

# Display the first few rows 
print(data.head())

# Display basic information about the dataset
print(data.info()) 

# View a summary of statistical values
print(data.describe())  

# Check for missing values
print(data.isnull().sum())

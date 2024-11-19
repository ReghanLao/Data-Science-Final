from load_data import load_data
from clean_data import clean_data

data = load_data("Sports_and_Outdoors_5.json")

#Inspecting data & brief overview on dataset 
print("First Few Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())  

print("\nStatistical Summary:")
print(data.describe()) 

print("\nMissing Values:")
print(data.isnull().sum())  

cleaned_data = clean_data(data)

cleaned_data.to_csv("cleaned_data.csv", index=False)
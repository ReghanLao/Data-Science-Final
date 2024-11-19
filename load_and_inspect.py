import pandas as pd

def load_and_inspect_data(file_path):
    data = pd.read_json(file_path, lines=True)
    
    print("First Few Rows:")
    print(data.head())

    #Basic overview info on dataset
    print("\nDataset Info:")
    print(data.info())  

    print("\nStatistical Summary:")
    print(data.describe()) 
    
    print("\nMissing Values:")
    print(data.isnull().sum())  
    
    return data

data = load_and_inspect_data("Sports_and_Outdoors_5.json")

import pandas as pd

# Load JSON dataset
data = pd.read_json("Sports_and_Outdoors_5.json", lines=True, chunksize=10000)  

# View first few rows
print(data.head())

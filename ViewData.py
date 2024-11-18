import pandas as pd

# Load JSON dataset
small_data = pd.read_json("Sports_and_Outdoors_5.json", lines=True, chunksize=10000)  

data_sample = next(small_data)

print(data_sample.head())

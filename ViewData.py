import pandas as pd

# Load the JSON dataset
data = pd.read_json("Sports_and_Outdoors_5.json", lines=True)  # Add `lines=True` if it's line-delimited JSON

# View the first few rows
print(data.head())

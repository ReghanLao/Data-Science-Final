import pandas as pd

def load_data(file_path):
    data = pd.read_json(file_path, lines=True)  
    return data
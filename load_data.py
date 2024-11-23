import pandas as pd

def load_data(file_path, chunk_size=10000):
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
        yield chunk

import pandas as pd

def clean_data(data):
  data = data.drop(columns=['image', 'vote'], errors='ignore')

  for column in ['reviewText', 'reviewerName', 'style', 'summary']:
    data[column] = data[column].fillna(pd.NA)
  
  if 'style' in data.columns:
        data['style'] = data['style'].apply(lambda x: str(x) if isinstance(x, dict) else x)

  # Remove duplicate rows if they exist
  data = data.drop_duplicates()

  data['reviewTime'] = pd.to_datetime(data['reviewTime'], errors='coerce')

  return data

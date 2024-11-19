import pandas as pd

def split_data(data, test_size=0.2):
  train = []
  test = []

  groups = data.groupby("reviewerID")

  for user_id, group in groups:
    split_index = int(len(group) * (1-test_size))
    group = group.sample(frac=1) #randomize order of ratings
    train.append(group.iloc[:split_index])
    test.append(group.iloc[split_data:])

  train_data = pd.concat(train)
  test_data = pd.concat(test)

  return train_data, test_data
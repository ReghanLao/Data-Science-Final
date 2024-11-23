import numpy as np

def split_data(data, test_size=0.2):
    train_mask = np.random.rand(len(data)) < (1 - test_size)
    train_data = data[train_mask]
    test_data = data[~train_mask]
    return train_data, test_data
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_user_item_matrx(training_data):
  user_item_matrix = training_data.pivot_table(index='reviewerID', columns='asin', values='overall', fill_value=0)
  return user_item_matrix

def compute_similarity(user_item_matrix):
  similarity = cosine_similarity(user_item_matrix.T)
  similarity_matrix = pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
  return similarity_matrix
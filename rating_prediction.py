import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def create_user_item_matrix_sparse(training_data):
  # Map reviewerID and asin to integer indices
  user_map = {user: idx for idx, user in enumerate(training_data['reviewerID'].unique())}
  item_map = {item: idx for idx, item in enumerate(training_data['asin'].unique())}
  
  # Replace IDs with indices
  training_data['user_idx'] = training_data['reviewerID'].map(user_map)
  training_data['item_idx'] = training_data['asin'].map(item_map)
  
  # Create sparse matrix
  user_item_matrix_sparse = csr_matrix(
      (training_data['overall'], (training_data['user_idx'], training_data['item_idx']))
  )
  
  return user_item_matrix_sparse, user_map, item_map

# TO ADD: compute similarity while handle missing values properly, 
# predicting the ratings in the testing set,
# and evaluate your predictions by calculating the MAE1 and RMSE2.
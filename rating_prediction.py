import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

def create_user_item_matrix_sparse(training_data):
  # Map reviewerID and asin to integer indices
  user_map = {user: idx for idx, user in enumerate(training_data['reviewerID'].unique())}
  item_map = {item: idx for idx, item in enumerate(training_data['asin'].unique())}
  
  # Replace IDs with indices
  training_data['user_idx'] = training_data['reviewerID'].map(user_map)
  training_data['item_idx'] = training_data['asin'].map(item_map)
  
  # Create sparse matrix
  user_item_matrix_sparse = csr_matrix((training_data['overall'], (training_data['user_idx'], training_data['item_idx'])))
  
  return user_item_matrix_sparse, user_map, item_map

def compute_similarity(user_item_matrix_sparse, item_map):
  similarity = cosine_similarity(user_item_matrix_sparse.T)
  similarity_matrix = pd.DataFrame(similarity, index=item_map.keys(), columns=item_map.keys())
  return similarity_matrix

def predict_ratings(user_item_matrix_sparse, similarity_matrix, test_data, user_map, item_map):
  predictions = []
  true_ratings = []

  for _, row in test_data.iterrows():
    user_idx = user_map.get(row['reviewerID'], None)
    item_idx = item_map.get(row['asin'], None)

    if user_idx is not None and item_idx is not None:
      user_ratings = user_item_matrix_sparse[user_idx, :].toarray().flatten()
      item_similarities = similarity_matrix.iloc[item_idx, :].to_numpy()

      # weighted sum for prediction
      weighted_sum = np.dot(user_ratings, item_similarities)
      similarity_sum = np.sum(np.abs(item_similarities[user_ratings > 0]))

      predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else np.mean(user_ratings[user_ratings > 0])
      predictions.append(predicted_rating)
      true_ratings.append(row['overall'])

  return predictions, true_ratings

def evaluate_predictions(predictions, true_ratings):
  mae = mean_absolute_error(true_ratings, predictions)
  rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
  return mae, rmse

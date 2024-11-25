from load_data import load_data
from clean_data import clean_data
from rating_prediction import create_user_item_matrix_sparse, compute_similarity, predict_ratings, evaluate_predictions
from split_data import split_data
from top_N_Recs import recommend_top_n, evaluate_recommendations
import random 

# Load and clean data
data = load_data("Sports_and_Outdoors_5.json") 
cleaned_data = clean_data(data)

# Split data into training and testing 
train_data, test_data = split_data(cleaned_data)

# Create sparse user-item matrix
user_item_matrix_sparse, user_map, item_map = create_user_item_matrix_sparse(train_data)

# Create similarity matrix 
similarity_matrix = compute_similarity(user_item_matrix_sparse, item_map)

predictions, true_ratings = predict_ratings(user_item_matrix_sparse, similarity_matrix, test_data, user_map, item_map)

mae, rmse = evaluate_predictions(predictions, true_ratings)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

random_user_id = random.choice(test_data['reviewerID'].unique())
top_10_recommendations = recommend_top_n(random_user_id, similarity_matrix, user_item_matrix_sparse, user_map, item_map)

print(f"Top 10 recommendations for user {random_user_id}: {top_10_recommendations}")

metrics = evaluate_recommendations(test_data, similarity_matrix, user_item_matrix_sparse, user_map, item_map)
print(f"Top-N Recommendation Metrics: {metrics}")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def predict_rating(user_vector, item_vectors, known_ratings):
    similarities = cosine_similarity([user_vector], item_vectors)[0]
    weighted_sum = np.sum(similarities * known_ratings)
    sum_similarities = np.sum(similarities)

    if sum_similarities == 0:
        return np.mean(known_ratings)  # Use mean rating as fallback

    return weighted_sum / sum_similarities

def evaluate_predictions(true_ratings, predicted_ratings):
    # Ensure both arrays are of the same length
    true_ratings = np.array(true_ratings)
    predicted_ratings = np.array(predicted_ratings)

    # Check if lengths match
    if len(true_ratings) != len(predicted_ratings):
        raise ValueError("Length of true ratings and predicted ratings must match.")

    mae = np.mean(np.abs(true_ratings - predicted_ratings))
    rmse = np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))
    return mae, rmse
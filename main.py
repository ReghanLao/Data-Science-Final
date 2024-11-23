from load_data import load_data
from clean_data import clean_data
from embed_text import embed_text
from vector_db import VectorDB
from split_data import split_data
from rating_prediction import predict_rating, evaluate_predictions
from item_recommendation import recommend_items, evaluate_recommendations
import numpy as np

# Initialize vector database
vector_db = VectorDB(384)  # Load existing vectors if available

# Load and process data in chunks
for chunk in load_data("Sports_and_Outdoors_5.json"):
    cleaned_chunk = clean_data(chunk)

    # Embed review text with a check
    review_vectors = []
    for text in cleaned_chunk['reviewText']:
        vector = embed_text(text)
        if vector is not None:  # Ensure embedding is valid
            review_vectors.append(vector)

    # Add to vector database
    vector_db.add_vectors(review_vectors, cleaned_chunk['asin'].tolist())

    # Split data
    train_data, test_data = split_data(cleaned_chunk)

    # Rating prediction
    user_vectors = [embed_text(text) for text in train_data['reviewText']]
    item_vectors = [embed_text(text) for text in train_data['asin']]

    # Create a mapping of ASINs to indices
    asin_to_index = {asin: idx for idx, asin in enumerate(train_data['asin'])}

    predicted_ratings = []
    true_ratings = []

    # Generate predictions for each user in the test set
    for user_id in test_data['reviewerID'].unique():
        user_data = test_data[test_data['reviewerID'] == user_id]
        user_vector = embed_text(user_data['reviewText'].iloc[0])  # Get the first review text as user vector

        for asin in user_data['asin']:
            if asin in asin_to_index:  # Only predict if ASIN is in training data
                item_vector = item_vectors[asin_to_index[asin]]
                predicted_rating = predict_rating(user_vector, item_vectors, train_data['overall'])
                predicted_ratings.append(predicted_rating)
                true_ratings.append(user_data[user_data['asin'] == asin]['overall'].values[0])  # Get true rating

    # Now true_ratings and predicted_ratings should have matching lengths
    mae, rmse = evaluate_predictions(np.array(true_ratings), np.array(predicted_ratings))
    print(f"MAE: {mae}, RMSE: {rmse}")

    # Item recommendation
    for user_id in test_data['reviewerID'].unique():
        user_data = test_data[test_data['reviewerID'] == user_id]
        user_vector = embed_text(user_data['reviewText'].iloc[0])

        recommended_items = recommend_items(user_vector, item_vectors)
        true_items = user_data['asin'].tolist()

        # Debugging output
        print(f"User {user_id}: Recommended Items: {recommended_items}, True Items: {true_items}")

        precision, recall, f_measure, ndcg = evaluate_recommendations(recommended_items, true_items)
        print(f"User {user_id}: Precision: {precision}, Recall: {recall}, F-measure: {f_measure}, NDCG: {ndcg}")

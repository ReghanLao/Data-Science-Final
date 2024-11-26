#!/usr/bin/env python3
import sys
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the embedding model
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])


def mapper():
    train_data = []
    test_data = []
    user_item_reviews = {}

    # Read input data and separate into train and test sets
    for line in sys.stdin:
        try:
            data = json.loads(line)
            asin = data.get('asin')
            review_text = data.get('reviewText')
            reviewerID = data.get('reviewerID')
            overall = data.get('overall')

            if asin and review_text and reviewerID and overall is not None:
                review_embedding = model.encode(review_text).tolist()
                review_info = {
                    'asin': asin,
                    'embedding': review_embedding,
                    'overall': overall,
                    'reviewerID': reviewerID
                }

                # Assign to train (80%) or test (20%)
                if random.random() < 0.8:
                    train_data.append(review_info)
                    if reviewerID not in user_item_reviews:
                        user_item_reviews[reviewerID] = {}
                    user_item_reviews[reviewerID][asin] = review_info
                else:
                    test_data.append(review_info)

        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Create user vectors from training data
    user_vectors = {}
    item_vectors = {}

    for review in train_data:
        reviewerID = review['reviewerID']
        embedding = np.array(review['embedding'])

        if reviewerID not in user_vectors:
            user_vectors[reviewerID] = []

        user_vectors[reviewerID].append(embedding)

        # Store item vectors
        item_vectors[review['asin']] = embedding

    # Average the embeddings to create user vectors
    for user, embeddings in user_vectors.items():
        user_vectors[user] = np.mean(embeddings, axis=0)

    # Prepare output dictionary for predictions
    output_dict = {}

    # Predict ratings for test data
    for review in test_data:
        reviewerID = review['reviewerID']
        asin = review['asin']

        # Only predict if the item is not in user's training data
        if reviewerID in user_item_reviews and asin not in user_item_reviews[reviewerID]:
            if reviewerID in user_vectors and asin in item_vectors:
                user_vector = user_vectors[reviewerID].reshape(1, -1)
                item_vector = item_vectors[asin].reshape(1, -1)

                # Calculate similarity (cosine similarity)
                similarity_score = cosine_similarity(user_vector, item_vector)[0][0]

                # Scale similarity score to a rating scale (e.g., 1-5)
                predicted_rating = 1 + 4 * similarity_score  # Assuming ratings are from 1 to 5

                output_dict.setdefault(reviewerID, []).append({
                    'asin': asin,
                    'predicted_rating': predicted_rating,
                    'true_rating': review['overall']
                })

    # Print output as JSON
    print(json.dumps(output_dict))


if __name__ == "__main__":
    mapper()
#!/usr/bin/env python3
import sys
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Initialize the embedding model
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])


def mapper():
    user_data, item_reviews, all_items = {}, {}, set()
    global_true_ratings = {}

    # Read input data
    for line in sys.stdin:
        try:
            data = json.loads(line)  # Parse JSON input
            asin = data.get('asin')
            review_text = data.get('reviewText', '').strip()
            reviewerID = data.get('reviewerID')
            overall = data.get('overall')

            # Only proceed if all necessary fields are present
            if asin and review_text and reviewerID and overall is not None:
                if reviewerID not in user_data:
                    user_data[reviewerID] = {'train': [], 'test': []}

                # Randomly assign data to train/test sets (80/20 split)
                data_type = 'train' if random.random() < 0.8 else 'test'
                review_embedding = model.encode(review_text).tolist()
                review_info = {'asin': asin, 'embedding': np.array(review_embedding), 'overall': overall}

                user_data[reviewerID][data_type].append(review_info)

                if data_type == 'train':
                    item_reviews.setdefault(asin, []).append(review_info)
                else:
                    # Collect true ratings globally from test data
                    if asin not in global_true_ratings:
                        global_true_ratings[asin] = []
                    global_true_ratings[asin].append(overall)

                all_items.add(asin)

        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Compute item vectors using weighted average of review embeddings
    item_vectors = {}
    for item, reviews in item_reviews.items():
        item_weights = np.array([review['overall'] for review in reviews])
        if np.sum(item_weights) > 0:
            item_vectors[item] = sum(
                review['embedding'] * weight for review, weight in zip(reviews, item_weights)) / np.sum(item_weights)

    # Populate test set data correctly during the mapper phase
    for user, data in user_data.items():
        train_data = data['train']

        # Skip if there is no training data
        if not train_data:
            continue

        train_weights = np.array([review['overall'] for review in train_data])
        if np.sum(train_weights) == 0:
            continue

        user_vector = sum(review['embedding'] * weight for review, weight in zip(train_data, train_weights)) / np.sum(
            train_weights)

        predicted_ratings = []

        # Generate recommendations for all items
        for item in all_items:
            if item in item_vectors:
                item_vector_norm = np.linalg.norm(item_vectors[item]) + 1e-9
                user_vector_norm = np.linalg.norm(user_vector) + 1e-9

                # Calculate cosine similarity between user vector and item vector
                similarity = np.dot(item_vectors[item], user_vector) / (item_vector_norm * user_vector_norm)

                predicted_rating = sum(similarity * review['overall'] for review in item_reviews.get(item, [])) / (
                        sum(abs(similarity) for review in item_reviews.get(item, [])) + 1e-9)

                predicted_ratings.append([item, predicted_rating])

        # Sort recommendations by predicted rating
        predicted_ratings.sort(key=lambda x: -x[1])

        # Collect true ratings only for items that have been predicted
        true_ratings_dict = {item: global_true_ratings[item] for item, _ in predicted_ratings if
                             item in global_true_ratings}

        # Output recommendations and true ratings for evaluation
        output = {
            'ratings': predicted_ratings,
            'true_ratings': true_ratings_dict  # Only include true ratings that correspond to predicted items globally
        }

        print(f"{user}\t{json.dumps(output)}")


if __name__ == "__main__":
    mapper()
#!/usr/bin/env python3
import sys
import json
import random
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Initialize a StaticEmbedding module with a Model2Vec model
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")  # Update with your Model2Vec model
model = SentenceTransformer(modules=[static_embedding])


def compute_embeddings(texts):
    """Compute the embedding for a given text."""
    return model.encode(texts).tolist()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def mapper():
    user_data = defaultdict(lambda: {'train': [], 'test': []})
    all_items = set()

    # Read input from stdin
    for line in sys.stdin:
        try:
            data = json.loads(line)
            asin = data.get('asin')
            review_text = data.get('reviewText')
            reviewerID = data.get('reviewerID')
            overall = data.get('overall')

            if asin and review_text and reviewerID is not None and overall is not None:
                # Randomly assign the review to either training (80%) or testing (20%) set
                data_type = 'train' if random.random() < 0.8 else 'test'

                # Prepare the review text and compute the embedding
                review_embedding = compute_embeddings([review_text])[0]  # Get the embedding for the review

                # Store review data in user_data dictionary
                review_info = {
                    'reviewerID': reviewerID,
                    'asin': asin,
                    'embedding': review_embedding,
                    'overall': overall,
                    'type': data_type
                }
                user_data[reviewerID][data_type].append(review_info)
                all_items.add(asin)

        except json.JSONDecodeError:
            continue  # Skip malformed lines

    # Calculate average ratings and item similarities for each user in mapper
    for user, data in user_data.items():
        train_data = data['train']
        test_data = data['test']

        if not train_data or not test_data:
            continue

        avg_rating = np.mean([review['overall'] for review in train_data])
        item_embeddings = {review['asin']: review['embedding'] for review in train_data}

        recommendations = []
        for item in all_items:
            if item not in item_embeddings:
                continue
            avg_similarity = np.mean(
                [cosine_similarity(item_embeddings[item], review['embedding']) for review in train_data])
            recommendations.append((item, avg_similarity))

        recommendations.sort(key=lambda x: -x[1])  # Sort by similarity descending
        top_10_recommendations = [item for item, _ in recommendations[:10]]

        # Emit user recommendations with average rating and top recommendations
        output = {
            'user': user,
            'avg_rating': avg_rating,
            'top_recommendations': top_10_recommendations,
            'test_data': test_data  # Pass test data to reducer for evaluation
        }

        print(f"{user}\t{json.dumps(output)}")


if __name__ == "__main__":
    mapper()
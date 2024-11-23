import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def recommend_items(user_vector, item_vectors, n=10, threshold=0.1):
    similarities = cosine_similarity([user_vector], item_vectors)[0]
    # Filter items based on the threshold
    filtered_indices = np.where(similarities >= threshold)[0]
    if len(filtered_indices) == 0:
        return []  # No items meet the threshold

    top_indices = filtered_indices[np.argsort(similarities[filtered_indices])[-n:][::-1]]
    return top_indices


def evaluate_recommendations(recommended_items, true_items):
    if len(recommended_items) == 0:
        return 0, 0, 0, 0  # Return zero metrics if no recommendations

    precision = len(set(recommended_items) & set(true_items)) / len(recommended_items)
    recall = len(set(recommended_items) & set(true_items)) / len(true_items)
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    dcg = sum((1 / np.log2(i + 2) for i, item in enumerate(recommended_items) if item in true_items))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), len(recommended_items))))
    ndcg = dcg / idcg if idcg > 0 else 0

    return precision, recall, f_measure, ndcg
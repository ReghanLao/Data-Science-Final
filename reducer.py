#!/usr/bin/env python3
import sys
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, ndcg_score


def calculate_ndcg(true_ratings, predicted_ratings, k=10):
    return ndcg_score([true_ratings], [predicted_ratings], k=k)


def reducer():
    global_predictions = []
    global_true_ratings = []
    user_recommendations = {}

    # Read and process input from mappers
    for line in sys.stdin:
        user_info = json.loads(line.strip().split('\t')[1])

        user = user_info['user']
        avg_rating = user_info['avg_rating']
        top_recommendations = user_info['top_recommendations']
        test_data = user_info['test_data']

        # Store recommendations for each user
        user_recommendations[user] = top_recommendations

        for test_review in test_data:
            true_rating = test_review['overall']

            predicted_rating = avg_rating  # Start with average rating
            global_predictions.append(predicted_rating)
            global_true_ratings.append(true_rating)

    # Calculate global metrics
    mae = mean_absolute_error(global_true_ratings, global_predictions)
    rmse = np.sqrt(mean_squared_error(global_true_ratings, global_predictions))

    binary_true = [1 if r >= 4 else 0 for r in global_true_ratings]
    binary_pred = [1 if r >= 4 else 0 for r in global_predictions]

    precision = precision_score(binary_true, binary_pred)
    recall = recall_score(binary_true, binary_pred)
    f_measure = f1_score(binary_true, binary_pred)

    ndcg_value = calculate_ndcg(global_true_ratings, global_predictions)

    # Print global metrics
    print(f"MAE\t{mae:.4f}")
    print(f"RMSE\t{rmse:.4f}")
    print(f"Precision\t{precision:.4f}")
    print(f"Recall\t{recall:.4f}")
    print(f"F-measure\t{f_measure:.4f}")
    print(f"NDCG\t{ndcg_value:.4f}")

    # Write recommendations to a text file
    with open("output/user_recommendations.txt", "w") as f:
        f.writelines(
            f"{user}\t{', '.join(recommendations)}\n" for user, recommendations in user_recommendations.items())


if __name__ == "__main__":
    reducer()
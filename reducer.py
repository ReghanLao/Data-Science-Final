#!/usr/bin/env python3
import sys
import json
import numpy as np

def reducer(output_file):
    user_recommendations = {}
    predictions = []
    true_ratings = []

    # Read input data from stdin
    for line in sys.stdin:
        try:
            data = json.loads(line)
            for reviewerID, reviews in data.items():
                if reviewerID not in user_recommendations:
                    user_recommendations[reviewerID] = []
                user_recommendations[reviewerID].extend(reviews)

                # Collect predictions and true ratings for MAE and RMSE calculation
                for review in reviews:
                    predictions.append(review['predicted_rating'])
                    true_ratings.append(review['true_rating'])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # Prepare top 10 recommendations for each user
    top_recommendations = {}
    for reviewerID, reviews in user_recommendations.items():
        # Sort reviews by predicted rating in descending order
        sorted_reviews = sorted(reviews, key=lambda x: x['predicted_rating'], reverse=True)
        # Check if we have >= 10 recommendations
        if len(sorted_reviews) >= 10:
            top_recommendations[reviewerID] = sorted_reviews[:10]

    # Write top recommendations to a file
    with open(output_file, 'w') as f:
        json.dump(top_recommendations, f, indent=4)

    # Calculate MAE and RMSE
    predictions = np.array(predictions)
    true_ratings = np.array(true_ratings)

    mae = np.mean(np.abs(predictions - true_ratings))
    rmse = np.sqrt(np.mean((predictions - true_ratings) ** 2))

    # Write results to a text file
    with open('mae_rmse.txt', 'w') as f:
        f.write(f'Mean Absolute Error (MAE): {mae:.4f}\n')
        f.write(f'Root Mean Square Error (RMSE): {rmse:.4f}\n')

if __name__ == "__main__":
    # Specify the output file name for recommendations
    output_file_name = 'top_recommendations.json'
    reducer(output_file_name)
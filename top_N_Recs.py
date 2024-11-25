import numpy as np

def recommend_top_n(userId, similarityMatrix, userItemMatrixSparse, userMap, itemMap, n=10):
    if userId not in userMap:
        return []

    userIndex = userMap[userId]
    userRatings = userItemMatrixSparse[userIndex, :].toarray().flatten()

    unratedItems = np.where(userRatings == 0)[0]

    predictedRatings = []
    for itemIdx in unratedItems:
        itemSimilarities = similarityMatrix.iloc[itemIdx, :].to_numpy()
        weightedSum = np.dot(userRatings, itemSimilarities)
        similaritySum = np.sum(np.abs(itemSimilarities[userRatings > 0]))

        predictedRating = weightedSum / similaritySum if similaritySum > 0 else 0
        predictedRatings.append((itemIdx, predictedRating))

    topNItems = sorted(predictedRatings, key=lambda x: x[1], reverse=True)[:n]

    reverseItemMap = {idx: item for item, idx in itemMap.items()}
    return [reverseItemMap[itemIdx] for itemIdx, _ in topNItems]

def evaluate_recommendations(test_data, similarity_matrix, user_item_matrix_sparse, user_map, item_map, n=10):
    precision_list = []
    recall_list = []
    ndcg_list = []

    groups = test_data.groupby("reviewerID")
    for user_id, group in groups:
        ground_truth_items = group['asin'].values

        recommended_items = recommend_top_n(
            user_id, similarity_matrix, user_item_matrix_sparse, user_map, item_map, n
        )

        hits = len(set(recommended_items) & set(ground_truth_items))
        precision = hits / n
        recall = hits / len(ground_truth_items) if len(ground_truth_items) > 0 else 0

        dcg = sum([
            1 / np.log2(idx + 2)
            for idx, item in enumerate(recommended_items) if item in ground_truth_items
        ])
        idcg = sum([
            1 / np.log2(idx + 2)
            for idx in range(min(len(ground_truth_items), n))
        ])
        ndcg = dcg / idcg if idcg > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

    return {
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "NDCG": np.mean(ndcg_list)
    }
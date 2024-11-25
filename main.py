import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc  # Garbage collector
from tqdm import tqdm  # For progress bars
import time
import warnings
warnings.filterwarnings('ignore')

def print_time_estimate(start_time):
    """Print elapsed time in a human readable format"""
    elapsed = time.time() - start_time
    print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

print("Starting recommendation system...")
total_start = time.time()

# Load data with chunks to handle memory better
print("Loading data...")
start_time = time.time()

chunks = []
chunk_size = 100000  # Adjust based on your available RAM
for chunk in pd.read_json("Sports_and_Outdoors_5", lines=True, chunksize=chunk_size):
    chunks.append(chunk[['reviewerID', 'asin', 'overall']])
df = pd.concat(chunks)
del chunks
gc.collect()

print(f"Data loaded. {len(df)} records found.")
print_time_estimate(start_time)

# Basic preprocessing
print("\nPreprocessing data...")
start_time = time.time()

df.columns = ['user_id', 'item_id', 'rating']

# Create mappings using numerical index
print("Creating user/item mappings...")
user_mapping = {user: idx for idx, user in enumerate(df['user_id'].unique())}
item_mapping = {item: idx for idx, item in enumerate(df['item_id'].unique())}

df['user_idx'] = df['user_id'].map(user_mapping)
df['item_idx'] = df['item_id'].map(item_mapping)

num_users = len(user_mapping)
num_items = len(item_mapping)

print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")
print(f"Number of ratings: {len(df)}")
print(f"Sparsity: {(1 - len(df)/(num_users*num_items))*100:.2f}%")
print_time_estimate(start_time)

# Memory efficient train/test split
print("\nSplitting data...")
start_time = time.time()

def train_test_split_by_user(df, test_size=0.2):
    train_mask = np.zeros(len(df), dtype=bool)
    for user_id in tqdm(df['user_idx'].unique(), desc="Splitting data by user"):
        user_mask = df['user_idx'] == user_id
        user_indices = np.where(user_mask)[0]
        n_test = max(1, int(len(user_indices) * test_size))
        train_indices = np.random.choice(user_indices, len(user_indices) - n_test, replace=False)
        train_mask[train_indices] = True
    return df[train_mask], df[~train_mask]

train_df, test_df = train_test_split_by_user(df, test_size=0.2)
print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")
print_time_estimate(start_time)

# Create sparse matrix
print("\nCreating sparse matrix...")
start_time = time.time()

train_sparse = csr_matrix(
    (train_df['rating'], (train_df['user_idx'], train_df['item_idx'])),
    shape=(num_users, num_items)
)
print_time_estimate(start_time)

# Perform SVD
print("\nPerforming SVD...")
start_time = time.time()

n_components = 50
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(train_sparse)
item_factors = svd.components_

# Free memory
del train_sparse
gc.collect()

print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
print_time_estimate(start_time)

def evaluate_predictions_batch(test_df, user_factors, item_factors, batch_size=10000):
    """Memory efficient batch prediction evaluation"""
    total_mae = 0
    total_mse = 0
    total_samples = 0
    
    for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Evaluating predictions"):
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        
        # Get predictions for batch
        batch_predictions = np.sum(
            user_factors[batch_df['user_idx']] * item_factors[:, batch_df['item_idx']].T,
            axis=1
        )
        
        # Clip predictions
        batch_predictions = np.clip(batch_predictions, 1, 5)
        
        # Update metrics
        batch_mae = np.sum(np.abs(batch_predictions - batch_df['rating']))
        batch_mse = np.sum(np.square(batch_predictions - batch_df['rating']))
        
        total_mae += batch_mae
        total_mse += batch_mse
        total_samples += len(batch_df)
    
    mae = total_mae / total_samples
    rmse = np.sqrt(total_mse / total_samples)
    
    return mae, rmse

# Evaluate predictions
print("\nEvaluating rating predictions...")
start_time = time.time()
mae, rmse = evaluate_predictions_batch(test_df, user_factors, item_factors)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print_time_estimate(start_time)

def generate_recommendations_batch(test_df, user_factors, item_factors, train_df, 
                                n=10, batch_size=1000):
    """Memory efficient batch recommendation generation"""
    recommendations = {}
    test_users = test_df['user_idx'].unique()
    
    for start_idx in tqdm(range(0, len(test_users), batch_size), desc="Generating recommendations"):
        end_idx = min(start_idx + batch_size, len(test_users))
        batch_users = test_users[start_idx:end_idx]
        
        # Generate predictions for batch
        batch_predictions = np.dot(user_factors[batch_users], item_factors)
        
        for i, user_idx in enumerate(batch_users):
            # Mask out items the user has already rated
            rated_items = set(train_df[train_df['user_idx'] == user_idx]['item_idx'])
            user_predictions = batch_predictions[i]
            mask = np.ones(len(user_predictions), dtype=bool)
            mask[list(rated_items)] = False
            
            # Get top N items
            top_items = np.argsort(user_predictions[mask])[-n:][::-1]
            recommendations[user_idx] = top_items
    
    return recommendations

# Generate recommendations
print("\nGenerating recommendations...")
start_time = time.time()
recommendations = generate_recommendations_batch(test_df, user_factors, item_factors, train_df)
print_time_estimate(start_time)

# Calculate metrics
print("\nCalculating recommendation metrics...")
start_time = time.time()

metrics = {
    'precision': [],
    'recall': [],
    'ndcg': [],
    'f1': []
}

for user_idx in tqdm(test_df['user_idx'].unique(), desc="Calculating metrics"):
    ground_truth = set(test_df[test_df['user_idx'] == user_idx]['item_idx'])
    if len(ground_truth) == 0:
        continue
    
    if user_idx not in recommendations:
        continue
        
    recommended = set(recommendations[user_idx])
    
    # Calculate basic metrics
    hits = len(ground_truth.intersection(recommended))
    precision = hits / len(recommended)
    recall = hits / len(ground_truth)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate NDCG
    dcg = sum([1 / np.log2(i + 2) for i, item in 
               enumerate(recommendations[user_idx]) if item in ground_truth])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), 10))])
    ndcg = dcg / idcg if idcg > 0 else 0
    
    metrics['precision'].append(precision)
    metrics['recall'].append(recall)
    metrics['f1'].append(f1)
    metrics['ndcg'].append(ndcg)

# Print final metrics
print("\nFinal Metrics:")
print(f"Precision@10: {np.mean(metrics['precision']):.4f}")
print(f"Recall@10: {np.mean(metrics['recall']):.4f}")
print(f"F1@10: {np.mean(metrics['f1']):.4f}")
print(f"NDCG@10: {np.mean(metrics['ndcg']):.4f}")

print("\nTotal runtime:")
print_time_estimate(total_start)
from load_data import load_data
from clean_data import clean_data
from rating_prediction import create_user_item_matrix_sparse
from split_data import split_data

# Load and clean data
data = load_data("Sports_and_Outdoors_5.json") 
cleaned_data = clean_data(data)

# Split data into training and testing 
train_data, test_data = split_data(cleaned_data)

# Create sparse user-item matrix
user_item_matrix_sparse, user_map, item_map = create_user_item_matrix_sparse(train_data)


from load_data import load_data
from clean_data import clean_data
from rating_prediction import create_user_item_matrx

data = load_data("Sports_and_Outdoors_5.json") 

cleaned_data = clean_data(data)

user_item_matrix = create_user_item_matrx(cleaned_data)

print(user_item_matrix.head())




import numpy as np
import pandas as pd
from randomForest import random_forest_algorithm, random_forest_predictions, calculate_accuracy
import joblib
import time

np.random.seed(42)

df = pd.read_csv('../data/extracted_features_test.csv')

forest = joblib.load('rf_model_new.pkl')
predicted_class = random_forest_predictions(df, forest)

print("RF Predicted Class:", predicted_class)

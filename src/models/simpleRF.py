
import numpy as np
import pandas as pd
from randomForest import random_forest_algorithm, random_forest_predictions, calculate_accuracy
import joblib
import time

np.random.seed(42)

df = pd.read_csv('../data/extracted_features_gl.csv')

hyperparameters = {'n_trees': 6, 'n_bootstrap': 1200, 'n_features': 3, 'dt_max_depth': 5}

start = time.time()
forest = random_forest_algorithm(df, **hyperparameters)
end = time.time()
train = end-start
print(train)

start=time.time()
predictions = random_forest_predictions(df, forest)
end=time.time()
test=end-start
print(test)

accuracy = calculate_accuracy(predictions, df['label'])

print("Hyperparameters:", hyperparameters)
print("Accuracy:", accuracy)

joblib.dump(forest, 'rf_model_new.pkl')



'''
#Optional Code for Tuning Hyperparameters

param_grid = {
    'n_trees': [4, 6],
    'n_bootstrap': [400, 800, 1200],
    'n_features': [1, 2, 3],
    'dt_max_depth': [3, 4, 5]
}

param_combinations = list(itertools.product(*param_grid.values()))

results = []

for params in param_combinations:
    param_dict = dict(zip(param_grid.keys(), params))
    fold_results = {'params': param_dict, 'train_accuracies': [], 'test_accuracies': []}

    for train_index, test_index in skf.split(df, df['label']):
        train_df, test_df = df.iloc[train_index], df.iloc[test_index]

        # Train the random forest
        forest = random_forest_algorithm(train_df, **param_dict)

        # Predictions on training set
        train_predictions = random_forest_predictions(train_df, forest)
        train_accuracy = calculate_accuracy(train_predictions, train_df['label'])
        fold_results['train_accuracies'].append(train_accuracy)

        # Predictions on testing set
        test_predictions = random_forest_predictions(test_df, forest)
        test_accuracy = calculate_accuracy(test_predictions, test_df['label'])
        fold_results['test_accuracies'].append(test_accuracy)

    avg_train_accuracy = np.mean(fold_results['train_accuracies'])
    avg_test_accuracy = np.mean(fold_results['test_accuracies'])

    result_entry = {'params': param_dict, 'avg_train_accuracy': avg_train_accuracy, 'avg_test_accuracy': avg_test_accuracy}
    results.append(result_entry)

    print("Hyperparameters:", param_dict)
    print("Average Training Accuracy:", avg_train_accuracy)
    print("Average Testing Accuracy:", avg_test_accuracy)
    print()

results.sort(key=lambda x: x['avg_test_accuracy'], reverse=True)

top_5_results = results[:5]

for i, result in enumerate(top_5_results):
    print(f"Top {i + 1}: Hyperparameters: {result['params']}, Average Training Accuracy: {result['avg_train_accuracy']}, Average Testing Accuracy: {result['avg_test_accuracy']}")
'''

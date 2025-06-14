# Sources: https://github.com/harrypnh/random-forest-from-scratch
#          https://www.youtube.com/watch?v=WvmPnGmCaIM&list=PLPOTBrypY74y0DviMOagKRUhDdk0JyM_r

import numpy as np
import pandas as pd
import random
from pprint import pprint

from decisionTree import decision_tree_algorithm, decision_tree_predictions, train_test_split, calculate_accuracy

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    
    return forest

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions
    
#use case : 
#forest = random_forest_algorithm(train_df, n_trees=4, n_bootstrap=800, n_features=2, dt_max_depth=4)
#predictions = random_forest_predictions(test_df, forest)
#accuracy = calculate_accuracy(predictions, test_df.label)
#print("Accuracy = {}".format(accuracy))

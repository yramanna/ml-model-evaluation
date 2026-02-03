from itertools import product
import pandas as pd
from model import DecisionTree, AdaBoostClassifier
import numpy as np
import argparse
from data import load_data

def cross_validation(cv_folds: list, model_type: str, max_depth_values: list, min_samples_split_values: list, criteria: list, n_estimators: int):
    best_score = -1
    best_params = {}
    for max_depth, min_samples_split, criterion in product(max_depth_values, min_samples_split_values, criteria):
        scores = []
        for i in range(len(cv_folds)):
            valid = cv_folds[i]
            train_folds = pd.concat([f for j, f in enumerate(cv_folds) if j != i])
            x_train, y_train = train_folds.iloc[:, :-1], train_folds.iloc[:, -1]
            x_valid, y_valid = valid.iloc[:, :-1], valid.iloc[:, -1]

            if model_type == 'adaboost':
                model = AdaBoostClassifier(DecisionTree, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
            else:
                model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

            model.train(x_train, y_train)
            predictions = model.predict(x_valid)
            accuracy = sum(predictions[i] == y_valid.iloc[i] for i in range(len(predictions))) / len(predictions)
            scores.append(accuracy)

        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'criterion': criterion}

    return best_params, best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['decision_tree', 'adaboost'], default='adaboost')
    parser.add_argument('--max_depth_values', '-d', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--min_samples_split_values', '-s', nargs='+', type=int, default=[2, 5])
    parser.add_argument('--criteria', '-c', nargs='+', type=str, choices=['entropy', 'gini'], default=['entropy'])
    parser.add_argument('--folds', '-k', type=int, default=5)
    parser.add_argument('--n_estimators', '-n', type=int, default=50)
    args = parser.parse_args()

    data_dict = load_data()
    df = data_dict['train'].sample(frac=1, random_state=1).reset_index(drop=True)
    cv_folds = np.array_split(df, args.folds)

    best_hyperparams, best_accuracy = cross_validation(cv_folds, args.model, args.max_depth_values, args.min_samples_split_values, args.criteria, args.n_estimators)

    print('\nBest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\nAccuracy: {best_accuracy:.3f}\n')
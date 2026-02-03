from itertools import product
import pandas as pd
from model import DecisionTree
import numpy as np
import argparse
from data import load_data

from model import Model, DecisionTree, MODEL_OPTIONS


def init_model(model_name: str, max_depth: int, min_samples_split: int, criterion: str) -> Model:
    '''
    Initialize the appropriate model with corresponding hyperparameters.

    Args:
        model_name (str): which model to use. Should be either "svm" or "logistic_regression"
        lr0 (float): the initial learning rate (gamma_0)
        reg_tradeoff (float): the regularization/loss tradeoff hyperparameter for SVM and Logistic Regression
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the corresponding hyperparameters
    '''

    if model_name == 'decision_tree':
        return DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    
    raise ValueError(f"Unknown model name: {model_name}")

def cross_validation(cv_folds: list, max_depth_values: list, min_samples_split_values: list, criteria: list):
    '''
    Perform grid search over hyperparameters.

    Returns:
        dict: best hyperparameter combination
        float: average accuracy
    '''
    best_score = -1
    best_params = {}

    for max_depth, min_samples_split, criterion in product(max_depth_values, min_samples_split_values, criteria):
        scores = []

        for i in range(len(cv_folds)):
            valid = cv_folds[i]
            train_folds = pd.concat([f for j, f in enumerate(cv_folds) if j != i])
            x_train, y_train = train_folds.iloc[:, :-1], train_folds.iloc[:, -1]
            x_valid, y_valid = valid.iloc[:, :-1], valid.iloc[:, -1]
            model = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
            model.train(x_train, y_train)
            predictions = model.predict(x_valid)
            accuracy = sum(predictions[i] == y_valid.iloc[i] for i in range(len(predictions))) / len(predictions)
            scores.append(accuracy)

        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'criterion': criterion
            }

    return best_params, best_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a decision tree using cross-validation')
    parser.add_argument('--model', '-m', type=str, choices=MODEL_OPTIONS,
                        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--max_depth_values', '-d', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6],
                        help='List of maximum depths to try.')
    parser.add_argument('--min_samples_split_values', '-s', nargs='+', type=int, default=[2, 5, 10],
                        help='List of minimum samples split values to try.')
    parser.add_argument('--criteria', '-c', nargs='+', type=str, choices=['entropy', 'gini'], default=['entropy'],
                        help='List of information gain criteria to try.')
    parser.add_argument('--folds', '-k', type=int, default=5,
                        help='Number of cross-validation folds to create.')
    args = parser.parse_args()

    # Load data
    print('Loading data...')
    data_dict = load_data()
    df = data_dict['train'].sample(frac=1, random_state=1).reset_index(drop=True)
    cv_folds = np.array_split(df, args.folds)

    # Run cross-validation
    print('Running cross-validation...')
    best_hyperparams, best_accuracy = cross_validation(
        cv_folds=cv_folds,
        max_depth_values=args.max_depth_values,
        min_samples_split_values=args.min_samples_split_values,
        criteria=args.criteria
    )

    # Print best hyperparameters and accuracy
    print('\nBest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\nAccuracy: {best_accuracy:.3f}\n')

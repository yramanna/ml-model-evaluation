import argparse
import itertools
from typing import List, Tuple
import numpy as np
import pandas as pd
from train import remove_zero_features
from data import load_data, DATASET_OPTIONS
from evaluate import accuracy
from model import LogisticRegression, Model, SupportVectorMachine, MODEL_OPTIONS

def init_model(model_name: str, lr0: float, reg_tradeoff: float, num_features: int) -> Model:
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

    if model_name == 'svm':
        model = SupportVectorMachine(num_features=num_features, lr0=lr0, C=reg_tradeoff)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(num_features=num_features, lr0=lr0, sigma2=reg_tradeoff)
    return model

def cross_validation(
        cv_folds: List[pd.DataFrame], 
        model_name: str, 
        lr0_values: list, 
        reg_tradeoff_values: list,
        epochs: int = 5) -> Tuple[dict, float]:
    '''
    Run cross-validation to determine the best hyperparameters.

    Args:
        cv_folds (list): a list of pandas DataFrames, corresponding to folds of the data. 
            The last column of each DataFrame, called "label", corresponds to y, 
            while the remaining columns are the features x.
        model_name (str): which model to use. Should be either "svm" or "logistic_regression"
        lr0_values (list): a list of initial learning rate values to try. 
            Equivalent to "C" hyperparam for SVM, or "sigma2" hyperparam for Logistic Regression.
        reg_tradeoff_values (list): a list of regularization tradeoff values to try.
        epochs (int): how many epochs to train each model for. Defaults to 5

    Returns:
        dict: a dictionary with the best hyperparameters discovered during cross-validation
        float: the average cross-validation accuracy corresponding to the best hyperparameters

    Hints:
        - We've provided a helper function `init_model()` above to initialize your model. 
        - The python `itertools.product()` function returns the Cartesian product of multiple lists.
        - You can convert a pandas DataFrame to a numpy ndarray with `df.to_numpy()`
    '''

    best_hyperparams = {'lr0': None, 'reg_tradeoff': None}
    best_avg_accuracy = 0
    num_features = cv_folds[0].shape[1] - 1


    # YOUR CODE HERE
    for lr0, reg in itertools.product(lr0_values, reg_tradeoff_values):
        fold_accuracies = []

        # for each fold i, train on all but i, validate on i
        for i in range(len(cv_folds)):
            # validation split
            val_df = cv_folds[i]
            # training splits
            train_dfs = [cv_folds[j] for j in range(len(cv_folds)) if j != i]
            train_df = pd.concat(train_dfs, ignore_index=True)

            # separate features / labels
            X_train = train_df.iloc[:, :-1].to_numpy()
            y_train = train_df['label'].to_numpy()
            X_val   = val_df.iloc[:, :-1].to_numpy()
            y_val   = val_df['label'].to_numpy()

            # initialize, train, predict
            model = init_model(model_name, lr0, reg, num_features)
            model.train(X_train, y_train, epochs)
            y_pred = model.predict(X_val)

            # evaluate
            fold_accuracies.append(accuracy(y_val, y_pred))

        avg_acc = sum(fold_accuracies) / len(fold_accuracies)
        if avg_acc > best_avg_accuracy:
            best_avg_accuracy = avg_acc
            best_hyperparams['lr0'] = lr0
            best_hyperparams['reg_tradeoff'] = reg

    return best_hyperparams, best_avg_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-validation for different hyperparameters')
    parser.add_argument('--model', '-m', type=str, choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--dataset', '-d', type=str, default='100', choices=DATASET_OPTIONS, 
        help=f'Which dataset to use. Must be one of {DATASET_OPTIONS}. Defaults to "100".')
    parser.add_argument('--lr0_values', nargs='+', type=float, default=[0.1],
        help='A list (space separated) of initial learning rate values to try. Defaults to [0.1].')
    parser.add_argument('--reg_tradeoff_values', nargs='+', type=float, default=[1],
        help=(
            'A list (space separated) of regularization tradeoff values to try. '
            'Equivalent to "C" hyperparam for SVM, or "sigma2" hyperparam for Logistic Regression. Defaults to [1].'))
    parser.add_argument('--epochs', '-e', type=int, default=5,
        help='How many epochs to train for. Defaults to 5.')
    parser.add_argument(
        '--folds', '-k', type=int, default=5,
        help='Number of cross-validation folds to create.'
    )
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data(args.dataset)
    df = data_dict['train']\
       .sample(frac=1, random_state=1)\
       .reset_index(drop=True)
    df = remove_zero_features(df)
    cv_folds = np.array_split(df, args.folds)

    # run cross_validation
    print(f'run cross-validation')
    best_hyperparams, best_accuracy = cross_validation(
        cv_folds=cv_folds, 
        model_name=args.model,
        lr0_values=args.lr0_values, 
        reg_tradeoff_values=args.reg_tradeoff_values, 
        epochs=args.epochs)
    
    # print best hyperparameters and accuracy
    print('\nbest hyperparameters from cross-validation:\n')
    for name, value in best_hyperparams.items():
        print(f'{name:>15}: {value}')
    print(f'\naccuracy: {best_accuracy:.3f}\n')

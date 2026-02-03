''' This file contains the code for training and evaluating a model.
    You don't need to change this file.
'''

import argparse
import pandas as pd
from data import load_data, DATASET_OPTIONS
from sklearn.feature_selection import VarianceThreshold
from evaluate import accuracy
from model import LogisticRegression, MajorityBaseline, Model, SupportVectorMachine, MODEL_OPTIONS

def create_submission_file(eval_ids_file: str, predictions: list, output_file: str):
    """
    Create a CSV submission file for Kaggle using the given example IDs and predictions.
    
    Args:
        eval_ids_file (str): Path to the file containing the example IDs.
        predictions (list): List of predictions (either 0 or 1) for each example.
        output_file (str): Path to save the generated CSV submission file.
    """
    # Read the example IDs from the eval.ids file
    with open(eval_ids_file, 'r') as f:
        eval_ids = f.readlines()

    # Clean up any trailing whitespace or newline characters
    eval_ids = [id.strip() for id in eval_ids]

    # Create a DataFrame with the example_id and label (prediction)
    submission_df = pd.DataFrame({
        'example_id': eval_ids,
        'label': predictions
    })

    # Save the DataFrame to a CSV file with the appropriate header
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved as: {output_file}")

def remove_zero_features(df, variance_threshold=0.01):
    """
    Remove features that are entirely zero and apply variance threshold to further reduce features.

    Args:
    - df (pd.DataFrame): Input dataset in the form of a pandas DataFrame.
    - zero_threshold (float): Not used here since we remove features that are fully zero.
    - variance_threshold (float): Threshold for removing low-variance features.

    Returns:
    - pd.DataFrame: A new DataFrame with all-zero and low-variance features removed.
    """
    # Exclude 'label' column from processing
    feature_df = df.drop(columns=['label'], errors='ignore')
    
    # Remove all-zero columns
    non_zero_columns = feature_df.loc[:, (feature_df != 0).any(axis=0)].columns
    X = feature_df[non_zero_columns]
    
    # Apply VarianceThreshold
    selector = VarianceThreshold(threshold=variance_threshold)
    X_reduced = selector.fit_transform(X)
    selected_columns = X.columns[selector.get_support()]

    # Reconstruct DataFrame
    transformed_df = pd.DataFrame(X_reduced, columns=selected_columns, index=df.index)
    if 'label' in df.columns:
        transformed_df['label'] = df['label']

    print(f"Original shape: {df.shape}")
    print(f"Shape after removing all-zero columns and applying variance threshold: {transformed_df.shape}")
    
    return transformed_df

def init_model(args: object, num_features: int) -> Model:
    '''
    Initialize the appropriate model from command-line arguments.

    Args:
        args (object): the argparse Namespace mapping arguments to their values.
        num_features (int): the number of features (i.e. dimensions) the model will have

    Returns:
        Model: a Model object initialized with the hyperparameters in args.
    '''
    if args.model == 'majority_baseline':
        model = MajorityBaseline()
    
    elif args.model == 'svm':
        model = SupportVectorMachine(
            num_features=num_features, 
            lr0=args.lr0, 
            C=args.reg_tradeoff)

    elif args.model == 'logistic_regression':
        model = LogisticRegression(
            num_features=num_features, 
            lr0=args.lr0, 
            sigma2=args.reg_tradeoff)
    return model

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--dataset', '-d', type=str, default='100', choices=DATASET_OPTIONS, 
        help=f'Which dataset to use. Must be one of {DATASET_OPTIONS}. Defaults to "100".')
    parser.add_argument('--lr0', type=float, default=0.1, 
        help='The initial learning rate hyperparameter gamma_0. Defaults to 0.1.')
    parser.add_argument('--reg_tradeoff', type=float, default=1, 
        help='The regularization tradeoff hyperparameter for SVM and Logistic Regression. Defaults to 1.')
    parser.add_argument('--epochs', '-e', type=int, default=20,
        help='How many epochs to train for. Defaults to 20.')
    args = parser.parse_args()

    # load data
    data_dict = load_data()
    train_df = data_dict['train']
    test_df = data_dict['test']
    eval_df = data_dict['eval']
    
    # --- Step 1: Remove all-zero columns (based on train data only) ---
    feature_df = train_df.drop(columns='label')
    non_zero_columns = feature_df.loc[:, (feature_df != 0).any(axis=0)].columns

    # Apply to all splits
    train_df = train_df[non_zero_columns.tolist() + ['label']]
    test_df = test_df[non_zero_columns.tolist() + ['label']]
    eval_df = eval_df[non_zero_columns.tolist() + ['label']]

    # --- Step 2: Apply VarianceThreshold consistently ---
    selector = VarianceThreshold(threshold=0.01)
    train_x = selector.fit_transform(train_df.drop('label', axis=1))
    train_y = train_df['label'].values

    test_x = selector.transform(test_df.drop('label', axis=1))
    test_y = test_df['label'].values

    eval_x = selector.transform(eval_df.drop('label', axis=1))
    eval_y = eval_df['label'].values

    # load the model
    print(f'initialize model')
    model = init_model(args=args, num_features=train_x.shape[1])
    print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

    # train the model
    if args.model == 'majority_baseline':
        print(f'train model')
        model.train(x=train_x, y=train_y)
    else:
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

    # evaluate model on train and test data
    print('evaluate')
    train_predictions = model.predict(x=train_x)
    train_accuracy = accuracy(labels=train_y, predictions=train_predictions)
    print(f'  train accuracy: {train_accuracy:.3f}')
    test_predictions = model.predict(x=test_x)
    test_accuracy = accuracy(labels=test_y, predictions=test_predictions)
    print(f'  test accuracy: {test_accuracy:.3f}')
    eval_predictions = model.predict(x=eval_x)
    eval_accuracy = accuracy(labels=eval_y, predictions=eval_predictions)
    print(f'  eval accuracy: {eval_accuracy:.3f}')
    eval_ids_file = 'data/eval.id'  # Path to the eval.ids file
    output_file = 'submission.csv'  # Path to save the submission file

    # Generate the Kaggle submission file
    create_submission_file(eval_ids_file, eval_predictions, output_file)
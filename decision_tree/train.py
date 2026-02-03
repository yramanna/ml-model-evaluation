''' This file contains the functions for training and evaluating a model.
    You need to add your code wherever you see "YOUR CODE HERE".
'''

import argparse
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from data import load_data
from model import DecisionTree, MajorityBaseline, Model

def train(model: Model, x: pd.DataFrame, y: list):
    '''
    Learn a model from training data.

    Args:
        model (Model): an instantiated MajorityBaseline or DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be trained from
        y (list): a list with the target labels corresponding to each example
    '''
    
    # YOUR CODE HERE
    model.train(x,y)

def evaluate(model: Model, x: pd.DataFrame, y: list) -> float:
    '''
    Evaluate a trained model against a dataset

    Args:
        model (Model): an instance of a MajorityBaseline model or a DecisionTree model
        x (pd.DataFrame): a dataframe with the features the tree will be trained from
        y (list): a list with the target labels corresponding to each example

    Returns:
        float: the accuracy of the decision tree's predictions on x, when compared to y
    '''
    
    # YOUR CODE HERE
    predictions = model.predict(x)
    accuracy = calculate_f1_score(y, predictions)
    return accuracy, predictions

def calculate_f1_score(labels: list, predictions: list) -> float:
    '''
    Calculate the F1 score between ground-truth labels and candidate predictions.
    The F1 score evaluates the balance between precision and recall.

    Args:
        labels (list): the ground-truth labels from the data (0 or 1)
        predictions (list): the predicted labels from the model (0 or 1)

    Returns:
        float: the F1 score of the predictions, when compared to the ground-truth labels
    '''

    # Initialize variables for true positives, false positives, false negatives, and true negatives
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    tn = 0  # True negatives

    # Calculate tp, fp, fn, tn
    for i in range(len(labels)):
        if predictions[i] == 1 and labels[i] == 1:
            tp += 1
        elif predictions[i] == 1 and labels[i] == 0:
            fp += 1
        elif predictions[i] == 0 and labels[i] == 1:
            fn += 1
        elif predictions[i] == 0 and labels[i] == 0:
            tn += 1

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # To handle division by zero
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # To handle division by zero

    # Calculate F1 score
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0  # If both precision and recall are zero, F1 score is 0

    return f1_score

def calculate_accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    # YOUR CODE HERE    
    correct_predictions = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct_predictions += 1
    print(correct_predictions)
    accuracy = correct_predictions/len(labels)
    return accuracy

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

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model_type', '-m', type=str, choices=['majority_baseline', 'decision_tree'], default='decision_tree',
        help='Which model type to train')
    parser.add_argument('--max_depth', '-d', type=int, default=None, 
        help='The maximum depth of the Decision Tree (None for unlimited depth)')
    parser.add_argument('--criterion', '-c', type=str, choices=['entropy', 'gini'], default='entropy',
        help='Splitting criterion for Decision Tree')
    parser.add_argument('--min_samples_split', '-s', type=int, default=2,
        help='Minimum number of samples required to split a node')
    args = parser.parse_args()
    print(args)

    # load data
    data_dict = load_data()
    train_df = data_dict['train']
    test_df = data_dict['test']
    eval_df = data_dict['eval']
    
    # Remove features with more than 99% zeros from the train and test sets
    train_df = remove_zero_features(train_df)
    test_df = remove_zero_features(test_df)
    eval_df = remove_zero_features(eval_df)
    train_x = train_df.drop('label', axis=1)
    train_y = train_df['label'].tolist()

    test_x = test_df.drop('label', axis=1)
    test_y = test_df['label'].tolist()
    eval_x = eval_df.drop('label', axis=1)
    eval_y = eval_df['label'].tolist()

    # initialize the model
    if args.model_type == 'majority_baseline':
        model = MajorityBaseline()
    elif args.model_type == 'decision_tree':
        model = DecisionTree(max_depth=args.max_depth, criterion=args.criterion)
    else:
        raise ValueError(
            '--model_type must be one of "majority_baseline" or "decision_tree". ' +
            f'Received "{args.model_type}". ' +
            '\nRun `python train.py --help` for additional guidance.')

    # train the model
    train(model=model, x=train_x, y=train_y)

    # evaluate model on train and test data
    train_accuracy, predictions = evaluate(model=model, x=train_x, y=train_y)
    print(f'train accuracy: {train_accuracy:.3f}')
    test_accuracy, predictions = evaluate(model=model, x=test_x, y=test_y)
    print(f'test accuracy: {test_accuracy:.3f}')
    eval_accuracy, predictions = evaluate(model=model, x=eval_x, y=eval_y)
    print(f'eval accuracy: {eval_accuracy:.3f}')
    eval_ids_file = 'data/eval.id'  # Path to the eval.ids file
    output_file = 'dtree_submission.csv'  # Path to save the submission file

    # Generate the Kaggle submission file
    create_submission_file(eval_ids_file, predictions, output_file)
    print(f'test accuracy: {test_accuracy:.3f}')

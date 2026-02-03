import argparse
import pandas as pd
from data import load_data
from evaluate import accuracy
from sklearn.feature_selection import VarianceThreshold
from evaluate import calculate_f1_score
from model import init_perceptron, MajorityBaseline, MODEL_OPTIONS

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

# DON'T EDIT ANY OF THE CODE BELOW
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate a model')
    parser.add_argument('--model', '-m', type=str, default='simple', choices=MODEL_OPTIONS, 
        help=f'Which model to run. Must be one of {MODEL_OPTIONS}.')
    parser.add_argument('--lr', type=float, default=1, 
        help='The learning rate hyperparameter eta (same as the initial learning rate). Defaults to 1.')
    parser.add_argument('--mu', type=float, default=0, 
        help='The margin hyperparameter mu. Defaults to 0.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
        help='How many epochs to train for. Defaults to 10.')
    args = parser.parse_args()

    # load data
    print('load data')
    data_dict = load_data()
    train_x = data_dict['train'].drop('label', axis=1).to_numpy()
    train_y = data_dict['train']['label'].to_numpy()
    print(f'  train x shape: {train_x.shape}\n  train y shape: {train_y.shape}')
    test_x = data_dict['test'].drop('label', axis=1).to_numpy()
    test_y = data_dict['test']['label'].to_numpy()
    print(f'  test x shape: {test_x.shape}\n  test y shape: {test_y.shape}')
    eval_x = data_dict['eval'].drop('label', axis=1).to_numpy()
    eval_y = data_dict['eval']['label'].to_numpy()
    print(f'  eval x shape: {eval_x.shape}\n  eval y shape: {eval_y.shape}')

    # load model using helper function init_perceptron() from model.py
    print(f'initialize model')
    if args.model == 'majority_baseline':
        model = MajorityBaseline()
        # train the model
        print(f'train MajorityBaseline')
        model.train(x=train_x, y=train_y)
    
    else:
        model = init_perceptron(
            variant=args.model, 
            num_features=train_x.shape[1], 
            lr=args.lr, 
            mu=args.mu)
        print(f'  model type: {type(model).__name__}\n  hyperparameters: {model.get_hyperparams()}')

        # train the model
        print(f'train model for {args.epochs} epochs')
        model.train(x=train_x, y=train_y, epochs=args.epochs)

    # evaluate model on train and test data
    print('evaluate')
    train_predictions = model.predict(x=train_x)
    train_accuracy = calculate_f1_score(labels=train_y, predictions=train_predictions)
    print(f'  train accuracy: {train_accuracy:.3f}')
    test_predictions = model.predict(x=test_x)
    test_accuracy = calculate_f1_score(labels=test_y, predictions=test_predictions)
    print(f'  test accuracy: {test_accuracy:.3f}')
    val_predictions = model.predict(x=eval_x)
    val_accuracy = calculate_f1_score(labels=eval_y, predictions=val_predictions)
    print(f'  val accuracy: {val_accuracy:.3f}')
    # Generate the Kaggle submission file
    eval_ids_file = 'data/eval.id'  # Path to the eval.ids file
    output_file = 'submission.csv'  # Path to save the submission file
    create_submission_file(eval_ids_file, val_predictions, output_file)

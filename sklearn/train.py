import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from evaluate import calculate_f1_score

def create_submission_file(eval_ids_file: str, predictions: list, output_file: str):
    with open(eval_ids_file, 'r') as f:
        eval_ids = [id.strip() for id in f.readlines()]
    submission = pd.DataFrame({'example_id': eval_ids, 'label': predictions})
    submission.to_csv(output_file, index=False)

if __name__ == '__main__':
    # Load data
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    eval = pd.read_csv('data/eval.anon.csv')
    eval_x, eval_y = eval.drop(columns=['label']), eval['label']

    train_x, train_y = train.drop(columns=['label']), train['label']
    test_x, test_y = test.drop(columns=['label']), test['label']

    # Define parameter grid
    param_grid = {
        'logistic__C': [1.0],
        'logistic__solver': ['liblinear'],
        'logistic__max_iter': [100]
    }

    # Setup pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression())
    ])

    # Cross-validation
    print("Performing cross-validation...")
    cv_model = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
    cv_model.fit(train_x, train_y)

    print("Best hyperparameters from cross-validation:", cv_model.best_params_)

    # Predict
    train_preds = cv_model.predict(train_x)
    test_preds = cv_model.predict(test_x)
    eval_preds = cv_model.predict(eval_x)

    print("Train F1 Score:", calculate_f1_score(train_y, train_preds))
    print("Test F1 Score:", calculate_f1_score(test_y, test_preds))
    print("Eval F1 Score:", calculate_f1_score(eval_y, eval_preds))

    # Save predictions
    create_submission_file("data/eval.id", eval_preds.tolist(), "logreg_sklearn_submission.csv")

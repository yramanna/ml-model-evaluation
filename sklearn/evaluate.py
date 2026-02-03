def accuracy(labels: list, predictions: list) -> float:
    '''
    Calculate the accuracy between ground-truth labels and candidate predictions.
    Should be a float between 0 and 1.

    Args:
        labels (list): the ground-truth labels from the data
        predictions (list): the predicted labels from the model

    Returns:
        float: the accuracy of the predictions, when compared to the ground-truth labels
    '''

    assert len(labels) == len(predictions), (
        f'{len(labels)=} and {len(predictions)=} must be the same length.' + 
        '\n\n  Have you implemented model.predict()?\n'
    )
    
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    accuracy = correct / len(labels)

    return accuracy

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

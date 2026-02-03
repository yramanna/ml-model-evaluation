from typing import Protocol
import numpy as np
from collections import Counter
from utils import clip, shuffle_data

# set the numpy random seed so our randomness is reproducible
np.random.seed(1)

MODEL_OPTIONS = ['majority_baseline', 'svm', 'logistic_regression']

# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the classes further down.
class Model(Protocol):
    def __init__(**hyperparam_kwargs):
        ...

    def get_hyperparams(self) -> dict:
        ...

    def loss(self, ) -> float:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...

class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        self.majority_label = None

    def get_hyperparams(self) -> dict:
        return {}

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        return None
    
    def train(self, x: np.ndarray, y: np.ndarray):
        '''
        Train a baseline model that returns the most common label in the dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example

        Hints:
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
        '''
    
        # YOUR CODE HERE
        counter = Counter(y)
        self.majorityLabel = counter.most_common(1)[0][0]

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        return [self.majorityLabel]*len(x)

class SupportVectorMachine(Model):
    def __init__(self, num_features: int, lr0: float, C: float):
        '''
        Initialize a new SupportVectorMachine model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            C (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.C = C

        # YOUR CODE HERE
        self.w = np.zeros(num_features)
        self.b = 0.0

    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'C': self.C}

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Don't forget to convert the {0, 1} label to {-1, 1}.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        y = 2 * int(y_i) - 1
        margin = y * (np.dot(self.w, x_i) + self.b)
        hinge  = max(0.0, 1.0 - margin)
        reg = 0.5 * np.dot(self.w, self.w)
        return reg + self.C * hinge
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        for epoch in range(epochs):
            eta = self.lr0 / (1.0 + epoch)
            x_shuf, y_shuf = shuffle_data(x, y)

            for x_i, y_i in zip(x_shuf, y_shuf):
                y_m   = 2 * int(y_i) - 1
                score = np.dot(self.w, x_i) + self.b
                margin = y_m * score

                if margin < 1.0:
                    grad_w = self.w - self.C * y_m * x_i
                    grad_b = -self.C * y_m
                else:
                    grad_w = self.w
                    grad_b = 0.0

                self.w -= eta * grad_w
                self.b -= eta * grad_b

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        scores = x.dot(self.w) + self.b
        return [1 if s >= 0.0 else 0 for s in scores]

class LogisticRegression(Model):
    def __init__(self, num_features: int, lr0: float, sigma2: float):
        '''
        Initialize a new LogisticRegression model

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr0 (float): the initial learning rate (gamma_0)
            sigma2 (float): the regularization/loss tradeoff hyperparameter
        '''

        self.lr0 = lr0
        self.sigma2 = sigma2

        # YOUR CODE HERE
        self.w = np.zeros(num_features)

    
    def get_hyperparams(self) -> dict:
        return {'lr0': self.lr0, 'sigma2': self.sigma2}
    

    def loss(self, x_i: np.ndarray, y_i: int) -> float:
        '''
        Calculate the SVM loss on a single example.

        Args:
            x_i (np.ndarray): a 1-D np.ndarray (num_features) with the features for a single example
            y_i (int): the label for the example, either 0 or 1.

        Returns:
            float: the loss for the example using the current weights

        Hints:
            - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
                If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        y = 2 * int(y_i) - 1
        z = clip(-y * np.dot(self.w, x_i), 100)
        return np.log(1.0 + np.exp(z)) + (1.0 / self.sigma2) * np.dot(self.w, self.w)
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Shuffle your data between epochs. You can use `shuffle_data()` from utils.py to help with this.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        for epoch in range(epochs):
            x_shuf, y_shuf = shuffle_data(x, y)
            for x_i, y_i in zip(x_shuf, y_shuf):
                y_m = 2 * int(y_i) - 1
                raw = clip(np.dot(self.w, x_i), 100)
                z = -y_m * raw
                sigma = 1.0 / (1.0 + np.exp(-z))
                grad_loss = -y_m * x_i * sigma
                grad_reg  = (2.0 / self.sigma2) * self.w
                grad = grad_loss + grad_reg
                self.w -= self.lr0 * grad


    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        probs = [sigmoid(np.dot(self.w, x_i)) for x_i in x]
        return [1 if p >= 0.5 else 0 for p in probs]


def sigmoid(z: float) -> float:
    '''
    The sigmoid function.

    Args:
        z (float): the argument to the sigmoid function.

    Returns:
        float: the sigmoid applied to z.

    Hints:
        - Use the `clip()` function from utils.py to clip the input to exp() to be between -100 and 100.
            If you apply exp() to very small/large numbers, you'll likely run into a float overflow issue.
          
    '''    
    # YOUR CODE HERE, REMOVE THE LINE BELOW
    z = clip(z, 100)
    return 1.0 / (1.0 + np.exp(-z))

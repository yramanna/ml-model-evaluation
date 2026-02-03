from typing import Protocol, Tuple
from collections import Counter
import numpy as np

# set the numpy random seed so our randomness is reproducible
np.random.seed(1)

# DON'T CHANGE THE CLASS BELOW! 
# You will implement the train and predict functions in the Perceptron classes further down.
class Model(Protocol):
    def get_hyperparams(self) -> dict:
        ...
        
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        ...

    def predict(self, x: np.ndarray) -> list:
        ...

class MajorityBaseline(Model):
    def __init__(self):
        
        # YOUR CODE HERE, REMOVE THE LINE BELOW
        self.majorityLabel = None


    def get_hyperparams(self) -> dict:
        return {}
    

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


class Perceptron(Model):
    def __init__(self, num_features: int, lr: float, decay_lr: bool = False, mu: float = 0):
        '''
        Initialize a new Perceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate (eta). This is also the initial learning rate if decay_lr=True
            decay_lr (bool): whether or not to decay the initial learning rate lr
            mu (float): the margin (mu) that determines the threshold for a mistake. Defaults to 0
        '''     

        self.lr = lr
        self.decay_lr = decay_lr
        self.mu = mu
        self.updates = 0

        # YOUR CODE HERE
        self.weightVector = np.random.uniform(0, 0, num_features)
        self.bias = np.random.uniform(-0.01, 0.01)

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr, 'decay_lr': self.decay_lr, 'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE

        #Iterate over number of epochs to train the model
        for epoch in range(epochs):
            #Shuffle data
            x, y = shuffle_data(x, y)
            #Decay learning rate if needed
            if self.decay_lr:
                self.lr /= 1 + epoch
            #In each epoch, we make predictions for each example and update weight vector if needed
            for example in range(x.shape[0]):
                if y[example]*(np.dot(self.weightVector, x[example])+self.bias) < self.mu:
                    self.weightVector += self.lr*y[example]*x[example]
                    self.bias += self.lr*y[example]
                    self.updates += 1

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.
        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        predictions = np.dot(x, self.weightVector) + self.bias
        return [1 if prediction > 0 else -1 for prediction in predictions]
    

class AveragedPerceptron(Model):
    def __init__(self, num_features: int, lr: float):
        '''
        Initialize a new AveragedPerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            lr (float): the learning rate eta
        '''     

        self.lr = lr
        self.normal_updates = 0
        self.avg_updates = 0

        # YOUR CODE HERE
        self.weightVector = np.random.uniform(-0.01, 0.01, num_features)
        self.bias = np.random.uniform(-0.01, 0.01)
        self.averageVector = np.zeros(num_features)
        self.averageBias = np.random.uniform(-0.01, 0.01)

    def get_hyperparams(self) -> dict:
        return {'lr': self.lr}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE
        
        #Iterate over number of epochs to train the model
        for _ in range(epochs):
            #Shuffle data
            x, y = shuffle_data(x, y)
            #In each epoch, we make predictions for each example and update weight vector if needed
            for example in range(x.shape[0]):
                if y[example]*(np.dot(self.weightVector, x[example])+self.bias) < 0:
                    self.weightVector += self.lr*y[example]*x[example]
                    self.bias += self.lr*y[example]
                    self.normal_updates += 1
                self.averageVector += self.weightVector
                self.averageBias += self.bias
                self.avg_updates += 1

        print(self.normal_updates, self.avg_updates)

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        predictions = np.dot(x, self.averageVector) + self.averageBias
        return [1 if prediction > 0 else -1 for prediction in predictions]
    

class AggressivePerceptron(Model):
    def __init__(self, num_features: int, mu: float):
        '''
        Initialize a new AggressivePerceptron.

        Args:
            num_features (int): the number of features (i.e. dimensions) the model will have
            mu (float): the hyperparameter mu
        '''     

        self.mu = mu
        self.updates = 0
        # YOUR CODE HERE
        self.weightVector = np.random.uniform(-0.01, 0.01, num_features)
        self.bias = np.random.uniform(-0.01, 0.01)
        self.lr = 0.1

    def get_hyperparams(self) -> dict:
        return {'mu': self.mu}
    
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs: int):
        '''
        Train from examples (x_i, y_i) where 0 < i < num_examples

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features
            y (np.ndarray): a 1-D np.ndarray (num_examples) with the target labels corresponding to each example
            epochs (int): how many epochs to train for

        Hints:
            - Remember to shuffle your data between epochs.
            - If you'd rather use python lists, you can convert an np.ndarray `x` to a list with `x.tolist()`.
            - You can check the shape of an np.ndarray `x` with `print(x.shape)`
            - Take a look at `np.matmul()` for matrix multiplication between two np.ndarray matrices.
        '''

        # YOUR CODE HERE

        #Iterate over number of epochs to train the model
        for epoch in range(epochs):
            #Shuffle data
            x, y = shuffle_data(x, y)
            #In each epoch, we make predictions for each example and update weight vector if needed
            for example in range(x.shape[0]):
                prediction = y[example] * (np.dot(self.weightVector, x[example]) + self.bias)
                self.lr = (self.mu - prediction) / (np.dot(x[example], x[example]) + 1)
                if prediction <= self.mu:
                    self.weightVector += self.lr * y[example] * x[example]
                    self.bias += self.lr * y[example]
                    self.updates += 1
        print(self.updates)

    def predict(self, x: np.ndarray) -> list:
        '''
        Predict the labels for a dataset.

        Args:
            x (np.ndarray): a 2-D np.ndarray (num_examples x num_features) with examples and their features

        Returns:
            list: A list with the predicted labels, each corresponding to a row in `x`.
        '''

        # YOUR CODE HERE, REMOVE THE LINE BELOW
        predictions = np.dot(x, self.weightVector) + self.bias
        return [1 if prediction > 0 else -1 for prediction in predictions]


# DON'T MODIFY THE FUNCTIONS BELOW!
PERCEPTRON_VARIANTS = ['simple', 'decay', 'margin', 'averaged', 'aggressive']
MODEL_OPTIONS = ['majority_baseline'] + PERCEPTRON_VARIANTS
def init_perceptron(variant: str, num_features: int, lr: float, mu: float) -> Model:
    '''
    This is a helper function to help you initialize the correct variant of the Perceptron

    Args:
        variant (str): which variant of the perceptron to use. See PERCEPTRON_VARIANTS above for options
        num_features (int): the number of features (i.e. dimensions) the model will have
        lr (float): the learning rate hyperparameter eta. Same as initial learning rate for decay setting
        mu (float): the margin hyperparamter mu. Ignored for variants "simple", "decay", and "averaged"

    Returns
        (Model): the initialized perceptron model
    '''
    
    assert variant in PERCEPTRON_VARIANTS, f'{variant=} must be one of {PERCEPTRON_VARIANTS}'

    if variant == 'simple':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=False)
    elif variant == 'decay':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True)
    elif variant == 'margin':
        return Perceptron(num_features=num_features, lr=lr, decay_lr=True, mu=mu)
    elif variant == 'averaged':
        return AveragedPerceptron(num_features=num_features, lr=lr)
    elif variant == 'aggressive':
        return AggressivePerceptron(num_features=num_features, mu=mu)


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''
    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]

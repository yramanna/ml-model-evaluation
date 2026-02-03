''' This file provides utility functions for loading data that you may find useful.
    You don't need to change this file.
'''

from glob import glob

import pandas as pd


DATASET_OPTIONS = ['100', '1000', '5000']


def load_data(dataset_name: str = '100') -> dict:
    '''
    Loads the data for one of the three datasets provided in this assignment.

    Returns:
        dict: a dictionary containing the train, test, and cv data as (x, y) tuples of np.ndarray matrices
    '''

    dataset_path = f'data/data-{dataset_name}'

    # load train dataset
    train = pd.read_csv(f'data/train.csv')

    # load test dataset
    test = pd.read_csv(f'data/test.csv')

    # load cross validation datasets
    eval = pd.read_csv(f'data/eval.anon.csv')

    return {
        'train': train,
        'test': test,
        'eval': eval}

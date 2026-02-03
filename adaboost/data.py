
from glob import glob
import pandas as pd

def load_data() -> dict:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    eval = pd.read_csv('data/eval.anon.csv')
    cv_folds = [pd.read_csv(path) for path in glob('data/cv/*')]
    return {'train': train, 'test': test, 'eval': eval, 'cv_folds': cv_folds}

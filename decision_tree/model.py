from math import log2
from typing import Protocol
from collections import Counter
import pandas as pd

class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...
    def predict(self, x: pd.DataFrame) -> list:
        ...

MODEL_OPTIONS = ['majority_baseline', 'decision_tree']

class MajorityBaseline(Model):
    def __init__(self):
        self.majorityLabel = None
    def train(self, x: pd.DataFrame, y: list):
        counter = Counter(y)
        self.majorityLabel = counter.most_common(1)[0][0]
    def predict(self, x: pd.DataFrame) -> list:
        return [self.majorityLabel] * len(x)

class DecisionTree:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2, criterion: str = 'entropy', max_categories: int = 20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_categories = max_categories  # new param
        self.tree = None

    def calculate_impurity(self, y: list) -> float:
        counter = Counter(y)
        total = len(y)
        if self.criterion == 'entropy':
            return -sum((count/total) * log2(count/total) for count in counter.values() if count > 0)
        elif self.criterion == 'gini':
            return 1 - sum((count/total) ** 2 for count in counter.values())

    def information_gain(self, x_column: list, y: list, thresholds=None):
        y_series = pd.Series(y)
        x_series = pd.Series(x_column, index=y_series.index)
        impurity_before = self.calculate_impurity(y_series)

        best_threshold, best_gain = None, -1

        for threshold in thresholds:
            left_y = y_series[x_series <= threshold]
            right_y = y_series[x_series > threshold]
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            p = len(left_y) / len(y_series)
            gain = impurity_before - (p * self.calculate_impurity(left_y) + (1 - p) * self.calculate_impurity(right_y))

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold

        return best_threshold, best_gain

    def best_split(self, x: pd.DataFrame, y: list):
        best_feature = None
        best_value = None
        best_gain = -1

        for feature in x.columns:
            column = x[feature]
            values = column.unique()

            if column.dtype.kind in 'iuf':  # numeric
                thresholds = sorted(set((a + b) / 2 for a, b in zip(sorted(values)[:-1], sorted(values)[1:])))
            else:  # categorical
                if len(values) > self.max_categories:
                    continue  # avoid high-cardinality
                thresholds = values

            threshold, gain = self.information_gain(column, y, thresholds)
            if gain is None or gain <= 0:
                continue

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_value = threshold

        return best_feature, best_value

    def build_tree(self, x, y, depth):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]

        feature, value = self.best_split(x, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        column = x[feature]
        if column.dtype.kind in 'iuf':
            left_idx = column <= value
            right_idx = column > value
        else:
            left_idx = column == value
            right_idx = column != value

        y_series = pd.Series(y, index=x.index) if not isinstance(y, pd.Series) else y
        left_tree = self.build_tree(x[left_idx], y_series[left_idx], depth + 1)
        right_tree = self.build_tree(x[right_idx], y_series[right_idx], depth + 1)

        return (feature, value, left_tree, right_tree)

    def train(self, x: pd.DataFrame, y: list):
        self.tree = self.build_tree(x, y, 0)

    def predict_instance(self, row, node):
        if not isinstance(node, tuple):
            return node
        feature, value, left, right = node
        if isinstance(row[feature], (int, float)):
            return self.predict_instance(row, left if row[feature] <= value else right)
        else:
            return self.predict_instance(row, left if row[feature] == value else right)

    def predict(self, x: pd.DataFrame) -> list:
        return [self.predict_instance(row._asdict(), self.tree) for row in x.itertuples(index=False)]
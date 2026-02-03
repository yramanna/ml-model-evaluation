from math import log2
from typing import Protocol
from collections import Counter
import pandas as pd
import numpy as np

class Model(Protocol):
    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        ...
    def predict(self, x: pd.DataFrame) -> list:
        ...

MODEL_OPTIONS = ['majority_baseline', 'decision_tree', 'adaboost']

class MajorityBaseline(Model):
    def __init__(self):
        self.majorityLabel = None
    def train(self, x: pd.DataFrame, y: list):
        counter = Counter(y)
        self.majorityLabel = counter.most_common(1)[0][0]
    def predict(self, x: pd.DataFrame) -> list:
        return [self.majorityLabel] * len(x)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy', max_categories=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_categories = max_categories
        self.tree = None

    def calculate_impurity(self, y):
        counter = Counter(y)
        total = len(y)
        if self.criterion == 'entropy':
            return -sum((count/total) * log2(count/total) for count in counter.values() if count > 0)
        elif self.criterion == 'gini':
            return 1 - sum((count/total) ** 2 for count in counter.values())

    def information_gain(self, x_column, y, thresholds=None):
        x_series = pd.Series(x_column.values if isinstance(x_column, pd.Series) else x_column).reset_index(drop=True)
        y_series = pd.Series(y).reset_index(drop=True)
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

    def best_split(self, x, y):
        best_feature, best_value, best_gain = None, None, -1
        for feature in x.columns:
            column = x[feature]
            values = column.unique()
            if column.dtype.kind in 'iuf':
                sorted_vals = sorted(set(values))
                if len(sorted_vals) > 10:
                    idxs = np.linspace(0, len(sorted_vals) - 2, 10).astype(int)
                    thresholds = [(sorted_vals[i] + sorted_vals[i+1]) / 2 for i in idxs]
                else:
                    thresholds = [(a + b) / 2 for a, b in zip(sorted_vals[:-1], sorted_vals[1:])]
            else:
                if len(values) > self.max_categories:
                    continue
                thresholds = values

            threshold, gain = self.information_gain(column, y, thresholds)
            if gain is None or gain <= 0:
                continue
            if gain > best_gain:
                best_gain = gain
                best_feature, best_value = feature, threshold
        return best_feature, best_value

    def build_tree(self, x, y, depth):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        feature, value = self.best_split(x, y)
        if feature is None:
            return Counter(y).most_common(1)[0][0]

        column = x[feature]
        left_idx = column <= value if column.dtype.kind in 'iuf' else column == value
        right_idx = ~left_idx

        y_series = pd.Series(y, index=x.index) if not isinstance(y, pd.Series) else y
        left_tree = self.build_tree(x[left_idx], y_series[left_idx], depth + 1)
        right_tree = self.build_tree(x[right_idx], y_series[right_idx], depth + 1)
        return (feature, value, left_tree, right_tree)

    def train(self, x, y):
        self.tree = self.build_tree(x, y, 0)

    def predict_instance(self, row, node):
        if not isinstance(node, tuple):
            return node
        feature, value, left, right = node
        if isinstance(row[feature], (int, float)):
            return self.predict_instance(row, left if row[feature] <= value else right)
        else:
            return self.predict_instance(row, left if row[feature] == value else right)

    def predict(self, x):
        return [self.predict_instance(row._asdict(), self.tree) for row in x.itertuples(index=False)]

class AdaBoostClassifier:
    def __init__(self, base_learner_cls, n_estimators=10, **kwargs):  # Default reduced to 10
        self.n_estimators = n_estimators
        self.base_learner_cls = base_learner_cls
        self.base_learner_kwargs = kwargs
        self.learners = []
        self.alphas = []

    def train(self, x, y):
        n_samples = len(y)
        weights = np.ones(n_samples) / n_samples
        y_binary = np.array([1 if label == 1 else -1 for label in y])

        for _ in range(self.n_estimators):
            learner = self.base_learner_cls(**self.base_learner_kwargs)
            sampled_indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
            x_sample = x.iloc[sampled_indices]
            y_sample = y_binary[sampled_indices]

            learner.train(x_sample, y_sample)
            predictions = learner.predict(x_sample)
            predictions = np.array([1 if pred == 1 else -1 for pred in predictions])

            err = np.sum(weights[sampled_indices] * (predictions != y_sample)) / np.sum(weights[sampled_indices])
            if err > 0.5:
                continue

            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            weights[sampled_indices] *= np.exp(-alpha * y_sample * predictions)
            weights /= np.sum(weights)

            self.learners.append(learner)
            self.alphas.append(alpha)

    def predict(self, x):
        final_scores = np.zeros(x.shape[0])
        for alpha, learner in zip(self.alphas, self.learners):
            preds = learner.predict(x)
            preds = np.array([1 if p == 1 else -1 for p in preds])
            final_scores += alpha * preds
        return [1 if score > 0 else 0 for score in final_scores]

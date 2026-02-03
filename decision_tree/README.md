# Decision Tree

## Overview

Decision Trees are non-parametric models that recursively partition the feature space to maximize class purity. This implementation focuses on understanding the bias–variance trade-off introduced by tree depth and splitting criteria.

---

## Model Design

- **Splitting criteria**: Gini impurity and entropy
- **Stopping conditions**: Maximum depth and minimum samples per split
- **Tree growth**: Greedy, top-down induction

---

## Hyperparameters

- `max_depth`
- `min_samples_split`
- `criterion`

Tree depth was carefully controlled to prevent overfitting while maintaining expressiveness.

---

## Reproducibility

### Training
```bash
python train.py -m decision_tree -d 10 -s 5 -c gini
```

### Cross Validation
```bash
python cross_validation.py -d 13 14 15 -s 2 5 -c entropy gini
```

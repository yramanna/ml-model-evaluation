# AdaBoost

## Overview

AdaBoost (Adaptive Boosting) is an ensemble learning method that constructs a strong classifier by iteratively combining weak learners. In this project, AdaBoost is implemented using shallow decision trees (decision stumps) as base learners.

The objective is to study how boosting improves classification performance by reweighting misclassified examples across iterations.

---

## Model Design

- **Base learner**: Decision Tree (depth-limited)
- **Loss reweighting**: Exponential loss
- **Ensemble aggregation**: Weighted majority vote

Decision stumps were selected to maintain interpretability and to emphasize the boosting mechanism rather than base learner complexity.

---

## Hyperparameters

The following hyperparameters were explored via cross-validation:

- `max_depth`: Depth of base decision trees
- `min_samples_split`: Minimum samples required to split a node
- `criterion`: Splitting criterion (`entropy`, `gini`)

---

## Reproducibility

### Training with Fixed Hyperparameters
```bash
python train.py --model_type adaboost --max_depth 1 --min_samples_split 2 --criterion entropy
```
### Cross Validation
```bash
python cross_validation.py -d 1 2 -s 2 3 -c entropy gini
```

---

## Notes

AdaBoost demonstrated strong performance when weak learners were sufficiently constrained. Increasing base learner complexity led to diminishing returns and higher variance.

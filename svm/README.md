# Support Vector Machine

## Overview

Support Vector Machines (SVMs) aim to maximize the margin between classes while penalizing misclassification. This implementation uses a primal SGD-based optimization approach.

---

## Model Design

- **Objective**: Hinge loss with regularization
- **Optimization**: Stochastic Gradient Descent
- **Kernel**: Linear

---

## Hyperparameters

- `learning_rate (lr0)`
- `regularization_tradeoff`
- `epochs`
- `feature_dimension`

---

## Reproducibility

### Training
```bash
python train.py -m svm -d 100 --lr0 0.0001 --reg_tradeoff 0.001 --epochs 13
```

### Cross Validation
```bash
python cross_validation.py -m svm -d 100 \
  --lr0_values 1 0.1 0.01 0.001 0.0001 \
  --reg_tradeoff_values 1 0.1 0.01 0.001 \
  --epochs 10
```

## Notes

SVM performance was highly dependent on the balance between margin maximization and regularization strength, highlighting sensitivity to hyperparameter scaling.

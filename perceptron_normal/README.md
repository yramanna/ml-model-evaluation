# Perceptron

## Overview

The Perceptron is a linear classifier that updates weights based on misclassified examples. This project uses a margin-based Perceptron variant to improve convergence stability and robustness.

---

## Model Design

- **Update rule**: Margin Perceptron
- **Optimization**: Online stochastic updates
- **Feature space**: Raw features without transformations

---

## Hyperparameters

- `learning_rate (lr)`
- `margin (mu)`
- `epochs`

---

## Reproducibility

### Training
```bash
python train.py -m margin --lr 1.0 --mu 10.0 -e 10
```

### Cross Validation
```bash
python cross_validation.py -m margin \
  --lr_values 1 0.1 0.01 \
  --mu_values 10 1 0.1 0.01 \
  -e 10
```

## Notes

The Perceptron served as a baseline linear classifier. Performance was highly sensitive to margin selection, reinforcing the importance of regularization even in simple models.

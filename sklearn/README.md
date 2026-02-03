# Logistic Regression (scikit-learn Baseline)

## Overview

A scikit-learn implementation of Logistic Regression was used as a baseline to contextualize the performance of from-scratch models.

---

## Purpose

This model serves as a reference point rather than a primary contribution, allowing comparison between hand-implemented algorithms and optimized library implementations.

---

## Reproducibility

```bash
python train.py -m logistic_regression -d 100 --lr0 0.0001 --reg_tradeoff 1000 --epochs 12
```
## Cross Validation

```bash
python cross_validation.py -m logistic_regression -d 100 \
  --lr0_values 1 0.1 0.01 0.001 0.0001 \
  --reg_tradeoff_values 1 10 100 1000 \
  --epochs 10
```

## Notes

Library implementations consistently converged faster and served as a useful benchmark for evaluating correctness and performance gaps.

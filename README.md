# Comparative Analysis of Machine Learning Models

This repository contains the final project report for a comparative study of supervised machine learning models. The project focuses on understanding model behavior, performance trade-offs, and the effects of feature engineering and hyperparameter tuning by implementing all algorithms **from scratch** in Python.

The full technical report is written in LaTeX and is provided as `main.tex`.

---

## Overview

This project investigates the performance of several supervised learning models on a real-world dataset by applying them under consistent experimental conditions. Rather than relying on high-level machine learning libraries, all models were implemented manually to gain deeper insight into their internal mechanics and optimization behavior.

The study emphasizes:
- How different models respond to the same dataset
- The impact of feature preprocessing decisions
- Sensitivity to hyperparameter choices
- Generalization versus overfitting behavior

---

## Models Implemented

The following models were implemented and evaluated:

- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Perceptron**
- **Decision Tree**
- **AdaBoost**

Each model was trained and tested using multiple feature configurations and tuning strategies to ensure fair comparison.

---

## Feature Engineering

Several feature engineering strategies were explored to understand their effect on model performance:

- Removal of features containing excessive zero values  
- Refinement of zero-value filtering to avoid removing rare but informative features  
- Variance-based feature elimination  
- Empirical evaluation of preprocessing thresholds  

These steps were carefully adjusted to balance dimensionality reduction with information retention.

---

## Hyperparameter Tuning

Hyperparameters were selected using cross-validation and empirical evaluation. The tuning process included:

- Learning rates and regularization strength for linear models  
- Margin parameters for SVM  
- Tree depth and splitting criteria for Decision Trees  
- Number of weak learners and weighting behavior for AdaBoost  

The project emphasizes understanding *why* certain hyperparameters work better rather than treating tuning as a black-box optimization process.

---

## Evaluation Methodology

Models were evaluated using consistent training and testing splits. Performance comparisons focused on:

- Classification accuracy
- Stability across folds
- Sensitivity to feature selection
- Signs of underfitting or overfitting

Results were analyzed qualitatively and quantitatively in the report.


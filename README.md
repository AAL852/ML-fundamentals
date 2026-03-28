# ml-fundamentals

A practical study of core machine learning techniques across three problem domains — regression, multi-class classification, and deep learning — benchmarking classical and neural approaches on real-world datasets.

---

## 📋 Overview

This project covers three independent ML problems, each exploring a different area of the machine learning pipeline: model selection, regularisation, hyperparameter tuning, and deep network training. All models are evaluated rigorously on held-out test sets with appropriate metrics and visualisations.

---

## 📁 Project Structure

```
ml-fundamentals/
├── q1_regression.py        # Linear, Ridge, and LASSO regression on crime data
├── q2_classification.py    # KNN, SVM, and Random Forest on land use data
├── q3_deep_networks.py     # DCNN (with/without augmentation) vs SVM on SVHN
├── outputs/                # Saved figures and plots (auto-created)
└── Data/                   # Dataset directory (not included — see below)
    ├── Q1/                 # communities_train/val/test.csv
    ├── Q2/                 # training/validation/testing.csv
    └── Q3/                 # q3_train.mat, q3_test.mat
```

---

## 🔄 Problems

**Regression: Socio-economic Crime Prediction**

Predicts violent crime rates per capita from 1990 US Census socio-economic features. Three regression models are trained and compared: standard Linear Regression, Ridge, and LASSO. Regularisation strength λ is selected for Ridge and LASSO via validation MSE across 100 log-spaced values. Models are evaluated on test MSE, R², and residual distributions. LASSO's coefficient sparsity is also reported as a measure of feature selection behaviour.

**Classification: Land Use from Aerial Spectral Data**

Classifies land type (Sugi forest, Hinoki forest, Mixed deciduous, Other) from 27 spectral reflectance features captured by aerial sensors. Three multi-class classifiers are trained — KNN, SVM, and Random Forest — with hyperparameters selected via grid search on a predefined validation split. Models are evaluated on macro F1 score and confusion matrices on the test set.

**Deep Networks: SVHN Digit Recognition**

Trains a custom DCNN on a limited subset of the Street View House Numbers (SVHN) dataset (1,000 training samples, 10,000 test samples). Two versions are compared — with and without data augmentation — and both are benchmarked against an SVM baseline. Evaluation considers accuracy, macro F1, training time, and inference time.

---

## 📊 Key Results

### Q1 — Regression

| Model | Test MSE | R² | Best λ |
|-------|----------|----|--------|
| Linear Regression | 0.87196 | — | N/A |
| Ridge Regression | 0.02119 | — | 1.0 |
| LASSO Regression | 0.01690 | — | 0.0035 |

LASSO achieves the best test performance by driving irrelevant coefficients to zero, effectively performing implicit feature selection on the high-dimensional socio-economic data.

### Q2 — Classification

| Model | F1 Score (macro) | Best Parameters |
|-------|-----------------|-----------------|
| KNN | 0.866 | k=5, metric=manhattan, weights=uniform |
| SVM | 0.915 | C=100, kernel=poly, degree=3 |
| Random Forest | 1.000 | max_depth=10, n_estimators=200 |

### Q3 — Deep Networks

| Model | F1 Score | Training Time (s) |
|-------|----------|--------------------|
| DCNN | 0.2337 | 10.26 |
| DCNN + Augmentation | 0.2929 | 12.76 |
| SVM | 0.3463 | 1.63 |

With only 1,000 training samples, the SVM outperforms both DCNNs — data augmentation narrows the gap but does not close it. This highlights the data efficiency advantage of kernel methods in low-data regimes.

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow scipy
```

---

## 🗄️ Data

Datasets are not included in this repository. Place them in the `Data/` directory as follows:

- **Q1**: [UCI Communities and Crime dataset](https://archive.ics.uci.edu/ml/datasets/communities+and+crime) — `communities_train.csv`, `communities_val.csv`, `communities_test.csv`
- **Q2**: Aerial spectral land use dataset — `training.csv`, `validation.csv`, `testing.csv`
- **Q3**: [SVHN dataset](http://ufldl.stanford.edu/housenumbers/) (abridged subset) — `q3_train.mat`, `q3_test.mat`

Add the following to `.gitignore` to keep data out of version control:

```
Data/
outputs/
*.mat
```

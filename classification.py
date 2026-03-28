"""
CAB420 Assignment 1A — Question 2: Classification
==================================================
Land use classification from aerial spectral reflectance data
using KNN, Random Forest, and an SVM ensemble.

Dataset: Q2/training.csv, Q2/validation.csv, Q2/testing.csv
Classes: s (Sugi forest), h (Hinoki forest),
         d (Mixed deciduous forest), o (Other non-forest)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
)

# ── 1. Data Loading ───────────────────────────────────────────────────────────
train = pd.read_csv('../Data/Q2/training.csv')
val   = pd.read_csv('../Data/Q2/validation.csv')
test  = pd.read_csv('../Data/Q2/testing.csv')

X_train, Y_train = train.iloc[:, 1:].to_numpy(), train.iloc[:, 0].to_numpy()
X_val,   Y_val   = val.iloc[:, 1:].to_numpy(),   val.iloc[:, 0].to_numpy()
X_test,  Y_test  = test.iloc[:, 1:].to_numpy(),  test.iloc[:, 0].to_numpy()

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Classes: {np.unique(Y_train)}")


# ── 2. Pre-processing ─────────────────────────────────────────────────────────
# Standardise features — important for distance-based models (KNN, SVM)
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# ── 3. Grid Search Setup (PredefinedSplit) ────────────────────────────────────
# Combine train and val for GridSearchCV with a fixed validation fold
X_train_val = np.vstack((X_train_scaled, X_val_scaled))
Y_train_val = np.hstack((Y_train, Y_val))

# mask: -1 = training fold, 0 = validation fold
split_mask = np.zeros(len(Y_train_val))
split_mask[:len(Y_train)] = -1
predefined_split = PredefinedSplit(split_mask)


# ── 4. K-Nearest Neighbours ───────────────────────────────────────────────────
print("\n[KNN] Running grid search...")
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'metric':      ['euclidean', 'manhattan', 'minkowski'],
    'weights':     ['uniform', 'distance'],
}

knn_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_param_grid,
    cv=predefined_split,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1,
)
knn_search.fit(X_train_val, Y_train_val)

best_knn = knn_search.best_estimator_
print(f"Best KNN parameters: {knn_search.best_params_}")


# ── 5. Support Vector Machine ─────────────────────────────────────────────────
print("\n[SVM] Running grid search...")
svm_param_grid = {
    'C':      [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4],
    'gamma':  ['scale', 'auto'],
}

svm_search = GridSearchCV(
    estimator=SVC(),
    param_grid=svm_param_grid,
    cv=predefined_split,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1,
)
svm_search.fit(X_train_val, Y_train_val)

best_svm = svm_search.best_estimator_
print(f"Best SVM parameters: {svm_search.best_params_}")


# ── 6. Random Forest ─────────────────────────────────────────────────────────
print("\n[Random Forest] Running grid search...")
rf_param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth':    [10, 20, 30, None],
}

rf_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=predefined_split,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1,
)
rf_search.fit(X_train_val, Y_train_val)

best_rf = rf_search.best_estimator_
print(f"Best Random Forest parameters: {rf_search.best_params_}")


# ── 7. Evaluation on Test Set ─────────────────────────────────────────────────
models = {
    'KNN':           best_knn,
    'SVM':           best_svm,
    'Random Forest': best_rf,
}

f1_scores = {}
print("\n" + "="*60)
for name, model in models.items():
    Y_pred = model.predict(X_test_scaled)
    f1     = f1_score(Y_test, Y_pred, average='macro')
    f1_scores[name] = f1
    print(f"\n{name} — F1 (macro): {f1:.5f}")
    print(classification_report(Y_test, Y_pred))

# Summary table
results = pd.DataFrame({
    'Model':    list(f1_scores.keys()),
    'F1 Score': list(f1_scores.values()),
    'Best Params': [
        str(knn_search.best_params_),
        str(svm_search.best_params_),
        str(rf_search.best_params_),
    ],
})
print("\nTable 1: Model Comparison")
print(results.to_string(index=False))


# ── 8. Confusion Matrix Visualisation ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
class_labels = sorted(np.unique(Y_test))

for ax, (name, model) in zip(axes, models.items()):
    Y_pred = model.predict(X_test_scaled)
    cm     = confusion_matrix(Y_test, Y_pred, labels=class_labels)
    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        xticklabels=class_labels, yticklabels=class_labels,
        cmap='Blues', cbar=False,
    )
    ax.set_title(f'{name}\nF1: {f1_scores[name]:.4f}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

fig.suptitle('Fig 1: Confusion Matrices — KNN, SVM, and Random Forest on Test Set')
plt.tight_layout()
plt.savefig('outputs/q2_confusion_matrices.png', dpi=150)
plt.show()

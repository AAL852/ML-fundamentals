"""
CAB420 Assignment 1A — Question 1: Regression
==============================================
Predicting violent crime rates from socio-economic features
using Linear, Ridge, and LASSO regression.

Dataset: 1990 US Census Communities (communities_train/val/test.csv)
Target:  ViolentCrimesPerPop
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. Data Loading ───────────────────────────────────────────────────────────
train = pd.read_csv('../Data/Q1/communities_train.csv')
val   = pd.read_csv('../Data/Q1/communities_val.csv')
test  = pd.read_csv('../Data/Q1/communities_test.csv')

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_val,   y_val   = val.iloc[:, :-1],   val.iloc[:, -1]
X_test,  y_test  = test.iloc[:, :-1],  test.iloc[:, -1]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"Missing values — Train: {X_train.isna().sum().sum()}, "
      f"Val: {X_val.isna().sum().sum()}, Test: {X_test.isna().sum().sum()}")


# ── 2. Pre-processing ─────────────────────────────────────────────────────────
# Impute any missing values with the column mean (none found, but applied for robustness)
X_train = X_train.fillna(X_train.mean())
X_val   = X_val.fillna(X_train.mean())   # use training mean to avoid leakage
X_test  = X_test.fillna(X_train.mean())

# Standardise features: fit scaler on training data only
scaler        = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# No categorical variables present — one-hot encoding not required
# Target variable is left unscaled as regression models do not require it


# ── 3. Linear Regression ─────────────────────────────────────────────────────
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr    = mean_squared_error(y_test, y_pred_lr)
r2_lr     = r2_score(y_test, y_pred_lr)

print(f"\nLinear Regression — Test MSE: {mse_lr:.5f} | R²: {r2_lr:.5f}")


# ── 4. Ridge & LASSO — λ Selection via Validation MSE ────────────────────────
alphas     = np.logspace(-3, 0, 100)  # 100 log-spaced values: 10⁻³ to 1
ridge_mse  = []
lasso_mse  = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_mse.append(mean_squared_error(y_val, ridge.predict(X_val_scaled)))

    # LASSO
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_mse.append(mean_squared_error(y_val, lasso.predict(X_val_scaled)))

best_alpha_ridge = alphas[np.argmin(ridge_mse)]
best_alpha_lasso = alphas[np.argmin(lasso_mse)]

print(f"\nOptimal λ — Ridge: {best_alpha_ridge:.5f} | LASSO: {best_alpha_lasso:.5f}")

# Plot MSE vs λ
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(alphas, ridge_mse, label='Ridge')
ax.plot(alphas, lasso_mse, label='LASSO')
ax.axvline(best_alpha_ridge, color='steelblue', linestyle='--', alpha=0.6)
ax.axvline(best_alpha_lasso, color='darkorange', linestyle='--', alpha=0.6)
ax.set_xscale('log')
ax.set_xlabel('Regularisation Strength (λ)')
ax.set_ylabel('Validation MSE')
ax.set_title('Fig 1: Validation MSE vs. λ for Ridge and LASSO Regression')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/q1_mse_vs_lambda.png', dpi=150)
plt.show()


# ── 5. Final Regularised Models ───────────────────────────────────────────────
final_ridge = Ridge(alpha=best_alpha_ridge)
final_ridge.fit(X_train_scaled, y_train)

final_lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000)
final_lasso.fit(X_train_scaled, y_train)

y_pred_ridge = final_ridge.predict(X_test_scaled)
y_pred_lasso = final_lasso.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_ridge  = r2_score(y_test, y_pred_ridge)
r2_lasso  = r2_score(y_test, y_pred_lasso)

print(f"Ridge Regression  — Test MSE: {mse_ridge:.5f} | R²: {r2_ridge:.5f}")
print(f"LASSO Regression  — Test MSE: {mse_lasso:.5f} | R²: {r2_lasso:.5f}")


# ── 6. Results Summary ────────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model':    ['Linear Regression', 'Ridge Regression', 'LASSO Regression'],
    'Test MSE': [mse_lr, mse_ridge, mse_lasso],
    'R²':       [r2_lr, r2_ridge, r2_lasso],
    'Best λ':   ['N/A', f'{best_alpha_ridge:.5f}', f'{best_alpha_lasso:.5f}'],
})
print("\nTable 1: Model Comparison")
print(results.to_string(index=False))


# ── 7. Residual Analysis ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models_info = [
    ('Linear Regression', y_pred_lr),
    ('Ridge Regression',  y_pred_ridge),
    ('LASSO Regression',  y_pred_lasso),
]

for ax, (name, y_pred) in zip(axes, models_info):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.4, s=15)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(name)

fig.suptitle('Fig 2: Residual Plots for the Three Regression Models')
plt.tight_layout()
plt.savefig('outputs/q1_residual_plots.png', dpi=150)
plt.show()


# ── 8. LASSO Feature Selection ────────────────────────────────────────────────
# LASSO drives irrelevant coefficients to zero — useful for feature selection
n_nonzero = np.sum(final_lasso.coef_ != 0)
n_total   = len(final_lasso.coef_)
print(f"\nLASSO active features: {n_nonzero} / {n_total}")

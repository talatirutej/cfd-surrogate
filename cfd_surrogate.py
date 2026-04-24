# cfd_surrogate_v1.py
# ============================================================
# CFD Surrogate Model — Mahindra Research Valley
# Predicts drag coefficient (Cd) from aerodynamic parameters
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1 — Load data
# ============================================================
# Your CSV should have columns matching your Excel sheet:
# yaw, speed, rear_angle, spoiler_angle, ground_clearance,
# frontal_area, actual_cd
# ============================================================

df = pd.read_csv('cfd_data.csv')
print("Dataset shape:", df.shape)
print(df.head())

# Define features and target
FEATURES = ['yaw', 'speed', 'rear_angle', 'spoiler_angle',
            'ground_clearance', 'frontal_area']
TARGET = 'actual_cd'

X = df[FEATURES].values
y = df[TARGET].values

# ============================================================
# STEP 2 — Scale features
# Engineering reason: GP and some models are sensitive to scale.
# yaw might range 0-15, speed 20-200 — unscaled, speed dominates.
# StandardScaler makes every input zero-mean and unit-variance.
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# STEP 3 — K-Fold cross-validation
# With small data, never do a fixed train/test split.
# K-Fold uses all data for testing (just at different times).
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# STEP 4 — Define models
# ============================================================

# Model A: Random Forest
rf = RandomForestRegressor(
    n_estimators=200,      # 200 trees in the forest
    max_depth=None,        # let trees grow fully (regularize via n_estimators)
    min_samples_leaf=2,    # IMPORTANT for small data: no leaf with <2 samples
    random_state=42
)

# Model B: Gradient Boosting (often better than RF on small tabular data)
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,    # slow learning rate = better generalization
    max_depth=3,           # shallow trees prevent overfitting
    random_state=42
)

# Model C: Gaussian Process (BEST for small data — gives uncertainty too)
kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5) 
# Matern(nu=2.5) is the standard choice for smooth physical functions
# It assumes the function is twice differentiable — appropriate for aerodynamics
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,           # numerical stability
    normalize_y=True,     # zero-mean the output — helps GP training
    n_restarts_optimizer=10  # try 10 random starts to find the best kernel params
)

models = {
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Gaussian Process': gp
}

# ============================================================
# STEP 5 — Cross-validation comparison
# ============================================================
print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*50)

cv_results = {}
for name, model in models.items():
    # Use scaled data for GP, raw for tree models
    X_input = X_scaled if name == 'Gaussian Process' else X
    scores = cross_val_score(model, X_input, y,
                             cv=kf, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    cv_results[name] = rmse_scores
    print(f"{name:25s}: RMSE = {rmse_scores.mean():.5f} ± {rmse_scores.std():.5f}")

# ============================================================
# STEP 6 — Retrain best model on full data and evaluate
# ============================================================
# For now train all three and show full metrics

print("\n" + "="*50)
print("FULL DATASET METRICS (after retraining on all data)")
print("These will look too good — use CV results as the real metric!")
print("="*50)

for name, model in models.items():
    X_input = X_scaled if name == 'Gaussian Process' else X
    model.fit(X_input, y)
    y_pred = model.predict(X_input)

    mae  = mean_absolute_error(y, y_pred)
    mse  = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y, y_pred)

    print(f"\n{name}")
    print(f"  MAE  = {mae:.5f}")
    print(f"  MSE  = {mse:.6f}")
    print(f"  RMSE = {rmse:.5f}")
    print(f"  R²   = {r2:.4f}")

# ============================================================
# STEP 7 — Feature importance (Random Forest)
# Engineering reason: tells you WHICH design parameter most
# affects Cd. Useful to know before running more simulations.
# ============================================================
rf.fit(X, y)
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*50)
print(feat_df.to_string(index=False))

# ============================================================
# STEP 8 — Plots
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('CFD Surrogate Model — Mahindra Research Valley', fontsize=13)

# Plot 1: Actual vs Predicted for each model
colors = ['steelblue', 'coral', 'green']
for i, (name, model) in enumerate(models.items()):
    X_input = X_scaled if name == 'Gaussian Process' else X
    y_pred = model.predict(X_input)
    ax = axes[i]
    ax.scatter(y, y_pred, alpha=0.7, color=colors[i], edgecolors='black', linewidth=0.5)
    lims = [min(y.min(), y_pred.min()) - 0.005,
            max(y.max(), y_pred.max()) + 0.005]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6, label='Perfect fit')
    ax.set_xlabel('Actual Cd')
    ax.set_ylabel('Predicted Cd')
    ax.set_title(name)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.5f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Feature importance bar chart
fig2, ax2 = plt.subplots(figsize=(7, 4))
bars = ax2.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue', alpha=0.8)
ax2.set_xlabel('Importance score')
ax2.set_title('Feature importance — Random Forest\n(which inputs most affect Cd)')
ax2.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: GP uncertainty band (predict across spoiler_angle range)
fig3, ax3 = plt.subplots(figsize=(7, 4))
spoiler_idx = FEATURES.index('spoiler_angle')
spoiler_range = np.linspace(X[:, spoiler_idx].min(), X[:, spoiler_idx].max(), 100)
X_test = np.zeros((100, len(FEATURES)))
# Hold all other features at their mean
for j in range(len(FEATURES)):
    X_test[:, j] = X[:, j].mean()
X_test[:, spoiler_idx] = spoiler_range
X_test_scaled = scaler.transform(X_test)

gp.fit(X_scaled, y)
y_gp, y_std = gp.predict(X_test_scaled, return_std=True)

ax3.plot(spoiler_range, y_gp, 'b-', linewidth=2, label='GP mean prediction')
ax3.fill_between(spoiler_range, y_gp - 2*y_std, y_gp + 2*y_std,
                 alpha=0.2, color='blue', label='95% confidence interval')
ax3.scatter(X[:, spoiler_idx], y, color='red', zorder=5, s=40,
            label='Training data', edgecolors='black', linewidth=0.5)
ax3.set_xlabel('Spoiler angle (deg)')
ax3.set_ylabel('Cd')
ax3.set_title('Gaussian Process — Cd vs spoiler angle\n(other params held at mean)')
ax3.legend(fontsize=9)
plt.tight_layout()
plt.savefig('gp_uncertainty.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 9 — Save the best model
# Save both the model and the scaler (needed for future predictions)
# ============================================================
joblib.dump(gb, 'surrogate_gradient_boosting.pkl')
joblib.dump(gp, 'surrogate_gaussian_process.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("\nModels saved: surrogate_gradient_boosting.pkl, surrogate_gaussian_process.pkl")

# ============================================================
# STEP 10 — Predict Cd for a new design
# ============================================================
new_design = pd.DataFrame([{
    'yaw': 3.0,
    'speed': 120.0,
    'rear_angle': 28.0,
    'spoiler_angle': 10.0,
    'ground_clearance': 140.0,
    'frontal_area': 2.25
}])

new_scaled = scaler.transform(new_design[FEATURES])
cd_pred_gb = gb.predict(new_design[FEATURES])[0]
cd_pred_gp, cd_std = gp.predict(new_scaled, return_std=True)

print("\n" + "="*50)
print("NEW DESIGN PREDICTION")
print("="*50)
print(f"Gradient Boosting Cd: {cd_pred_gb:.5f}")
print(f"Gaussian Process Cd:  {cd_pred_gp[0]:.5f} ± {cd_std[0]:.5f} (1-sigma)")
print(f"GP 95% interval:      [{cd_pred_gp[0]-2*cd_std[0]:.5f}, {cd_pred_gp[0]+2*cd_std[0]:.5f}]")
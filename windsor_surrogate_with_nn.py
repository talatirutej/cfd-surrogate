# windsor_surrogate_with_nn.py
# ============================================================
# CFD Surrogate Model — WindsorML Dataset
# Mahindra Research Valley — Aerodynamics Team
#
# Models trained:
#   1. Random Forest
#   2. Gradient Boosting
#   3. Gaussian Process
#   4. MLP Neural Network  <-- NEW
#
# HOW TO RUN:
#   Step 1: pip install scikit-learn pandas numpy matplotlib joblib huggingface_hub
#   Step 2: python windsor_surrogate_with_nn.py
#   It will auto-download WindsorML CSVs from HuggingFace (no login needed)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.neural_network import MLPRegressor          # <-- Neural Network
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ============================================================
# STEP 1 — Download WindsorML CSVs from HuggingFace
# No login or access request required — fully open dataset
# ============================================================
print("="*60)
print("  WindsorML Surrogate Model — Mahindra Research Valley")
print("="*60)
print("\nStep 1: Downloading WindsorML from HuggingFace...")

try:
    from huggingface_hub import hf_hub_download

    force_path = hf_hub_download(
        repo_id="neashton/windsorml",
        filename="force_mom_all.csv",
        repo_type="dataset"
    )
    geo_path = hf_hub_download(
        repo_id="neashton/windsorml",
        filename="geo_parameters_all.csv",
        repo_type="dataset"
    )
    print("  Downloaded successfully.")

    force_df = pd.read_csv(force_path)
    geo_df   = pd.read_csv(geo_path)

    print(f"\n  force_mom_all columns : {force_df.columns.tolist()}")
    print(f"  geo_parameters columns: {geo_df.columns.tolist()}")
    print(f"  Force rows: {len(force_df)}  |  Geo rows: {len(geo_df)}")

    # --------------------------------------------------------
    # Merge on run identifier
    # Windsor uses 'run_id' or 'Run' — detect automatically
    # --------------------------------------------------------
    force_key = [c for c in force_df.columns if 'run' in c.lower()][0]
    geo_key   = [c for c in geo_df.columns   if 'run' in c.lower()][0]

    force_df = force_df.rename(columns={force_key: 'run_id'})
    geo_df   = geo_df.rename(columns={geo_key:   'run_id'})
    df = pd.merge(geo_df, force_df, on='run_id')
    print(f"\n  Merged dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # --------------------------------------------------------
    # Identify geometric input columns and Cd output
    # Geometric cols = from geo_df (excluding run_id)
    # Cd = column containing 'cd' (case-insensitive) from force_df
    # --------------------------------------------------------
    geo_cols   = [c for c in geo_df.columns if c != 'run_id']
    cd_col_candidates = [c for c in force_df.columns
                         if 'cd' in c.lower() and 'run' not in c.lower()]

    if not cd_col_candidates:
        raise ValueError("Cannot find Cd column in force_mom_all.csv. "
                         "Print force_df.columns and update cd_col manually.")

    CD_COL   = cd_col_candidates[0]
    FEATURES = geo_cols
    print(f"\n  Geometric input features ({len(FEATURES)}): {FEATURES}")
    print(f"  Target (Cd column): '{CD_COL}'")

    # Drop rows with NaN in features or target
    df = df[FEATURES + [CD_COL]].dropna()
    print(f"  Rows after dropping NaN: {len(df)}")

    X = df[FEATURES].values
    y = df[CD_COL].values

    DATASET_SOURCE = "WindsorML (real HF-LES CFD)"

except Exception as e:
    # --------------------------------------------------------
    # FALLBACK: use your existing synthetic CSV if download
    # fails or huggingface_hub is not installed
    # --------------------------------------------------------
    print(f"\n  HuggingFace download failed: {e}")
    print("  Falling back to cfd_data.csv (your synthetic dataset).")
    print("  To use real WindsorML data: pip install huggingface_hub\n")

    df = pd.read_csv('cfd_data.csv')
    FEATURES = ['yaw', 'speed', 'rear_angle', 'spoiler_angle',
                'ground_clearance', 'frontal_area']
    CD_COL   = 'actual_cd'
    X = df[FEATURES].values
    y = df[CD_COL].values
    DATASET_SOURCE = "Synthetic CFD data (cfd_data.csv)"

print(f"\n  Dataset: {DATASET_SOURCE}")
print(f"  Shape  : {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Cd range: {y.min():.4f} — {y.max():.4f}  (mean={y.mean():.4f})")

# ============================================================
# STEP 2 — Scale features
# All models benefit from scaling; mandatory for GP and MLP
# ============================================================
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# STEP 3 — Train / Test split (80/20)
# With 355 real samples we can afford a held-out test set
# With <60 samples, use only cross-validation (no split)
# ============================================================
N = len(X)
USE_SPLIT = N >= 100

if USE_SPLIT:
    X_tr, X_te, Xs_tr, Xs_te, y_tr, y_te = train_test_split(
        X, X_scaled, y, test_size=0.2, random_state=42)
    print(f"\n  Train: {len(X_tr)}  |  Test: {len(X_te)}")
else:
    X_tr = X_te = X
    Xs_tr = Xs_te = X_scaled
    y_tr = y_te = y
    print(f"\n  Small dataset — using full data + cross-validation only.")

# ============================================================
# STEP 4 — Define all four models
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,
    n_restarts_optimizer=10
)

# ── MLP Neural Network ──────────────────────────────────────
# Architecture: Input → 128 → 64 → 32 → Output (Cd)
# Each layer uses ReLU activation (captures nonlinearity)
# Adam optimiser — most reliable for regression
# Early stopping: training stops when validation error stops
#   improving — this prevents overfitting automatically
# max_iter=2000: allows enough epochs to converge
# ────────────────────────────────────────────────────────────
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=2000,
    early_stopping=True,     # use 10% of training data as validation
    validation_fraction=0.1,
    n_iter_no_change=30,     # stop if no improvement for 30 epochs
    random_state=42,
    verbose=False
)

MODELS = {
    'Random Forest':     {'model': rf,  'scaled': False},
    'Gradient Boosting': {'model': gb,  'scaled': False},
    'Gaussian Process':  {'model': gp,  'scaled': True},
    'Neural Network':    {'model': mlp, 'scaled': True},   # <-- NEW
}

# ============================================================
# STEP 5 — Cross-validation (honest RMSE)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (5-fold)")
print("="*60)

cv_results = {}
for name, cfg in MODELS.items():
    X_in = X_scaled if cfg['scaled'] else X
    scores = cross_val_score(
        cfg['model'], X_in, y,
        cv=kf, scoring='neg_root_mean_squared_error'
    )
    rmse_cv = -scores
    cv_results[name] = {'mean': rmse_cv.mean(), 'std': rmse_cv.std()}
    bar = '#' * int(rmse_cv.mean() * 5000)
    print(f"  {name:22s}: RMSE = {rmse_cv.mean():.5f} ± {rmse_cv.std():.5f}  {bar}")

# ============================================================
# STEP 6 — Train on train split, evaluate on held-out test
# ============================================================
print("\n" + "="*60)
print("TEST SET METRICS (honest — model has never seen these rows)")
print("="*60)

test_results = {}
for name, cfg in MODELS.items():
    m   = cfg['model']
    sc  = cfg['scaled']
    X_in_tr = Xs_tr if sc else X_tr
    X_in_te = Xs_te if sc else X_te

    m.fit(X_in_tr, y_tr)
    y_pred = m.predict(X_in_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    test_results[name] = {'pred': y_pred, 'mae': mae, 'rmse': rmse, 'r2': r2}
    print(f"\n  {name}")
    print(f"    MAE  = {mae:.5f}")
    print(f"    RMSE = {rmse:.5f}  {'EXCELLENT' if rmse<0.003 else 'GOOD' if rmse<0.008 else 'OK'}")
    print(f"    R²   = {r2:.4f}")

# ============================================================
# STEP 7 — Feature importance (tree models only)
# MLP doesn't have built-in feature importance
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Random Forest — most reliable method)")
print("="*60)

rf.fit(X_tr, y_tr)
feat_imp = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in feat_imp.iterrows():
    bar = '█' * int(row['Importance'] * 40)
    print(f"  {row['Feature']:22s}: {row['Importance']:.4f}  {bar}")

# ============================================================
# STEP 8 — Neural network training curve
# Shows how the MLP loss decreased over epochs
# ============================================================
print(f"\n  MLP converged after {len(mlp.loss_curve_)} epochs")
print(f"  Final training loss : {mlp.loss_curve_[-1]:.6f}")
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
    print(f"  Final validation R² : {mlp.best_validation_score_:.4f}")

# ============================================================
# STEP 9 — Plots (2 rows × 2 cols)
# Row 1: Actual vs Predicted for all 4 models
# Row 2: Feature importance + NN training curve
# ============================================================
COLORS = {
    'Random Forest':     '#4f8ef7',
    'Gradient Boosting': '#f76f6f',
    'Gaussian Process':  '#4ecb71',
    'Neural Network':    '#f7b731',
}
MARKERS = {
    'Random Forest':     'o',
    'Gradient Boosting': 's',
    'Gaussian Process':  'D',
    'Neural Network':    '^',
}

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'CFD Surrogate Models — Mahindra Research Valley\nDataset: {DATASET_SOURCE}',
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

# ── Row 1: Actual vs Predicted (4 subplots) ─────────────────
for idx, (name, cfg) in enumerate(MODELS.items()):
    ax = fig.add_subplot(gs[0, idx])
    res = test_results[name]
    y_pred = res['pred']
    col = COLORS[name]

    ax.scatter(y_te, y_pred, alpha=0.7, color=col,
               edgecolors='black', linewidth=0.4,
               marker=MARKERS[name], s=40, zorder=3)

    lims = [min(y_te.min(), y_pred.min()) - 0.005,
            max(y_te.max(), y_pred.max()) + 0.005]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='Perfect fit')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Actual Cd', fontsize=9)
    ax.set_ylabel('Predicted Cd', fontsize=9)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.text(0.05, 0.95,
            f"R²   = {res['r2']:.3f}\nRMSE = {res['rmse']:.5f}\nMAE  = {res['mae']:.5f}",
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# ── Row 2 Left (span 2): Feature importance ─────────────────
ax_feat = fig.add_subplot(gs[1, :2])
bars = ax_feat.barh(feat_imp['Feature'], feat_imp['Importance'],
                    color='#4f8ef7', alpha=0.85, edgecolor='black', linewidth=0.4)
ax_feat.set_xlabel('Importance score', fontsize=9)
ax_feat.set_title('Feature importance (Random Forest)', fontsize=10, fontweight='bold')
ax_feat.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
ax_feat.grid(axis='x', alpha=0.3)

# ── Row 2 Right (span 2): NN training curve ─────────────────
ax_loss = fig.add_subplot(gs[1, 2:])
epochs = range(1, len(mlp.loss_curve_) + 1)
ax_loss.plot(epochs, mlp.loss_curve_, color='#f7b731',
             linewidth=2, label='Training loss')

if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
    ax_loss2 = ax_loss.twinx()
    ax_loss2.plot(epochs, mlp.validation_scores_,
                  color='#4ecb71', linewidth=1.5,
                  linestyle='--', label='Validation R²')
    ax_loss2.set_ylabel('Validation R²', fontsize=9, color='#4ecb71')
    ax_loss2.tick_params(axis='y', colors='#4ecb71')
    ax_loss2.legend(loc='lower right', fontsize=8)

ax_loss.set_xlabel('Training epoch', fontsize=9)
ax_loss.set_ylabel('MSE loss', fontsize=9)
ax_loss.set_title('Neural Network training curve\n(loss should decrease smoothly)',
                  fontsize=10, fontweight='bold')
ax_loss.legend(loc='upper right', fontsize=8)
ax_loss.grid(alpha=0.3)

# Annotate the early-stop point
best_epoch = np.argmin(mlp.loss_curve_)
ax_loss.axvline(best_epoch + 1, color='red', linestyle=':',
                linewidth=1, alpha=0.7)
ax_loss.text(best_epoch + 1, ax_loss.get_ylim()[1] * 0.95,
             f' Best\n epoch\n {best_epoch+1}',
             color='red', fontsize=7, va='top')

plt.savefig('windsor_all_models.png', dpi=150, bbox_inches='tight')
print("\n  Plot saved: windsor_all_models.png")
plt.show()

# ============================================================
# STEP 10 — Model comparison summary table
# ============================================================
print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)
print(f"  {'Model':22s}  {'CV RMSE':>10}  {'Test RMSE':>10}  {'Test R²':>8}")
print("  " + "-"*55)
for name in MODELS:
    cv  = cv_results[name]
    ts  = test_results[name]
    tag = " <-- BEST" if name == min(test_results,
          key=lambda n: test_results[n]['rmse']) else ""
    print(f"  {name:22s}  {cv['mean']:10.5f}  "
          f"{ts['rmse']:10.5f}  {ts['r2']:8.4f}{tag}")

print("\n  RMSE guide: < 0.003 = excellent  |  0.003-0.008 = good  |  > 0.008 = needs work")

# ============================================================
# STEP 11 — Save all models
# ============================================================
for name, cfg in MODELS.items():
    fname = 'surrogate_' + name.lower().replace(' ', '_') + '.pkl'
    joblib.dump(cfg['model'], fname)
joblib.dump(scaler, 'windsor_scaler.pkl')
print("\n  All 4 models saved as .pkl files")
print("  Scaler saved: windsor_scaler.pkl")

# ============================================================
# STEP 12 — Predict Cd for a new design
# Uses all 4 models and shows comparison
# ============================================================
print("\n" + "="*60)
print("PREDICT Cd FOR A NEW DESIGN")
print("="*60)

# Use mean values of each feature as a sample new design
new_vals = {feat: float(np.mean(X[:, i]))
            for i, feat in enumerate(FEATURES)}
print("  New design (mean values of each parameter):")
for k, v in new_vals.items():
    print(f"    {k:22s}: {v:.4f}")

new_row  = np.array([[new_vals[f] for f in FEATURES]])
new_scaled = scaler.transform(new_row)

print(f"\n  {'Model':22s}  {'Predicted Cd':>14}")
print("  " + "-"*40)
for name, cfg in MODELS.items():
    X_in = new_scaled if cfg['scaled'] else new_row
    cd   = cfg['model'].predict(X_in)[0]
    if name == 'Gaussian Process':
        _, std = gp.predict(new_scaled, return_std=True)
        print(f"  {name:22s}  {cd:14.5f}  (±{std[0]:.5f})")
    else:
        print(f"  {name:22s}  {cd:14.5f}")

print("\n  Done. All results printed and plots saved.")
print("="*60)
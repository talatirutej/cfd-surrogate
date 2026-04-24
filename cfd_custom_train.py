# ============================================================
# Driver — train & benchmark the Custom CFD Surrogate Model
# ("MRV-Aero-Stack v1") vs the baseline RF / GB / GP / NN models.
#
# Usage:
#   1. Put cfd_data.csv in the same folder.
#      (or the WindsorML CSV with flag below)
#   2. python cfd_custom_train.py
#
# Outputs:
#   - custom_model_mrv.pkl        — production Cd model
#   - custom_vs_baselines.png     — 2x3 scatter benchmark
#   - feature_importance_custom.png
#   - uncertainty_band.png
#   - benchmark_report.txt
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# import the custom model from the module in this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfd_custom_model import CustomCFDModel


# ------------------------------------------------------------
# CONFIG — edit these if using a different dataset
# ------------------------------------------------------------
DATA_CSV = "cfd_data.csv"
# Mahindra internal schema
FEATURES = ["yaw", "speed", "rear_angle", "spoiler_angle",
            "ground_clearance", "frontal_area"]
TARGET = "actual_cd"
SPEED_COL = "speed"
AREA_COL = "frontal_area"

# If you're running on WindsorML, set these:
# FEATURES = ["ratio_length_back_fast", "ratio_height_nose_windshield",
#             "ratio_height_fast_back", "side_taper",
#             "clearance", "bottom_taper_angle", "frontal_area"]
# TARGET = "cd"
# SPEED_COL = None
# AREA_COL = "frontal_area"


# ------------------------------------------------------------
# Baselines (same config as the old cfd_surrogate.py / windsor_* )
# ------------------------------------------------------------
def train_baselines(X_raw: np.ndarray, y: np.ndarray,
                    X_scaled: np.ndarray, kf: KFold) -> dict:
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=3, random_state=42)
    kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  normalize_y=True,
                                  n_restarts_optimizer=3,
                                  random_state=0)
    nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                      solver="adam", learning_rate_init=1e-3,
                      max_iter=2000, early_stopping=True,
                      validation_fraction=0.2, random_state=42)

    specs = {"Random Forest":     (rf, X_raw),
             "Gradient Boosting": (gb, X_raw),
             "Gaussian Process":  (gp, X_scaled),
             "Neural Network":    (nn, X_scaled)}

    results = {}
    n = len(y)
    for name, (model, X_use) in specs.items():
        oof = np.zeros(n)
        for tr, va in kf.split(X_use):
            # fresh clone via sklearn.base.clone
            from sklearn.base import clone
            m_ = clone(model)
            m_.fit(X_use[tr], y[tr])
            oof[va] = m_.predict(X_use[va])
        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        mae  = float(mean_absolute_error(y, oof))
        r2   = float(r2_score(y, oof))
        results[name] = {"oof": oof, "rmse": rmse, "mae": mae, "r2": r2}
    return results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: '{DATA_CSV}' not found.")
        print("Place your CFD data CSV (with columns matching "
              f"{FEATURES + [TARGET]}) in this folder and re-run.")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {DATA_CSV}: {df.shape[0]} rows, {df.shape[1]} cols")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns in CSV: {missing}")
        sys.exit(1)

    X_raw = df[FEATURES].values
    y     = df[TARGET].values
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ---- baselines ----
    print("\n[1/3] Training baseline models (OOF 5-fold CV)...")
    baselines = train_baselines(X_raw, y, X_scaled, kf)
    for name, m in baselines.items():
        print(f"  {name:18s}  R²={m['r2']: .4f}  "
              f"RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}")

    # ---- custom model ----
    print("\n[2/3] Training Custom Built Model (MRV-Aero-Stack v1)...")
    custom = CustomCFDModel(
        feature_cols=FEATURES,
        speed_col=SPEED_COL,
        frontal_area_col=AREA_COL,
        log_target=True,
        n_seeds=3,
        n_folds=5,
        random_state=42,
    )
    custom.fit(df, y, verbose=True)

    oof_custom = custom.oof_ @ custom.weights_ \
        if custom.final_mode_ == "hill_climb" \
        else custom.ridge_.predict(custom.oof_)

    custom_metrics = {
        "r2":   float(r2_score(y, oof_custom)),
        "rmse": float(np.sqrt(mean_squared_error(y, oof_custom))),
        "mae":  float(mean_absolute_error(y, oof_custom)),
    }

    # ---- plots ----
    print("\n[3/3] Generating plots and report...")
    all_results = {**baselines,
                   "Custom Built Model": {"oof": oof_custom, **custom_metrics}}

    # 2x3 grid (baselines + custom)
    order = ["Random Forest", "Gradient Boosting", "Gaussian Process",
             "Neural Network", "Custom Built Model"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "CFD Surrogate Models — OOF (5-fold CV) Benchmark\n"
        "Mahindra Research Valley — MRV-Aero-Stack v1",
        fontsize=13, fontweight="bold"
    )
    colors = ["steelblue", "coral", "green", "goldenrod", "crimson"]
    for ax, name, color in zip(axes.flat, order, colors):
        pred = all_results[name]["oof"]
        r2_ = all_results[name]["r2"]
        rmse_ = all_results[name]["rmse"]
        mae_ = all_results[name]["mae"]
        ax.scatter(y, pred, alpha=0.7, color=color,
                   edgecolors="black", linewidth=0.4, s=40)
        lims = [min(y.min(), pred.min()) * 0.99,
                max(y.max(), pred.max()) * 1.01]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6, label="Perfect fit")
        ax.set_xlabel("Actual Cd"); ax.set_ylabel("Predicted Cd")
        ax.set_title(name, fontweight="bold")
        ax.text(0.04, 0.96,
                f"R² = {r2_:.3f}\nRMSE = {rmse_:.5f}\nMAE  = {mae_:.5f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.2)

    # summary panel
    ax = axes.flat[5]
    ax.axis("off")
    names = order
    rmses = [all_results[n]["rmse"] for n in names]
    bars = ax.barh(names, rmses,
                   color=["lightgray"] * 4 + ["crimson"], alpha=0.85)
    ax.set_xlabel("OOF RMSE (lower is better)")
    ax.set_title("Model Comparison — OOF RMSE", fontweight="bold")
    for bar, v in zip(bars, rmses):
        ax.text(v, bar.get_y() + bar.get_height() / 2,
                f"  {v:.5f}", va="center", fontsize=9)
    ax.axis("on"); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("custom_vs_baselines.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved custom_vs_baselines.png")

    # Uncertainty band plot — Cd vs a sweep of the most important feature
    try:
        # Use xgb feature importances from the first seed's fitted model
        xgb_model = custom.fitted_models_["xgb"][0]
        imps = xgb_model.feature_importances_
        top_feat_idx = int(np.argmax(imps))
        top_feat = custom.engineered_cols_[top_feat_idx]

        # Only sweep over a raw (non-engineered) feature for interpretability
        raw_feat = FEATURES[0]
        if any(f in FEATURES for f in custom.engineered_cols_):
            # pick whichever raw feature has highest univariate corr with y
            corrs = {f: abs(np.corrcoef(df[f], y)[0, 1]) for f in FEATURES}
            raw_feat = max(corrs, key=corrs.get)

        rng = np.linspace(df[raw_feat].min(), df[raw_feat].max(), 60)
        sweep = pd.DataFrame({f: [df[f].mean()] * 60 for f in FEATURES})
        sweep[raw_feat] = rng

        mean, std = custom.predict_with_uncertainty(sweep)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(rng, mean, "crimson", lw=2, label="Custom model — mean Cd")
        ax2.fill_between(rng, mean - 2 * std, mean + 2 * std,
                         color="crimson", alpha=0.2,
                         label="±2σ epistemic band")
        ax2.scatter(df[raw_feat], y, color="steelblue",
                    edgecolors="black", linewidth=0.4, s=40,
                    label="CFD training points")
        ax2.set_xlabel(raw_feat)
        ax2.set_ylabel("Cd")
        ax2.set_title(f"Custom model — Cd vs {raw_feat}\n"
                      "(other params held at dataset mean)",
                      fontweight="bold")
        ax2.legend(); ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("uncertainty_band.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  saved uncertainty_band.png")

        # Feature importance (averaged XGB importances over seeds)
        imps_all = np.mean(
            [m.feature_importances_ for m in custom.fitted_models_["xgb"]],
            axis=0)
        order_idx = np.argsort(imps_all)[::-1][:15]
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        bars = ax3.barh([custom.engineered_cols_[i] for i in order_idx][::-1],
                        imps_all[order_idx][::-1],
                        color="steelblue", alpha=0.85)
        ax3.set_xlabel("Importance (XGBoost, averaged over 5 seeds)")
        ax3.set_title("Top-15 features — Custom Built Model",
                      fontweight="bold")
        ax3.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax3.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig("feature_importance_custom.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        print("  saved feature_importance_custom.png")
    except Exception as e:
        print(f"  (skipped uncertainty / importance plot: {e})")

    # ---- report ----
    lines = []
    lines.append("=" * 64)
    lines.append(" CFD SURROGATE MODEL BENCHMARK REPORT")
    lines.append(" Mahindra Research Valley — MRV-Aero-Stack v1")
    lines.append("=" * 64)
    lines.append(f" Dataset: {DATA_CSV}   (n={len(df)})")
    lines.append(f" Target:  {TARGET}")
    lines.append(f" Features: {FEATURES}")
    lines.append("")
    lines.append(" Out-of-fold 5-fold CV metrics (the honest numbers):")
    lines.append(" " + "-" * 60)
    lines.append(f"   {'Model':<22}{'R²':>10}{'RMSE':>12}{'MAE':>12}")
    for name in order:
        m = all_results[name]
        lines.append(f"   {name:<22}{m['r2']:>10.4f}"
                     f"{m['rmse']:>12.5f}{m['mae']:>12.5f}")
    lines.append("")
    # improvement
    best_baseline_rmse = min(all_results[n]["rmse"]
                             for n in order if n != "Custom Built Model")
    best_baseline_name = min(
        (n for n in order if n != "Custom Built Model"),
        key=lambda n: all_results[n]["rmse"])
    pct = 100 * (best_baseline_rmse - custom_metrics["rmse"]) \
        / best_baseline_rmse
    lines.append(f" Improvement over best baseline "
                 f"({best_baseline_name}):  {pct:+.1f}% RMSE")
    lines.append("")
    lines.append(" Hill-climb weights inside the ensemble:")
    for k, v in custom.cv_metrics_["hill_climb_weights"].items():
        lines.append(f"   {k:<5}  {v:.3f}")
    lines.append(f" Final combiner chosen: {custom.final_mode_}")
    lines.append("")
    lines.append(" Architecture:")
    lines.append("   - Physics-engineered features (interactions, log, sqrt,")
    lines.append("     dynamic-pressure & Reynolds proxies)")
    lines.append("   - log(Cd) target transform")
    lines.append("   - 5 heterogeneous base learners: XGB, LGBM, CatBoost,")
    lines.append("     ExtraTrees, GaussianProcess")
    lines.append("   - 5-seed bagging inside each base learner")
    lines.append("   - Hill-climb OOF weight optimisation + Ridge meta, pick winner")
    lines.append("")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print("\n" + report)
    with open("benchmark_report.txt", "w") as fh:
        fh.write(report)
    print("\n  saved benchmark_report.txt")

    # ---- save model ----
    joblib.dump(custom, "custom_model_mrv.pkl")
    print("  saved custom_model_mrv.pkl")
    print("\nDone.")


if __name__ == "__main__":
    main()
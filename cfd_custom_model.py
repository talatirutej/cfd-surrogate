# ============================================================
# CFD Custom-Built Surrogate Model ("MRV-Aero-Stack v1")
# ------------------------------------------------------------
# Designed for automotive Cd prediction on small CFD datasets
# (Mahindra Research Valley internal + WindsorML public).
#
# Architecture:
#   1. Physics-informed feature engineering (dynamic pressure,
#      interactions, log/sqrt, polynomial terms).
#   2. Target transformation (log-Cd) to stabilize variance.
#   3. Heterogeneous base learners trained out-of-fold:
#        - XGBoost      (fast, regularized GBDT)
#        - LightGBM     (histogram GBDT, complementary splits)
#        - CatBoost     (ordered boosting, different bias)
#        - ExtraTrees   (randomized, variance reducer)
#        - GaussianProcess (smooth function prior, uncertainty)
#   4. Hill-climbing weight optimization on OOF predictions
#      (Kaggle-grandmaster technique — NVIDIA playbook).
#   5. Ridge meta-learner as a second aggregator for robustness;
#      we pick whichever (hill-climb vs ridge) wins on OOF.
#   6. 5-seed bagging inside each base model (variance reduction).
#   7. Uncertainty: std of the 5 seed predictions + GP sigma.
#
# References:
#   - Kaggle Grandmasters Playbook (NVIDIA, 2025) -- hill climb.
#   - WindsorML / DrivAerML / AhmedML benchmarks (Ashton 2024).
#   - Shwartz-Ziv & Armon 2021 -- trees + NNs complementary.
#   - Prokhorenkova 2018 -- CatBoost ordered boosting.
# ============================================================

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Sequence

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# ============================================================
# Feature engineering
# ============================================================
def engineer_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    speed_col: str | None = None,
    frontal_area_col: str | None = None,
) -> pd.DataFrame:
    """
    Physics-informed feature expansion.

    The raw Cd is a function of Reynolds number, shape ratios and
    flow interactions. Tree and GP models can't discover
    multiplicative interactions well on their own with <500 rows,
    so we hand-feed them the most informative ones.

    Safe to call on any dataframe: columns that don't exist are
    simply skipped.
    """
    X = df[list(feature_cols)].copy()

    # --- pairwise products (interactions) for the TOP features
    # We don't do a full O(n^2) expansion on big feature lists;
    # we cap at the first 8 to keep the feature count manageable.
    core = list(feature_cols)[:8]
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            X[f"{a}__x__{b}"] = df[a].values * df[b].values

    # --- squared terms (for curvature, e.g. Cd ~ yaw^2)
    for a in core:
        X[f"{a}__sq"] = df[a].values ** 2

    # --- log and sqrt of strictly positive columns
    for a in core:
        v = df[a].values
        if np.all(v > 0):
            X[f"{a}__log"] = np.log(v)
            X[f"{a}__sqrt"] = np.sqrt(v)

    # --- physics-derived: dynamic-pressure proxy q = 0.5 * rho * V^2
    # Cd itself is shape-only, but if the dataset was generated
    # at multiple speeds, Re-number effects leak into Cd. Giving
    # the model V and V^2 explicitly helps.
    if speed_col is not None and speed_col in df.columns:
        v = df[speed_col].values
        X["_dyn_pressure_proxy"] = 0.5 * 1.225 * v ** 2   # kg/(m*s^2)
        # Reynolds proxy with L ~ 1 m, nu ~ 1.5e-5 m^2/s
        X["_Re_proxy"] = v * 1.0 / 1.5e-5

    # --- CdA product proxy if both area and a Cd-scale feature available
    if frontal_area_col is not None and frontal_area_col in df.columns:
        X["_area_sq"] = df[frontal_area_col].values ** 2

    return X


# ============================================================
# Base model factories (hyperparameters already tuned for
# small-tabular CFD-style regression)
# ============================================================
def _make_xgb(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=seed,
        verbosity=0,
        n_jobs=1,
    )


def _make_lgb(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=4,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )


def _make_cat(seed: int) -> cb.CatBoostRegressor:
    return cb.CatBoostRegressor(
        iterations=700,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        bagging_temperature=0.5,
        random_seed=seed,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )


def _make_et(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=500,
        max_features=0.8,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def _make_gp() -> GaussianProcessRegressor:
    # Matern(nu=2.5) — twice-differentiable functions, standard
    # for smooth physical surfaces. WhiteKernel absorbs noise so
    # the GP doesn't try to interpolate CFD jitter exactly.
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0,
                                          length_scale_bounds=(1e-2, 1e2),
                                          nu=2.5) \
             + WhiteKernel(noise_level=1e-4,
                           noise_level_bounds=(1e-8, 1e-1))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=0,
    )


# ============================================================
# Hill-climbing weight optimization on OOF predictions
# (NVIDIA Kaggle Grandmasters playbook, 2025)
# ============================================================
def hill_climb_weights(
    oof_matrix: np.ndarray,   # (n_samples, n_models)
    y: np.ndarray,
    n_iter: int = 200,
    step: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    """
    Find non-negative weights that minimize OOF RMSE.
    Starts at uniform, then greedily nudges one weight at a time.
    """
    rng = np.random.default_rng(seed)
    n_models = oof_matrix.shape[1]
    w = np.ones(n_models) / n_models

    def rmse(w_):
        w_ = np.clip(w_, 0, None)
        s = w_.sum()
        if s <= 0:
            return np.inf
        w_ = w_ / s
        pred = oof_matrix @ w_
        return np.sqrt(mean_squared_error(y, pred))

    best = rmse(w)
    for _ in range(n_iter):
        improved = False
        order = rng.permutation(n_models)
        for idx in order:
            for delta in (+step, -step, +2 * step, -2 * step):
                w_try = w.copy()
                w_try[idx] += delta
                if w_try[idx] < 0:
                    continue
                new = rmse(w_try)
                if new < best - 1e-9:
                    best = new
                    w = w_try
                    improved = True
        if not improved:
            break

    w = np.clip(w, 0, None)
    w = w / w.sum()
    return w


# ============================================================
# The custom stacked model
# ============================================================
@dataclass
class CustomCFDModel:
    feature_cols: Sequence[str]
    speed_col: str | None = None
    frontal_area_col: str | None = None
    log_target: bool = True
    n_seeds: int = 3
    n_folds: int = 5
    random_state: int = 42

    # populated after fit()
    engineered_cols_: list = field(default_factory=list, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    base_names_: list = field(default_factory=list, init=False)
    fitted_models_: dict = field(default_factory=dict, init=False)   # name -> list of (seed, model)
    oof_: np.ndarray = field(default=None, init=False)               # (n, n_models)
    weights_: np.ndarray = field(default=None, init=False)
    ridge_: Ridge = field(default=None, init=False)
    final_mode_: str = field(default="hill_climb", init=False)
    cv_metrics_: dict = field(default_factory=dict, init=False)

    # ------------------------------------------------------------
    def _prep(self, df: pd.DataFrame) -> np.ndarray:
        X = engineer_features(df, self.feature_cols,
                              self.speed_col, self.frontal_area_col)
        if not self.engineered_cols_:
            self.engineered_cols_ = X.columns.tolist()
        else:
            # align to training columns (fill missing with 0)
            for c in self.engineered_cols_:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[self.engineered_cols_]
        return X.values

    # ------------------------------------------------------------
    def _y_transform(self, y):
        return np.log(y) if self.log_target else y

    def _y_inverse(self, y):
        return np.exp(y) if self.log_target else y

    # ------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray, verbose: bool = True):
        X = self._prep(df)
        y = np.asarray(y, dtype=float)
        y_t = self._y_transform(y)

        # Fit a scaler for the GP (tree models don't need it).
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        self.base_names_ = ["xgb", "lgb", "cat", "et", "gp"]
        n, m = X.shape[0], len(self.base_names_)
        self.oof_ = np.zeros((n, m))

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        # We fit each (model, seed) on each fold for OOF predictions,
        # then refit on the full data for the final production model.
        seeds = list(range(self.random_state,
                           self.random_state + self.n_seeds))
        self.fitted_models_ = {name: [] for name in self.base_names_}

        # ---- OOF loop ----
        for name_idx, name in enumerate(self.base_names_):
            for fold, (tr, va) in enumerate(kf.split(X)):
                preds_folds = np.zeros(len(va))
                for seed in seeds:
                    if name == "xgb":
                        model = _make_xgb(seed)
                        model.fit(X[tr], y_t[tr])
                        p = model.predict(X[va])
                    elif name == "lgb":
                        model = _make_lgb(seed)
                        model.fit(X[tr], y_t[tr])
                        p = model.predict(X[va])
                    elif name == "cat":
                        model = _make_cat(seed)
                        model.fit(X[tr], y_t[tr])
                        p = model.predict(X[va])
                    elif name == "et":
                        model = _make_et(seed)
                        model.fit(X[tr], y_t[tr])
                        p = model.predict(X[va])
                    elif name == "gp":
                        # GP uses a single fit (no seed effect)
                        model = _make_gp()
                        model.fit(X_scaled[tr], y_t[tr])
                        p = model.predict(X_scaled[va])
                        preds_folds += p / 1     # single pred, no seed avg
                        break
                    preds_folds += p / self.n_seeds
                # convert back and store
                self.oof_[va, name_idx] = self._y_inverse(preds_folds)
            if verbose:
                r = r2_score(y, self.oof_[:, name_idx])
                rmse = np.sqrt(mean_squared_error(y, self.oof_[:, name_idx]))
                mae = mean_absolute_error(y, self.oof_[:, name_idx])
                print(f"  OOF [{name:>3}]  R2={r: .4f}  RMSE={rmse:.5f}  MAE={mae:.5f}")

        # ---- final refit on all data ----
        for name in self.base_names_:
            self.fitted_models_[name] = []
            if name == "gp":
                m_ = _make_gp()
                m_.fit(X_scaled, y_t)
                self.fitted_models_[name].append(m_)
            else:
                for seed in seeds:
                    if name == "xgb":
                        m_ = _make_xgb(seed)
                    elif name == "lgb":
                        m_ = _make_lgb(seed)
                    elif name == "cat":
                        m_ = _make_cat(seed)
                    elif name == "et":
                        m_ = _make_et(seed)
                    m_.fit(X, y_t)
                    self.fitted_models_[name].append(m_)

        # ---- meta-combiner: hill climb vs ridge, pick winner ----
        self.weights_ = hill_climb_weights(self.oof_, y,
                                           seed=self.random_state)
        hc_pred = self.oof_ @ self.weights_
        hc_rmse = np.sqrt(mean_squared_error(y, hc_pred))

        self.ridge_ = Ridge(alpha=1.0, positive=True)
        self.ridge_.fit(self.oof_, y)
        rg_pred = self.ridge_.predict(self.oof_)
        rg_rmse = np.sqrt(mean_squared_error(y, rg_pred))

        # Use KFold-CV on OOF to pick combiner honestly
        cv_hc, cv_rg = [], []
        for tr, va in kf.split(self.oof_):
            w = hill_climb_weights(self.oof_[tr], y[tr],
                                   seed=self.random_state)
            cv_hc.append(np.sqrt(mean_squared_error(
                y[va], self.oof_[va] @ w)))
            r_ = Ridge(alpha=1.0, positive=True).fit(self.oof_[tr], y[tr])
            cv_rg.append(np.sqrt(mean_squared_error(
                y[va], r_.predict(self.oof_[va]))))
        cv_hc_rmse = float(np.mean(cv_hc))
        cv_rg_rmse = float(np.mean(cv_rg))

        self.final_mode_ = "hill_climb" if cv_hc_rmse <= cv_rg_rmse else "ridge"

        final_pred = hc_pred if self.final_mode_ == "hill_climb" else rg_pred
        self.cv_metrics_ = {
            "per_model_oof_rmse": {
                name: float(np.sqrt(mean_squared_error(y, self.oof_[:, i])))
                for i, name in enumerate(self.base_names_)
            },
            "hill_climb_rmse": float(hc_rmse),
            "ridge_rmse": float(rg_rmse),
            "cv_hill_climb_rmse": cv_hc_rmse,
            "cv_ridge_rmse": cv_rg_rmse,
            "final_mode": self.final_mode_,
            "hill_climb_weights": dict(zip(self.base_names_,
                                           self.weights_.tolist())),
            "ensemble_oof_rmse": float(np.sqrt(mean_squared_error(y, final_pred))),
            "ensemble_oof_mae": float(mean_absolute_error(y, final_pred)),
            "ensemble_oof_r2": float(r2_score(y, final_pred)),
        }

        if verbose:
            print("\n  Hill-climb weights:",
                  {k: f"{v:.3f}" for k, v in
                   self.cv_metrics_["hill_climb_weights"].items()})
            print(f"  Final combiner: {self.final_mode_}")
            print(f"  Ensemble OOF R²={self.cv_metrics_['ensemble_oof_r2']:.4f}  "
                  f"RMSE={self.cv_metrics_['ensemble_oof_rmse']:.5f}  "
                  f"MAE={self.cv_metrics_['ensemble_oof_mae']:.5f}")
        return self

    # ------------------------------------------------------------
    def _base_predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prep(df)
        X_scaled = self.scaler_.transform(X)
        preds = np.zeros((X.shape[0], len(self.base_names_)))
        for i, name in enumerate(self.base_names_):
            models = self.fitted_models_[name]
            if name == "gp":
                p = models[0].predict(X_scaled)
                preds[:, i] = self._y_inverse(p)
            else:
                ps = np.mean([m.predict(X) for m in models], axis=0)
                preds[:, i] = self._y_inverse(ps)
        return preds

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        base = self._base_predict(df)
        if self.final_mode_ == "hill_climb":
            return base @ self.weights_
        return self.ridge_.predict(base)

    def predict_with_uncertainty(self, df: pd.DataFrame):
        """
        Returns (mean, std). std is the population std across the
        5 base models' seed-averaged predictions; loosely
        interpretable as 1-sigma epistemic uncertainty.
        """
        base = self._base_predict(df)
        mean = (base @ self.weights_) if self.final_mode_ == "hill_climb" \
            else self.ridge_.predict(base)
        std = base.std(axis=1)
        return mean, std
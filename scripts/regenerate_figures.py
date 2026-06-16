"""
Regenerate all figures that need raw or processed data.
Run once from the project root:
    python3.11 scripts/regenerate_figures.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import learning_curve

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw

PROCESSED   = ROOT / "data" / "processed"
FIGURES     = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42


# ── 1. numerical_distributions.png ───────────────────────────────────────────
print("[1/6] numerical_distributions.png ...")

df_raw = load_raw("property_sales.csv")
df = df_raw[df_raw["nature_of_property"].str.upper().str.startswith("R")].copy()
df["contract_date"] = pd.to_datetime(df["contract_date"], dayfirst=True, errors="coerce")
df["year"]  = df["contract_date"].dt.year
df["area"]  = pd.to_numeric(df["area"], errors="coerce")

numerical_cols = ["area", "year"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, numerical_cols):
    data = df[col].dropna().clip(upper=df[col].quantile(0.99))
    ax.hist(data, bins=60, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.set_xlabel(col)

plt.suptitle("Numerical Feature Distributions", y=1.02)
plt.tight_layout()
fig.savefig(FIGURES / "numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")

del df, df_raw

# ── Load processed data (used for all remaining figures) ─────────────────────
print("Loading processed data ...")
X_train = pd.read_parquet(PROCESSED / "X_train.parquet")
X_test  = pd.read_parquet(PROCESSED / "X_test.parquet")
y_train = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
y_test  = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()
feature_names = list(X_train.columns)
print(f"  {X_train.shape} train  |  {X_test.shape} test")

# ── 2. Fit models (Ridge, Lasso, RF, XGBoost) ────────────────────────────────
print("[models] Fitting Ridge and Lasso ...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=1e-3, max_iter=10_000)
lasso.fit(X_train, y_train)
print(f"  Ridge alpha=1.0 | Lasso alpha=1e-3, non-zero coefs: {(lasso.coef_ != 0).sum()}")

print("[models] Fitting XGBoost (200 rounds) ...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0,
)
xgb_model.fit(X_train, y_train)
print("  XGBoost done.")

print("[models] Fitting Random Forest (200 trees) ...")
rf = RandomForestRegressor(
    n_estimators=200, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_STATE,
)
rf.fit(X_train, y_train)
print("  Random Forest done.")


# ── 3. linear_coefficients.png ───────────────────────────────────────────────
print("[3/6] linear_coefficients.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, name in [(axes[0], ridge, "Ridge"), (axes[1], lasso, "Lasso")]:
    coefs = pd.Series(model.coef_, index=feature_names)
    coefs_nonzero = coefs[coefs != 0].sort_values()
    colors = ["coral" if c < 0 else "steelblue" for c in coefs_nonzero]
    ax.barh(range(len(coefs_nonzero)), coefs_nonzero.values, color=colors)
    ax.set_yticks(range(len(coefs_nonzero)))
    ax.set_yticklabels(coefs_nonzero.index)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{name} Coefficients (standardised features)")
    ax.set_xlabel("Coefficient value")

plt.suptitle("Linear Model Coefficients\n(positive = increases predicted price)",
             y=1.02, fontsize=12)
plt.tight_layout()
fig.savefig(FIGURES / "linear_coefficients.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")


# ── 4. feature_importances.png ───────────────────────────────────────────────
print("[4/6] feature_importances.png ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, name in [(axes[0], rf, "Random Forest"), (axes[1], xgb_model, "XGBoost")]:
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{name} - Feature Importance (impurity)")
    ax.set_xlabel("Importance")

plt.tight_layout()
fig.savefig(FIGURES / "feature_importances.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")


# ── 5. residual_analysis.png ─────────────────────────────────────────────────
print("[5/6] residual_analysis.png ...")

y_pred_log = xgb_model.predict(X_test)
y_pred_aud = np.expm1(y_pred_log)
y_true_aud = np.expm1(y_test)
residuals  = y_true_aud - y_pred_aud
pct_error  = (residuals / y_true_aud) * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

max_val = max(y_pred_aud.max(), y_true_aud.max()) / 1e6
axes[0].scatter(y_pred_aud / 1e6, y_true_aud / 1e6, alpha=0.3, s=5, color="steelblue")
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1)
axes[0].set_xlabel("Predicted Price (AUD M)")
axes[0].set_ylabel("Actual Price (AUD M)")
axes[0].set_title("Predicted vs. Actual (XGBoost)")

axes[1].scatter(y_pred_aud / 1e6, residuals / 1e3, alpha=0.3, s=5, color="coral")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Predicted Price (AUD M)")
axes[1].set_ylabel("Residual (AUD thousands)")
axes[1].set_title("Residuals vs. Predicted")

axes[2].hist(pct_error.clip(-50, 50), bins=60, color="steelblue", edgecolor="white")
axes[2].axvline(0, color="red", linewidth=0.8)
axes[2].set_xlabel("% Error (clipped ±50%)")
axes[2].set_title("Prediction Error Distribution")

plt.suptitle("XGBoost Residual Analysis", y=1.02, fontsize=13)
plt.tight_layout()
fig.savefig(FIGURES / "residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

within_10 = (pct_error.abs() <= 10).mean() * 100
within_20 = (pct_error.abs() <= 20).mean() * 100
print(f"  saved.  Within 10%: {within_10:.1f}%  |  Within 20%: {within_20:.1f}%")


# ── 6. SHAP figures ──────────────────────────────────────────────────────────
print("[6a/6] SHAP values (2000 samples) ...")

SHAP_SAMPLE = 2000
explainer   = shap.TreeExplainer(xgb_model)
X_shap      = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer(X_shap)
print(f"  computed for {len(X_shap):,} samples.")

print("[6b/6] shap_beeswarm.png ...")
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
plt.title("SHAP Beeswarm - XGBoost (top 10 features)")
plt.tight_layout()
fig.savefig(FIGURES / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")

print("[6c/6] shap_waterfall.png ...")
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Waterfall - Single Property (median-price example)")
plt.tight_layout()
fig = plt.gcf()
fig.savefig(FIGURES / "shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")


# ── 7. learning_curves.png ───────────────────────────────────────────────────
print("[7/6] learning_curves.png (slow - training on subsets) ...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, model, name in [
    (axes[0], RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    ), "Random Forest"),
    (axes[1], xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0,
    ), "XGBoost"),
]:
    print(f"  learning_curve: {name} ...")
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=3, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 8),
    )
    train_mean = -train_scores.mean(axis=1)
    val_mean   = -val_scores.mean(axis=1)

    ax.plot(train_sizes, train_mean, label="Train RMSE", color="steelblue")
    ax.plot(train_sizes, val_mean,   label="Val RMSE",   color="coral")
    ax.fill_between(train_sizes,
                    train_mean - train_scores.std(axis=1),
                    train_mean + train_scores.std(axis=1),
                    alpha=0.1, color="steelblue")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE (log-space)")
    ax.set_title(f"Learning Curve - {name}")
    ax.legend()

plt.tight_layout()
fig.savefig(FIGURES / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved.")

print("\nAll figures regenerated.")

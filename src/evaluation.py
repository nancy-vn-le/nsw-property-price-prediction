"""Metrics and comparison table utilities for model evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """
    Compute MSE, RMSE, MAE, and R² for a set of predictions.

    Parameters
    ----------
    y_true : pd.Series
        Actual target values.
    y_pred : np.ndarray
        Model predictions.

    Returns
    -------
    dict
        Keys: "MSE", "RMSE", "MAE", "R2".
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a formatted model comparison DataFrame from per-model metric dicts.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are model names; values are dicts from compute_metrics().

    Returns
    -------
    pd.DataFrame
        Rows = models, columns = MSE / RMSE / MAE / R², sorted by RMSE ascending.

    Examples
    --------
    >>> results = {
    ...     "OLS": compute_metrics(y_test, ols_preds),
    ...     "Ridge": compute_metrics(y_test, ridge_preds),
    ... }
    >>> table = build_comparison_table(results)
    """
    df = pd.DataFrame(results).T
    df = df.sort_values("RMSE")
    df[["MSE", "RMSE", "MAE"]] = df[["MSE", "RMSE", "MAE"]].applymap(
        lambda x: f"${x:,.0f}"
    )
    df["R2"] = df["R2"].apply(lambda x: f"{x:.4f}")
    return df


def plot_comparison(results: dict[str, dict], save: bool = True) -> None:
    """
    Bar chart of RMSE by model; optionally save to reports/figures/.

    Parameters
    ----------
    results : dict[str, dict]
        Keys are model names; values are dicts from compute_metrics().
    save : bool, optional
        Whether to save the figure as model_comparison.png (default True).
    """
    models = list(results.keys())
    rmses = [results[m]["RMSE"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models, rmses, color="steelblue", edgecolor="white")
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in rmses], padding=5)
    ax.set_xlabel("RMSE (AUD)")
    ax.set_title("Model Comparison — Test Set RMSE")
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
        print(f"Saved → {FIGURES_DIR / 'model_comparison.png'}")
    plt.show()

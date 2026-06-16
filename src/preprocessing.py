"""Build the sklearn preprocessing pipeline for property features."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_STATE = 42


def make_pipeline(
    numerical_features: list[str],
    low_card_cat_features: list[str],
) -> Pipeline:
    """
    Build a preprocessing Pipeline for mixed numerical/categorical features.

    Numerical features are scaled with StandardScaler.
    Low-cardinality categoricals are one-hot encoded (drop='first' to avoid
    multicollinearity with linear models).

    Parameters
    ----------
    numerical_features : list[str]
        Column names for numerical inputs.
    low_card_cat_features : list[str]
        Column names for categorical inputs with low cardinality
        (e.g. property_type, bedrooms).

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline ready for .fit_transform() / .transform().
    """
    numeric_transformer = StandardScaler()

    # drop='first' reduces one dummy variable per feature - avoids the dummy
    # variable trap that inflates OLS standard errors.
    categorical_transformer = OneHotEncoder(
        drop="first", handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, low_card_cat_features),
        ],
        remainder="drop",
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])


def split_and_save(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train/test sets and save them to data/processed/.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    target_col : str
        Name of the target column (e.g. "sale_price").
    feature_cols : list[str]
        Column names to use as model inputs.
    test_size : float, optional
        Fraction of data to hold out for testing (default 0.2).

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of DataFrames / Series
    """
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train.to_frame().to_parquet(PROCESSED_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(PROCESSED_DIR / "y_test.parquet")

    print(
        f"Saved splits → train: {len(X_train):,} rows, test: {len(X_test):,} rows"
    )
    return X_train, X_test, y_train, y_test

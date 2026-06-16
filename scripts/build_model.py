"""
Build lightweight model artifacts for the Streamlit app.

Run once after executing notebooks 01-03:
    python scripts/build_model.py

Reads from:
    data/raw/property_sales.csv          (raw dataset)
    data/processed/X_train.parquet       (notebook 02 output)
    data/processed/y_train.parquet
    data/processed/X_test.parquet
    data/processed/y_test.parquet
    data/processed/pipeline.pkl
    data/processed/suburb_encoding.parquet

Writes to models/:
    xgb_model.json              compact XGBoost model (~3-5 MB)
    pipeline.pkl                fitted StandardScaler pipeline
    suburb_encoding.parquet     suburb -> mean log-price map
    suburb_annual_stats.parquet suburb x year median prices (for trend charts)
    nsw_annual_stats.parquet    NSW-wide annual median
    suburb_summary.parquet      suburb overall median + total sales
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw
from src.evaluation import compute_metrics

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Step 0: verify inputs ─────────────────────────────────────────────────────

def check_inputs():
    required = [
        PROCESSED_DIR / "X_train.parquet",
        PROCESSED_DIR / "y_train.parquet",
        PROCESSED_DIR / "X_test.parquet",
        PROCESSED_DIR / "y_test.parquet",
        PROCESSED_DIR / "pipeline.pkl",
        PROCESSED_DIR / "suburb_encoding.parquet",
        ROOT / "data" / "raw" / "property_sales.csv",
    ]
    missing = [str(f.relative_to(ROOT)) for f in required if not f.exists()]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"  {f}")
        print("\nRun notebooks 01 → 02 first, then re-run this script.")
        sys.exit(1)


# ── Step 1: train compact XGBoost ────────────────────────────────────────────

def train_model():
    print("Loading processed features...")
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test  = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet").squeeze()
    y_test  = pd.read_parquet(PROCESSED_DIR / "y_test.parquet").squeeze()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    print("Training XGBoost (200 rounds)...")
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds   = model.predict(X_test)
    metrics = compute_metrics(np.expm1(y_test), np.expm1(preds))
    print(f"  App model  RMSE: ${metrics['RMSE']:,.0f}  R2: {metrics['R2']:.3f}")

    out = str(MODELS_DIR / "xgb_model.json")
    model.save_model(out)
    size_mb = Path(out).stat().st_size / 1e6
    print(f"  Saved models/xgb_model.json  ({size_mb:.1f} MB)")


# ── Step 2: copy pipeline and encoding ───────────────────────────────────────

def copy_artifacts():
    for fname in ["pipeline.pkl", "suburb_encoding.parquet"]:
        shutil.copy(PROCESSED_DIR / fname, MODELS_DIR / fname)
        print(f"  Copied models/{fname}")


# ── Step 3: build suburb statistics from raw data ────────────────────────────

def build_suburb_stats():
    print("Loading raw data (this takes ~1 min)...")
    df = load_raw("property_sales.csv")

    # Apply the same cleaning as notebook 02
    df = df[df["nature_of_property"].str.upper().str.startswith("R")].copy()
    df["contract_date"] = pd.to_datetime(df["contract_date"], errors="coerce")
    df = df[df["contract_date"] >= "2010-01-01"]
    df = df[df["purchase_price"].between(50_000, 30_000_000)]
    df["year"]   = df["contract_date"].dt.year
    df["suburb"] = df["suburb"].str.upper().str.strip()
    print(f"  {len(df):,} rows after cleaning")

    # Suburb × year: median price and sale count
    suburb_annual = (
        df.groupby(["suburb", "year"])["purchase_price"]
        .agg(median_price="median", n_sales="count")
        .reset_index()
    )
    suburb_annual.to_parquet(MODELS_DIR / "suburb_annual_stats.parquet", index=False)
    print(f"  Saved models/suburb_annual_stats.parquet  ({len(suburb_annual):,} rows)")

    # NSW-wide annual median
    nsw_annual = (
        df.groupby("year")["purchase_price"]
        .median()
        .reset_index()
        .rename(columns={"purchase_price": "median_price"})
    )
    nsw_annual.to_parquet(MODELS_DIR / "nsw_annual_stats.parquet", index=False)
    print(f"  Saved models/nsw_annual_stats.parquet  ({len(nsw_annual)} rows)")

    # Suburb overall median and total sales (all years)
    suburb_summary = (
        df.groupby("suburb")["purchase_price"]
        .agg(median_price="median", n_sales="count")
        .reset_index()
    )
    suburb_summary.to_parquet(MODELS_DIR / "suburb_summary.parquet", index=False)
    print(f"  Saved models/suburb_summary.parquet  ({len(suburb_summary):,} suburbs)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  NSW Property Price - Building App Artifacts")
    print("=" * 52)

    check_inputs()

    print("\n[1/3] Train model")
    train_model()

    print("\n[2/3] Copy pipeline and encoding")
    copy_artifacts()

    print("\n[3/3] Build suburb statistics")
    build_suburb_stats()

    print("\nDone. Start the app:")
    print("  streamlit run app.py")

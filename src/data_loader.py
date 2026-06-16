"""Load and perform basic validation on the raw property sales dataset."""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Maps original NSW Land Registry CSV headers → clean snake_case names.
# Raw file is never modified; renaming happens immediately on load.
COLUMN_RENAME = {
    "Property ID":               "property_id",
    "Sale counter":              "sale_counter",
    "Download date / time":      "download_date",
    "Property name":             "property_name",
    "Property unit number":      "unit_number",
    "Property house number":     "house_number",
    "Property street name":      "street_name",
    "Property locality":         "suburb",
    "Property post code":        "postcode",
    "Area":                      "area",
    "Area type":                 "area_type",
    "Contract date":             "contract_date",
    "Settlement date":           "settlement_date",
    "Purchase price":            "purchase_price",
    "Zoning":                    "zoning",
    "Nature of property":        "nature_of_property",
    "Primary purpose":           "primary_purpose",
    "Strata lot number":         "strata_lot_number",
    "Dealing number":            "dealing_number",
    "Property legal description": "legal_description",
}


def load_raw(filename: str) -> pd.DataFrame:
    """
    Load a CSV from data/raw/, rename columns to snake_case, and return.

    The raw file is never modified. Column renaming is the first and only
    transformation applied here - all other cleaning happens downstream.

    Parameters
    ----------
    filename : str
        Name of the CSV file inside data/raw/ (e.g. "property_sales.csv").

    Returns
    -------
    pd.DataFrame
        DataFrame with snake_case column names, no other transformations.

    Raises
    ------
    FileNotFoundError
        If the file does not exist. See README for download instructions.
    """
    filepath = RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"{filepath} not found. "
            "Download the dataset following the instructions in README.md."
        )
    df = pd.read_csv(filepath, low_memory=False)
    return df.rename(columns=COLUMN_RENAME)


def summarise(df: pd.DataFrame) -> None:
    """
    Print a quick summary of a DataFrame: shape, dtypes, missing counts.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarise.
    """
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts().to_string()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("\nNo missing values.")
    else:
        pct = (missing / len(df) * 100).round(1)
        summary = pd.DataFrame({"missing": missing, "pct": pct})
        print(f"\nMissing values:\n{summary.to_string()}")

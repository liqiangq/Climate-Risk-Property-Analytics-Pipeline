"""Data loading and cleaning helpers for Napier property transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CANDIDATES = (
    PROJECT_ROOT / "Empty_Land_Cleaned.csv",
    PROJECT_ROOT / "emptyt-land-cleaned.csv",
    PROJECT_ROOT / "empty-land-cleaned.csv",
    PROJECT_ROOT / "data" / "Empty_Land_Cleaned.csv",
)

CURRENCY_COLUMNS = ("Total$", "Land$", "Capital$")
NUMERIC_COLUMNS = (
    "Number",
    "Bathrooms",
    "Age",
    "Land Area m2",
    "Flrm2",
    "Brms",
    "Pirimai",
    "Onekawa",
    "Marewa",
    "Hospital Hill",
    "Bluff Hill",
    "flooding_status",
    "flooding_zone",
)


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Return an existing CSV path, accepting the common filename variants."""
    if path is not None:
        requested = Path(path)
        if requested.exists():
            return requested
        if not requested.is_absolute() and (PROJECT_ROOT / requested).exists():
            return PROJECT_ROOT / requested
        raise FileNotFoundError(f"Data file not found: {path}")

    for candidate in DEFAULT_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    candidates = ", ".join(str(item) for item in DEFAULT_DATA_CANDIDATES)
    raise FileNotFoundError(f"No cleaned property CSV found. Checked: {candidates}")


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned CSV used by the notebook."""
    return pd.read_csv(resolve_data_path(path))


def parse_money(series: pd.Series) -> pd.Series:
    """Convert values like '$425,000' into floats."""
    cleaned = series.astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "-": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def parse_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce the requested columns to numeric values when present."""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def clean_property_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and add the year/month fields used throughout the notebook."""
    result = df.copy()
    result.columns = [column.strip() for column in result.columns]

    if "Sale.Date" in result.columns:
        result["Sale.Date"] = pd.to_datetime(
            result["Sale.Date"], format="%d/%m/%Y", errors="coerce"
        )

    for column in CURRENCY_COLUMNS:
        if column in result.columns:
            result[column] = parse_money(result[column])

    result = parse_numeric_columns(result, NUMERIC_COLUMNS)

    if "Sale.Date" in result.columns:
        result["Year"] = result["Sale.Date"].dt.year
        result["Month"] = result["Sale.Date"].dt.month
        result["YearMonth"] = result["Sale.Date"].dt.to_period("M").astype("string")

    if "Total$" in result.columns:
        result.loc[result["Total$"] <= 0, "Total$"] = np.nan

    return result


def load_clean_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the default property dataset."""
    return clean_property_data(load_data(path))


def dataset_summary(df: pd.DataFrame) -> dict:
    """Build a compact JSON-friendly summary for the API dashboard."""
    summary: dict[str, object] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
    }

    if "Sale.Date" in df.columns:
        valid_dates = df["Sale.Date"].dropna()
        summary["date_min"] = valid_dates.min().date().isoformat() if not valid_dates.empty else None
        summary["date_max"] = valid_dates.max().date().isoformat() if not valid_dates.empty else None

    if "Total$" in df.columns:
        prices = df["Total$"].dropna()
        summary["average_total_price"] = float(prices.mean()) if not prices.empty else None
        summary["median_total_price"] = float(prices.median()) if not prices.empty else None

    for column in ("flooding_status", "flooding_zone"):
        if column in df.columns:
            counts = df[column].dropna().astype(int).value_counts().sort_index()
            summary[f"{column}_counts"] = {str(key): int(value) for key, value in counts.items()}

    return summary


def save_cleaned_data(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a cleaned dataframe to CSV."""
    target = Path(path)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target

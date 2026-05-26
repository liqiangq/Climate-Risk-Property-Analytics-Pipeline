"""Feature engineering and aggregation functions from the notebook workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_cleaning import clean_property_data


BASE_MODEL_COLUMNS = (
    "Ln_Total",
    "flooding_status",
    "flooding_zone",
    "Flrm2",
    "Bathrooms",
    "Brms",
    "Age",
)


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-price fields used by the DiD and Ridge models."""
    result = df.copy()
    if "Total$" in result.columns:
        result["Ln_Total"] = np.log(result["Total$"].where(result["Total$"] > 0))
    return result


def add_time_features(df: pd.DataFrame, flood_month: str = "2020-11") -> pd.DataFrame:
    """Add year, month, period, time trend, and post-flood indicator fields."""
    result = df.copy()
    if "Sale.Date" not in result.columns:
        return result

    result["Year"] = result["Sale.Date"].dt.year
    result["Month"] = result["Sale.Date"].dt.month
    result["YearMonth"] = result["Sale.Date"].dt.to_period("M")
    valid_years = result["Year"].dropna()
    min_year = int(valid_years.min()) if not valid_years.empty else 0
    result["time"] = result["Year"] - min_year
    result["post_flood"] = (result["YearMonth"].astype("string") >= flood_month).astype(int)
    return result


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction fields for a time-varying flood impact model."""
    result = df.copy()
    if {"flooding_status", "time"}.issubset(result.columns):
        result["flooding_status_time"] = result["flooding_status"] * result["time"]
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    result = clean_property_data(df)
    result = add_price_features(result)
    result = add_time_features(result)
    result = add_interaction_features(result)
    if "YearMonth" in result.columns:
        result["YearMonth"] = result["YearMonth"].astype("string")
    return result


def filter_last_n_years(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    """Keep records from the most recent n-year window available in the data."""
    if "Year" not in df.columns:
        return df.copy()
    valid_years = df["Year"].dropna()
    if valid_years.empty:
        return df.copy()
    cutoff = int(valid_years.max()) - years
    return df[df["Year"] >= cutoff].copy()


def prepare_model_frame(
    df: pd.DataFrame,
    last_n_years: int | None = 10,
    include_time: bool = False,
) -> pd.DataFrame:
    """Return a clean model-ready dataframe."""
    result = engineer_features(df)
    if last_n_years is not None:
        result = filter_last_n_years(result, years=last_n_years)

    columns = list(BASE_MODEL_COLUMNS)
    if include_time:
        columns.append("time")

    present = [column for column in columns if column in result.columns]
    result = result.dropna(subset=present).copy()
    for column in present:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=present)


def housing_price_index(df: pd.DataFrame, frequency: str = "yearly") -> pd.DataFrame:
    """Calculate yearly or monthly housing price index with the first period as 100."""
    result = engineer_features(df)
    period_column = "Year" if frequency == "yearly" else "YearMonth"
    if period_column not in result.columns:
        raise ValueError(f"Cannot build price index without {period_column}")

    index = (
        result.dropna(subset=[period_column, "Total$"])
        .groupby(period_column, as_index=False)["Total$"]
        .mean()
        .sort_values(period_column)
    )
    if index.empty:
        return pd.DataFrame(columns=[period_column, "AveragePrice", "PriceIndex"])

    base_price = float(index["Total$"].iloc[0])
    index = index.rename(columns={"Total$": "AveragePrice"})
    index["PriceIndex"] = (index["AveragePrice"] / base_price) * 100
    index[period_column] = index[period_column].astype(str)
    return index


def flood_zone_counts(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Count flood-zone records by year, or by month inside one year."""
    result = engineer_features(df)
    if year is not None:
        result = result[result["Year"] == year].copy()
        group_columns = ["Month", "flooding_zone"]
    else:
        group_columns = ["Year", "flooding_zone"]

    counts = (
        result.dropna(subset=group_columns)
        .groupby(group_columns)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(group_columns[0])
    )
    counts.columns = [str(column) for column in counts.columns]
    return counts


def monthly_price_trends_by_flood_status(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly average prices split by flooding status."""
    result = engineer_features(df)
    trends = (
        result.dropna(subset=["YearMonth", "flooding_status", "Total$"])
        .groupby(["YearMonth", "flooding_status"])["Total$"]
        .mean()
        .unstack()
        .reset_index()
        .sort_values("YearMonth")
    )
    trends.columns = [str(column) for column in trends.columns]
    return trends


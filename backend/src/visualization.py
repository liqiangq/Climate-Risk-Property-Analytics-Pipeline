"""Matplotlib visualizations for the property climate-risk dashboard."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path

MPL_CACHE_DIR = Path(__file__).resolve().parents[2] / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .feature_engineering import flood_zone_counts, housing_price_index
from .modelling import ModelResult


def figure_to_png_bytes(fig: plt.Figure) -> bytes:
    """Serialize a Matplotlib figure as PNG bytes."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def plot_price_index(df: pd.DataFrame, frequency: str = "monthly") -> plt.Figure:
    """Plot the housing price index, marking the November 2020 flood event."""
    price_index = housing_price_index(df, frequency=frequency)
    period_column = "Year" if frequency == "yearly" else "YearMonth"

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(price_index[period_column], price_index["PriceIndex"], marker="o", linewidth=2)
    if frequency == "monthly":
        ax.axvline("2020-11", color="#b91c1c", linestyle="--", label="Flood event")
    else:
        ax.axvline("2020", color="#b91c1c", linestyle="--", label="Flood event")
    ax.set_title("Housing Price Index")
    ax.set_xlabel(period_column)
    ax.set_ylabel("Index, first period = 100")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate(rotation=45)
    return fig


def plot_flood_zone_counts(df: pd.DataFrame, year: int | None = None) -> plt.Figure:
    """Plot flood-zone transaction counts by year or by month for one year."""
    counts = flood_zone_counts(df, year=year)
    period_column = "Month" if year is not None else "Year"
    value_columns = [column for column in counts.columns if column != period_column]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    counts.plot(x=period_column, y=value_columns, kind="bar", ax=ax)
    ax.set_title("Properties by Flooding Zone" if year is None else f"Properties by Flooding Zone in {year}")
    ax.set_xlabel(period_column)
    ax.set_ylabel("Number of Properties")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Flooding Zone")
    return fig


def plot_model_coefficients(result: ModelResult) -> plt.Figure:
    """Plot model coefficients, excluding intercept/constant terms."""
    coefficients = {
        key: value
        for key, value in result.coefficients.items()
        if key.lower() not in {"intercept", "const"}
    }
    items = pd.Series(coefficients).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    items.plot(kind="barh", ax=ax, color="#2563eb")
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(f"{result.model_type} Coefficients")
    ax.set_xlabel("Coefficient")
    ax.grid(True, axis="x", alpha=0.25)
    return fig

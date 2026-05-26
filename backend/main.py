"""FastAPI application for the climate-risk property analytics dashboard."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.src.data_cleaning import PROJECT_ROOT, dataset_summary, load_clean_data
from backend.src.feature_engineering import (
    flood_zone_counts,
    housing_price_index,
    monthly_price_trends_by_flood_status,
)
from backend.src.modelling import fit_did_model, fit_linear_price_model, fit_ridge_model
from backend.src.visualization import (
    figure_to_png_bytes,
    plot_flood_zone_counts,
    plot_model_coefficients,
    plot_price_index,
)


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Climate Risk Property Analytics API",
    version="0.1.0",
    description="FastAPI backend for Napier property flood-risk analytics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@lru_cache(maxsize=1)
def get_data() -> pd.DataFrame:
    """Cache the cleaned CSV for API calls."""
    return load_clean_data()


def records(df: pd.DataFrame) -> list[dict]:
    """Convert dataframe rows into JSON-safe dictionaries."""
    json_ready = df.copy()
    for column in json_ready.columns:
        if pd.api.types.is_datetime64_any_dtype(json_ready[column]):
            json_ready[column] = json_ready[column].dt.strftime("%Y-%m-%d")
    json_ready = json_ready.replace({np.nan: None})
    return json_ready.to_dict(orient="records")


def handle_model_error(error: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(error))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "data_rows": int(len(get_data()))}


@app.get("/api/summary")
def summary() -> dict:
    return dataset_summary(get_data())


@app.get("/api/records")
def sample_records(limit: int = Query(default=25, ge=1, le=200)) -> list[dict]:
    return records(get_data().head(limit))


@app.get("/api/price-index")
def price_index(frequency: str = Query(default="monthly", pattern="^(monthly|yearly)$")) -> list[dict]:
    frequency_arg = "yearly" if frequency == "yearly" else "monthly"
    return records(housing_price_index(get_data(), frequency=frequency_arg))


@app.get("/api/flood-zones")
def flood_zones(year: int | None = Query(default=None, ge=1900, le=2100)) -> list[dict]:
    return records(flood_zone_counts(get_data(), year=year))


@app.get("/api/price-trends")
def price_trends() -> list[dict]:
    return records(monthly_price_trends_by_flood_status(get_data()))


@app.get("/api/model/ols")
def ols_model() -> dict:
    try:
        return asdict(fit_linear_price_model(get_data()))
    except Exception as error:
        raise handle_model_error(error) from error


@app.get("/api/model/did")
def did_model(interaction: bool = False) -> dict:
    try:
        return asdict(fit_did_model(get_data(), include_interaction=interaction))
    except Exception as error:
        raise handle_model_error(error) from error


@app.get("/api/model/ridge")
def ridge_model(alpha: float = Query(default=1.0, gt=0.0, le=100.0)) -> dict:
    try:
        return asdict(fit_ridge_model(get_data(), alpha=alpha))
    except Exception as error:
        raise handle_model_error(error) from error


@app.get("/api/charts/price-index.png")
def price_index_chart(frequency: str = Query(default="monthly", pattern="^(monthly|yearly)$")) -> Response:
    fig = plot_price_index(get_data(), frequency=frequency)
    return Response(content=figure_to_png_bytes(fig), media_type="image/png")


@app.get("/api/charts/flood-zones.png")
def flood_zones_chart(year: int | None = Query(default=None, ge=1900, le=2100)) -> Response:
    fig = plot_flood_zone_counts(get_data(), year=year)
    return Response(content=figure_to_png_bytes(fig), media_type="image/png")


@app.get("/api/charts/model-coefficients.png")
def model_coefficients_chart(model: str = Query(default="did", pattern="^(did|ridge|ols)$")) -> Response:
    try:
        if model == "ridge":
            result = fit_ridge_model(get_data())
        elif model == "ols":
            result = fit_linear_price_model(get_data())
        else:
            result = fit_did_model(get_data())
        fig = plot_model_coefficients(result)
        return Response(content=figure_to_png_bytes(fig), media_type="image/png")
    except Exception as error:
        raise handle_model_error(error) from error

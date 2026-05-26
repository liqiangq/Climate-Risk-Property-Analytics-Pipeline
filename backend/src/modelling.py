"""Econometric and machine-learning models for property flood-risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_engineering import prepare_model_frame


OLS_FEATURES = (
    "Year",
    "Age",
    "Bathrooms",
    "Brms",
    "Pirimai",
    "Onekawa",
    "Marewa",
    "Hospital Hill",
    "Bluff Hill",
    "flooding_status",
    "flooding_zone",
)

DID_FORMULA = "Ln_Total ~ flooding_status + flooding_zone + Flrm2 + Bathrooms + Brms + Age"
DID_INTERACTION_FORMULA = (
    "Ln_Total ~ flooding_status * time + flooding_zone + Flrm2 + Bathrooms + Brms + Age"
)
RIDGE_FEATURES = ("flooding_status", "flooding_zone", "Bathrooms", "Flrm2", "Brms", "Age")


@dataclass(frozen=True)
class ModelResult:
    """JSON-friendly model result."""

    model_type: str
    formula: str | None
    rows: int
    metrics: dict[str, float]
    coefficients: dict[str, float]
    summary: str | None = None


def _float_dict(values: pd.Series | dict) -> dict[str, float]:
    return {str(key): float(value) for key, value in dict(values).items()}


def fit_linear_price_model(df: pd.DataFrame) -> ModelResult:
    """Fit the notebook's direct OLS model against Total$."""
    data = prepare_model_frame(df, last_n_years=None)
    if "Total$" not in data.columns:
        raise ValueError("Total$ column is required for the OLS price model.")

    features = [column for column in OLS_FEATURES if column in data.columns]
    data = data.dropna(subset=features + ["Total$"]).copy()
    if data.empty:
        raise ValueError("No rows available after cleaning for the OLS price model.")

    x = sm.add_constant(data[features])
    y = data["Total$"]
    model = sm.OLS(y, x).fit()
    return ModelResult(
        model_type="ols_total_price",
        formula="Total$ ~ " + " + ".join(features),
        rows=int(model.nobs),
        metrics={
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
        },
        coefficients=_float_dict(model.params),
        summary=str(model.summary()),
    )


def fit_did_model(
    df: pd.DataFrame,
    include_interaction: bool = False,
    last_n_years: int | None = 10,
) -> ModelResult:
    """Fit a log-price DiD-style OLS model."""
    data = prepare_model_frame(df, last_n_years=last_n_years, include_time=include_interaction)
    formula = DID_INTERACTION_FORMULA if include_interaction else DID_FORMULA
    if data.empty:
        raise ValueError("No rows available after cleaning for the DiD model.")

    model = smf.ols(formula=formula, data=data).fit()
    return ModelResult(
        model_type="did_ols_log_price",
        formula=formula,
        rows=int(model.nobs),
        metrics={
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "aic": float(model.aic),
            "bic": float(model.bic),
        },
        coefficients=_float_dict(model.params),
        summary=str(model.summary()),
    )


def fit_ridge_model(
    df: pd.DataFrame,
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    last_n_years: int | None = 10,
) -> ModelResult:
    """Fit a standardized Ridge regression model for log total price."""
    data = prepare_model_frame(df, last_n_years=last_n_years)
    features = [column for column in RIDGE_FEATURES if column in data.columns]
    data = data.dropna(subset=features + ["Ln_Total"]).copy()
    if len(data) < 5:
        raise ValueError("At least 5 clean rows are required for Ridge regression.")

    x = data[features]
    y = data["Ln_Total"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)
    predictions = model.predict(x_test_scaled)

    coefficients = {feature: float(value) for feature, value in zip(features, model.coef_)}
    coefficients["intercept"] = float(model.intercept_)
    return ModelResult(
        model_type="ridge_log_price",
        formula="Ln_Total ~ " + " + ".join(features),
        rows=int(len(data)),
        metrics={
            "r_squared_test": float(r2_score(y_test, predictions)),
            "rmse_test": float(mean_squared_error(y_test, predictions) ** 0.5),
            "alpha": float(alpha),
        },
        coefficients=coefficients,
        summary=None,
    )


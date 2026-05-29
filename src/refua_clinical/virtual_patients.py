"""Virtual patient generation using Gaussian copulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist  # type: ignore[import-untyped]
from scipy.stats import norm

from .models import CovariateSpec, VirtualPopulationSpec


@dataclass(slots=True)
class VirtualPopulation:
    table: pd.DataFrame
    correlation: np.ndarray


def generate_virtual_population(
    spec: VirtualPopulationSpec, *, seed: int
) -> VirtualPopulation:
    if spec.size < 1:
        raise ValueError("population.size must be >= 1")
    if not spec.covariates:
        raise ValueError("population.covariates must be non-empty")

    rng = np.random.default_rng(seed)
    dims = len(spec.covariates)
    correlation = _resolve_correlation(spec, dims)

    latent = rng.multivariate_normal(np.zeros(dims), correlation, size=spec.size)
    uniform = norm.cdf(latent)

    columns: dict[str, np.ndarray] = {}
    for idx, covariate in enumerate(spec.covariates):
        columns[covariate.name] = _sample_marginal(covariate, uniform[:, idx], rng)

    frame = pd.DataFrame(columns)
    frame["patient_id"] = np.arange(1, spec.size + 1)
    ordered = ["patient_id", *[cov.name for cov in spec.covariates]]
    return VirtualPopulation(table=frame[ordered], correlation=correlation)


def infer_population_spec_from_dataframe(
    frame: pd.DataFrame,
    *,
    size: int,
    columns: list[str] | None = None,
) -> VirtualPopulationSpec:
    if size < 1:
        raise ValueError("size must be >= 1")

    if columns is None:
        numeric = [
            name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])
        ]
        columns = numeric[:4]

    if not columns:
        raise ValueError(
            "No usable numeric columns found for virtual patient inference"
        )

    covariates: list[CovariateSpec] = []
    encoded_columns: dict[str, pd.Series[Any]] = {}
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().astype(float)
            if values.empty:
                raise ValueError(f"Column '{column}' has no non-missing values")
            array_values = values.to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            minimum = float(values.min())
            maximum = float(values.max())
            if sd <= 1e-12:
                skewness = 0.0
            else:
                centered = array_values - np.mean(array_values)
                skewness = float(np.mean((centered / sd) ** 3))

            if minimum > 0 and sd > 0 and skewness > 0.8:
                cv = float(sd / max(mean, 1e-6))
                covariates.append(
                    CovariateSpec(
                        name=column,
                        distribution="lognormal",
                        params={
                            "mean": mean,
                            "cv": max(cv, 0.05),
                            "min": minimum,
                            "max": maximum,
                        },
                    )
                )
            else:
                covariates.append(
                    CovariateSpec(
                        name=column,
                        distribution="normal",
                        params={
                            "mean": mean,
                            "sd": max(sd, 1e-6),
                            "min": minimum,
                            "max": maximum,
                        },
                    )
                )
            encoded_columns[column] = series.fillna(mean).astype(float)
        else:
            string_values = series.dropna().astype(str).str.strip()
            string_values = string_values[string_values != ""]
            if string_values.empty:
                raise ValueError(f"Column '{column}' has no usable categorical values")
            distribution = string_values.value_counts(normalize=True)
            covariates.append(
                CovariateSpec(
                    name=column,
                    distribution="categorical",
                    params={
                        "categories": distribution.index.tolist(),
                        "probs": distribution.to_numpy(dtype=float).tolist(),
                    },
                ),
            )
            encoded, _ = pd.factorize(series.fillna("__missing__").astype(str))
            encoded_columns[column] = pd.Series(
                encoded.astype(float), index=series.index
            )

        missing_rate = float(series.isna().mean())
        if 0.0 < missing_rate < 1.0:
            indicator_name = f"{column}_missing"
            covariates.append(
                CovariateSpec(
                    name=indicator_name,
                    distribution="categorical",
                    params={
                        "categories": [0, 1],
                        "probs": [1.0 - missing_rate, missing_rate],
                    },
                )
            )
            encoded_columns[indicator_name] = series.isna().astype(float)

    encoded_names = [cov.name for cov in covariates]
    encoded_frame = pd.DataFrame(
        {name: encoded_columns[name] for name in encoded_names}
    )
    correlation = encoded_frame.corr(numeric_only=True).to_numpy(dtype=float)
    return VirtualPopulationSpec(
        size=size, covariates=covariates, correlation=correlation.tolist()
    )


def _resolve_correlation(spec: VirtualPopulationSpec, dims: int) -> np.ndarray:
    if spec.correlation is None:
        return np.eye(dims)

    matrix = np.array(spec.correlation, dtype=float)
    if matrix.shape != (dims, dims):
        raise ValueError(
            f"population.correlation must be {dims}x{dims}, got {matrix.shape[0]}x{matrix.shape[1]}"
        )

    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)

    eigvals, eigvecs = np.linalg.eigh(matrix)
    clipped = np.clip(eigvals, 1e-6, None)
    repaired = (eigvecs @ np.diag(clipped) @ eigvecs.T).real
    d = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(d, d)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _sample_marginal(
    spec: CovariateSpec, u: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    params = spec.params
    name = spec.distribution

    if name == "normal":
        mean = float(params.get("mean", 0.0))
        sd = float(params.get("sd", 1.0))
        values = norm.ppf(u, loc=mean, scale=max(sd, 1e-6))
    elif name == "lognormal":
        mean = float(params.get("mean", 1.0))
        cv = float(params.get("cv", 0.25))
        sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
        mu = float(np.log(max(mean, 1e-6)) - 0.5 * sigma * sigma)
        values = np.exp(norm.ppf(u, loc=mu, scale=sigma))
    elif name == "beta":
        alpha = float(params.get("alpha", 2.0))
        beta_value = float(params.get("beta", 5.0))
        values = beta_dist.ppf(u, alpha, beta_value)
    elif name == "uniform":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        values = low + (high - low) * u
    elif name == "categorical":
        categories = params.get("categories", ["A", "B"])
        probs = np.array(params.get("probs", [0.5, 0.5]), dtype=float)
        probs = probs / probs.sum()
        thresholds = np.cumsum(probs)
        idx = np.searchsorted(thresholds, u, side="right")
        idx = np.clip(idx, 0, len(categories) - 1)
        return np.array([categories[int(i)] for i in idx], dtype=object)
    else:
        raise ValueError(f"Unsupported covariate distribution: {name}")

    min_value = params.get("min")
    max_value = params.get("max")
    if isinstance(min_value, int | float):
        values = np.maximum(values, float(min_value))
    if isinstance(max_value, int | float):
        values = np.minimum(values, float(max_value))

    if np.any(~np.isfinite(values)):
        repaired = np.asarray(values, dtype=float)
        bad = ~np.isfinite(repaired)
        repaired[bad] = rng.normal(
            loc=np.nanmean(repaired[~bad]), scale=1.0, size=int(np.sum(bad))
        )
        values = repaired

    return np.asarray(values)


def summarize_covariates(table: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(table), "covariates": {}}
    for column in table.columns:
        if column == "patient_id":
            continue
        series = table[column]
        if pd.api.types.is_numeric_dtype(series):
            summary["covariates"][column] = {
                "mean": float(series.mean()),
                "sd": float(series.std(ddof=1)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        else:
            counts = series.value_counts(normalize=True).to_dict()
            summary["covariates"][column] = {"distribution": counts}
    return summary

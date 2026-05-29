"""Transportability diagnostics between trial and target populations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CovariateSmd = dict[str, float | str]


def assess_transportability(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    method: str = "none",
) -> dict[str, Any]:
    if columns is None:
        columns = [
            name
            for name in reference.columns
            if name in target.columns and pd.api.types.is_numeric_dtype(reference[name])
        ][:8]

    if not columns:
        raise ValueError(
            "No common numeric columns available for transportability assessment"
        )

    smd_items: list[CovariateSmd] = []
    for column in columns:
        ref = reference[column].dropna().to_numpy(dtype=float)
        tgt = target[column].dropna().to_numpy(dtype=float)
        if ref.size < 2 or tgt.size < 2:
            continue

        pooled_sd = np.sqrt(0.5 * (np.var(ref, ddof=1) + np.var(tgt, ddof=1)))
        pooled_sd = max(float(pooled_sd), 1e-6)
        smd = float((np.mean(tgt) - np.mean(ref)) / pooled_sd)
        smd_items.append(
            {
                "covariate": column,
                "reference_mean": float(np.mean(ref)),
                "target_mean": float(np.mean(tgt)),
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )

    if not smd_items:
        raise ValueError(
            "Could not compute transportability metrics on selected columns"
        )

    abs_smd = np.array([item["abs_smd"] for item in smd_items], dtype=float)
    overlap_score = float(np.mean(np.exp(-0.5 * abs_smd**2)))

    risk_level = "low"
    if float(np.max(abs_smd)) > 0.5 or overlap_score < 0.78:
        risk_level = "high"
    elif float(np.max(abs_smd)) > 0.25 or overlap_score < 0.90:
        risk_level = "moderate"

    recommendation = _transportability_recommendation(risk_level)

    normalized_method = str(method).strip().lower()
    weighting: dict[str, Any] | None = None
    if normalized_method != "none":
        weights = _estimate_reference_weights(
            reference[columns].astype(float),
            target[columns].astype(float),
            method=normalized_method,
        )
        weighted_smd = _weighted_smd_table(
            reference[columns].astype(float),
            target[columns].astype(float),
            columns=columns,
            weights=weights,
        )
        weighted_abs_smd = np.array(
            [float(item["abs_smd"]) for item in weighted_smd],
            dtype=float,
        )
        weighted_overlap = float(np.mean(np.exp(-0.5 * weighted_abs_smd**2)))
        weighting = {
            "method": normalized_method,
            "effective_sample_size": float(
                (np.sum(weights) ** 2) / np.clip(np.sum(weights**2), 1e-9, None)
            ),
            "weight_summary": {
                "min": float(np.min(weights)),
                "p10": float(np.quantile(weights, 0.10)),
                "median": float(np.quantile(weights, 0.50)),
                "p90": float(np.quantile(weights, 0.90)),
                "max": float(np.max(weights)),
            },
            "weighted_overlap_score": weighted_overlap,
            "weighted_max_abs_smd": float(np.max(weighted_abs_smd)),
            "weighted_mean_abs_smd": float(np.mean(weighted_abs_smd)),
            "weighted_covariate_smd": weighted_smd,
        }

    payload = {
        "columns": columns,
        "max_abs_smd": float(np.max(abs_smd)),
        "mean_abs_smd": float(np.mean(abs_smd)),
        "overlap_score": overlap_score,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "covariate_smd": sorted(
            smd_items,
            key=lambda item: float(item["abs_smd"]),
            reverse=True,
        ),
    }
    if weighting is not None:
        payload["weighting"] = weighting
    return payload


def load_tabular(path: str) -> pd.DataFrame:
    lowered = path.lower()
    if lowered.endswith(".parquet"):
        return pd.read_parquet(path)
    if lowered.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def transportability_to_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Transportability Assessment", ""]
    lines.append(f"- Risk level: {payload['risk_level']}")
    lines.append(f"- Overlap score: {float(payload['overlap_score']):.3f}")
    lines.append(f"- Max abs SMD: {float(payload['max_abs_smd']):.3f}")
    lines.append(f"- Recommendation: {payload['recommendation']}")
    lines.append("")
    weighting = payload.get("weighting")
    if isinstance(weighting, dict):
        lines.append("## Weighting Repair")
        lines.append(f"- Method: {weighting['method']}")
        lines.append(
            f"- Effective sample size: {float(weighting['effective_sample_size']):.1f}"
        )
        lines.append(
            f"- Weighted overlap score: {float(weighting['weighted_overlap_score']):.3f}"
        )
        lines.append(
            f"- Weighted max abs SMD: {float(weighting['weighted_max_abs_smd']):.3f}"
        )
        lines.append("")
    lines.append("## Covariate Shift")
    for item in payload.get("covariate_smd", [])[:10]:
        lines.append(
            f"- {item['covariate']}: abs_smd={float(item['abs_smd']):.3f}, "
            f"ref={float(item['reference_mean']):.3f}, target={float(item['target_mean']):.3f}"
        )
    if isinstance(weighting, dict):
        lines.append("")
        lines.append("## Weighted Covariate Shift")
        for item in weighting.get("weighted_covariate_smd", [])[:10]:
            lines.append(
                f"- {item['covariate']}: abs_smd={float(item['abs_smd']):.3f}, "
                f"weighted_ref={float(item['reference_mean']):.3f}, "
                f"target={float(item['target_mean']):.3f}"
            )
    return "\n".join(lines) + "\n"


def _transportability_recommendation(risk_level: str) -> str:
    if risk_level == "high":
        return (
            "Use covariate reweighting/transport models, add calibration cohorts, and run "
            "sensitivity analyses before extrapolating decisions."
        )
    if risk_level == "moderate":
        return (
            "Apply weighted analyses and verify key subgroup consistency in "
            "scenario simulations."
        )
    return "Current covariate shift appears manageable for cautious transportability assumptions."


def _estimate_reference_weights(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    method: str,
) -> np.ndarray:
    if method == "ps_weighted":
        return _propensity_weights(reference, target)
    if method == "entropy_balanced":
        return _entropy_balancing_weights(reference, target)
    raise ValueError(
        "transportability method must be 'none', 'ps_weighted', or 'entropy_balanced'"
    )


def _propensity_weights(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    x_ref = reference.to_numpy(dtype=float)
    x_tgt = target.to_numpy(dtype=float)
    x = np.vstack([x_ref, x_tgt])
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    sd = np.where(sd > 1e-6, sd, 1.0)
    x_scaled = (x - mean) / sd
    y = np.concatenate(
        [np.zeros(len(x_ref), dtype=float), np.ones(len(x_tgt), dtype=float)]
    )

    def objective(params: np.ndarray) -> float:
        intercept = params[0]
        coef = params[1:]
        logits = intercept + x_scaled @ coef
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
        return float(loss + 0.01 * np.sum(coef**2))

    result = minimize(
        objective,
        x0=np.zeros(x_scaled.shape[1] + 1, dtype=float),
        method="L-BFGS-B",
    )
    params = (
        result.x if result.success else np.zeros(x_scaled.shape[1] + 1, dtype=float)
    )
    logits_ref = params[0] + ((x_ref - mean) / sd) @ params[1:]
    probs_ref = 1.0 / (1.0 + np.exp(-np.clip(logits_ref, -30.0, 30.0)))
    probs_ref = np.clip(probs_ref, 1e-4, 1.0 - 1e-4)
    odds = probs_ref / (1.0 - probs_ref)
    return _normalize_weights(odds, target_n=len(target))


def _entropy_balancing_weights(
    reference: pd.DataFrame,
    target: pd.DataFrame,
) -> np.ndarray:
    x_ref = reference.to_numpy(dtype=float)
    x_tgt = target.to_numpy(dtype=float)
    mean = np.mean(x_ref, axis=0)
    sd = np.std(x_ref, axis=0, ddof=1)
    sd = np.where(sd > 1e-6, sd, 1.0)
    ref_scaled = (x_ref - mean) / sd
    target_mean = np.mean((x_tgt - mean) / sd, axis=0)

    def objective(beta: np.ndarray) -> float:
        raw = ref_scaled @ beta
        weights = np.exp(np.clip(raw, -20.0, 20.0))
        weights = _normalize_weights(weights, target_n=len(target))
        weighted_mean = np.average(ref_scaled, axis=0, weights=weights)
        mismatch = weighted_mean - target_mean
        entropy = np.mean(weights * np.log(np.clip(weights, 1e-8, None)))
        return float(np.sum(mismatch**2) + 1e-4 * entropy + 1e-4 * np.sum(beta**2))

    result = minimize(
        objective,
        x0=np.zeros(ref_scaled.shape[1], dtype=float),
        method="L-BFGS-B",
    )
    beta = result.x if result.success else np.zeros(ref_scaled.shape[1], dtype=float)
    weights = np.exp(np.clip(ref_scaled @ beta, -20.0, 20.0))
    return _normalize_weights(weights, target_n=len(target))


def _normalize_weights(weights: np.ndarray, *, target_n: int) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-6, None)
    return clipped * (max(float(target_n), 1.0) / np.sum(clipped))


def _weighted_smd_table(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    columns: list[str],
    weights: np.ndarray,
) -> list[CovariateSmd]:
    items: list[CovariateSmd] = []
    for column in columns:
        ref = reference[column].to_numpy(dtype=float)
        tgt = target[column].to_numpy(dtype=float)
        ref_mean = float(np.average(ref, weights=weights))
        tgt_mean = float(np.mean(tgt))
        ref_var = float(np.average((ref - ref_mean) ** 2, weights=weights))
        tgt_var = float(np.var(tgt, ddof=1))
        pooled_sd = max(float(np.sqrt(0.5 * (ref_var + tgt_var))), 1e-6)
        smd = float((tgt_mean - ref_mean) / pooled_sd)
        items.append(
            {
                "covariate": column,
                "reference_mean": ref_mean,
                "target_mean": tgt_mean,
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )
    return sorted(items, key=lambda item: float(item["abs_smd"]), reverse=True)

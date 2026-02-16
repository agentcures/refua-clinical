"""Transportability diagnostics between trial and target populations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CovariateSmd = dict[str, float | str]


def assess_transportability(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    if columns is None:
        columns = [
            name
            for name in reference.columns
            if name in target.columns and pd.api.types.is_numeric_dtype(reference[name])
        ][:8]

    if not columns:
        raise ValueError("No common numeric columns available for transportability assessment")

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
        raise ValueError("Could not compute transportability metrics on selected columns")

    abs_smd = np.array([item["abs_smd"] for item in smd_items], dtype=float)
    overlap_score = float(np.mean(np.exp(-0.5 * abs_smd**2)))

    risk_level = "low"
    if float(np.max(abs_smd)) > 0.5 or overlap_score < 0.78:
        risk_level = "high"
    elif float(np.max(abs_smd)) > 0.25 or overlap_score < 0.90:
        risk_level = "moderate"

    recommendation = _transportability_recommendation(risk_level)

    return {
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
    lines.append("## Covariate Shift")
    for item in payload.get("covariate_smd", [])[:10]:
        lines.append(
            f"- {item['covariate']}: abs_smd={float(item['abs_smd']):.3f}, "
            f"ref={float(item['reference_mean']):.3f}, target={float(item['target_mean']):.3f}"
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
            "Apply weighted analyses and verify key subgroup consistency in scenario simulations."
        )
    return "Current covariate shift appears manageable for cautious transportability assumptions."

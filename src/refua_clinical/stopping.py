"""Adaptive stopping rules and alpha spending utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-untyped]

from .models import ExternalControlSpec, StoppingSpec
from .pkpd import two_arm_test


def alpha_spent(*, information_fraction: float, alpha: float, method: str) -> float:
    info = float(np.clip(information_fraction, 1e-6, 1.0))
    alpha = float(np.clip(alpha, 1e-8, 0.5))

    if method == "obrien_fleming":
        z_alpha = norm.ppf(1.0 - alpha)
        spent = 1.0 - norm.cdf(z_alpha / np.sqrt(info))
        return float(np.clip(spent, 0.0, alpha))

    if method == "pocock":
        spent = alpha * np.log(1.0 + (np.e - 1.0) * info)
        return float(np.clip(spent, 0.0, alpha))

    return float(np.clip(alpha * info, 0.0, alpha))


def evaluate_interim_decision(
    analysis_frame: pd.DataFrame,
    *,
    control_arm_id: str,
    treatment_ids: list[str],
    enrolled_n: int,
    total_n: int,
    interim_index: int,
    stopping: StoppingSpec,
    external_control: ExternalControlSpec,
    rng: np.random.Generator,
) -> dict[str, Any]:
    info_fraction = float(np.clip(enrolled_n / max(total_n, 1), 0.0, 1.0))
    spent_alpha = alpha_spent(
        information_fraction=info_fraction,
        alpha=stopping.one_sided_alpha,
        method=stopping.alpha_spending,
    )

    if analysis_frame.empty:
        return {
            "interim_index": interim_index,
            "enrolled_n": enrolled_n,
            "information_fraction": info_fraction,
            "spent_alpha": spent_alpha,
            "recommendation": "continue",
            "stop": False,
            "reason": None,
        }

    best = _best_arm_by_mean(analysis_frame, treatment_ids=treatment_ids)
    treatment_values = analysis_frame.loc[
        analysis_frame["arm_id"] == best, "analysis_value"
    ].to_numpy(dtype=float)
    control_values = analysis_frame.loc[
        analysis_frame["arm_id"] == control_arm_id, "analysis_value"
    ].to_numpy(dtype=float)

    effect, p_value, effective_weight = two_arm_test(
        treatment=treatment_values,
        control=control_values,
        external_control_n=external_control.n if external_control.enabled else 0,
        external_control_mean=external_control.mean,
        external_control_std=external_control.std,
        external_control_weight=external_control.weight,
        dynamic_borrowing=external_control.dynamic_borrowing,
        commensurability_scale=external_control.commensurability_scale,
        robust_mixture=external_control.robust_mixture,
    )

    posterior_prob = _posterior_probability_superior(
        treatment_values=treatment_values,
        control_values=control_values,
        samples=1800,
        rng=rng,
    )
    predictive_prob = float(np.clip(posterior_prob * np.sqrt(max(info_fraction, 1e-6)), 0.0, 1.0))

    meets_success = (
        enrolled_n >= stopping.min_interim_n
        and posterior_prob >= stopping.success_posterior_threshold
        and np.isfinite(p_value)
        and p_value <= spent_alpha
    )
    meets_futility = (
        enrolled_n >= stopping.min_interim_n
        and predictive_prob <= stopping.futility_posterior_threshold
    )

    stop = False
    reason: str | None = None
    recommendation = "continue"
    if stopping.enabled and meets_success:
        stop = True
        reason = "success"
        recommendation = "stop_for_success"
    elif stopping.enabled and meets_futility:
        stop = True
        reason = "futility"
        recommendation = "stop_for_futility"

    return {
        "interim_index": interim_index,
        "enrolled_n": enrolled_n,
        "information_fraction": info_fraction,
        "spent_alpha": spent_alpha,
        "p_value": float(p_value),
        "effect": float(effect),
        "best_arm": best,
        "posterior_superiority": posterior_prob,
        "predictive_success": predictive_prob,
        "recommendation": recommendation,
        "stop": stop,
        "reason": reason,
        "effective_external_weight": float(effective_weight),
    }


def _best_arm_by_mean(analysis_frame: pd.DataFrame, *, treatment_ids: list[str]) -> str:
    best_id = treatment_ids[0]
    best_mean = -np.inf
    for arm_id in treatment_ids:
        values = analysis_frame.loc[analysis_frame["arm_id"] == arm_id, "analysis_value"]
        if values.empty:
            continue
        mean_value = float(values.mean())
        if mean_value > best_mean:
            best_mean = mean_value
            best_id = arm_id
    return best_id


def _posterior_probability_superior(
    *,
    treatment_values: np.ndarray,
    control_values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    if treatment_values.size < 2 or control_values.size < 2:
        return 0.5

    tx_mean = float(np.mean(treatment_values))
    tx_se = float(np.std(treatment_values, ddof=1) / np.sqrt(max(treatment_values.size, 1)))
    ct_mean = float(np.mean(control_values))
    ct_se = float(np.std(control_values, ddof=1) / np.sqrt(max(control_values.size, 1)))

    tx_draw = rng.normal(tx_mean, max(tx_se, 1e-3), size=samples)
    ct_draw = rng.normal(ct_mean, max(ct_se, 1e-3), size=samples)
    return float(np.mean(tx_draw - ct_draw > 0.0))

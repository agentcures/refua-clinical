"""Adaptive stopping rules and alpha spending utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-untyped]

from .analysis import (
    posterior_probability_superior,
    select_best_treatment_result,
)
from .models import ArmSpec, EndpointSpec, ExternalControlSpec, StoppingSpec


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
    endpoint: EndpointSpec | None = None,
    control_arm_id: str,
    treatment_ids: list[str],
    enrolled_n: int,
    total_n: int,
    interim_index: int,
    stopping: StoppingSpec,
    external_control: ExternalControlSpec,
    rng: np.random.Generator,
    arm_lookup: dict[str, ArmSpec] | None = None,
    arm_activation_enrollment: dict[str, int] | None = None,
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

    endpoint_spec = endpoint or EndpointSpec()
    result = select_best_treatment_result(
        analysis_frame,
        endpoint=endpoint_spec,
        control_arm_id=control_arm_id,
        treatment_ids=treatment_ids,
        external_control=external_control,
        arm_lookup=arm_lookup,
        arm_activation_enrollment=arm_activation_enrollment,
    )

    posterior_prob = posterior_probability_superior(
        result,
        samples=1800,
        rng=rng,
    )
    predictive_prob = float(
        np.clip(posterior_prob * np.sqrt(max(info_fraction, 1e-6)), 0.0, 1.0)
    )

    meets_success = (
        enrolled_n >= stopping.min_interim_n
        and posterior_prob >= stopping.success_posterior_threshold
        and np.isfinite(result.p_value)
        and result.p_value <= spent_alpha
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
        "p_value": float(result.p_value),
        "effect": float(result.effect),
        "effect_raw": float(result.effect_raw),
        "effect_measure": result.effect_measure,
        "analysis_method": result.method,
        "best_arm": result.treatment_id,
        "posterior_superiority": posterior_prob,
        "predictive_success": predictive_prob,
        "recommendation": recommendation,
        "stop": stop,
        "reason": reason,
        "effective_external_weight": float(result.effective_external_weight),
        "concurrent_control_only": bool(result.concurrent_control_only),
    }

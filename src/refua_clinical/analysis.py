"""Endpoint-aware statistical analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import norm  # type: ignore[import-untyped]

from .models import ArmSpec, EndpointSpec, ExternalControlSpec
from .pkpd import two_arm_test


@dataclass(slots=True)
class AnalysisResult:
    treatment_id: str
    effect: float
    effect_raw: float
    p_value: float
    se: float
    effective_external_weight: float
    method: str
    effect_measure: str
    treatment_n: int
    control_n: int
    concurrent_control_only: bool = False


def analyze_treatment_comparison(
    analysis_frame: pd.DataFrame,
    *,
    endpoint: EndpointSpec,
    control_arm_id: str,
    treatment_id: str,
    external_control: ExternalControlSpec,
    arm_lookup: dict[str, ArmSpec] | None = None,
    arm_activation_enrollment: dict[str, int] | None = None,
) -> AnalysisResult:
    comparison, concurrent_only = _comparison_frame(
        analysis_frame,
        control_arm_id=control_arm_id,
        treatment_id=treatment_id,
        arm_lookup=arm_lookup,
        arm_activation_enrollment=arm_activation_enrollment,
    )
    treatment_n = int(np.sum(comparison["arm_id"] == treatment_id))
    control_n = int(np.sum(comparison["arm_id"] == control_arm_id))

    if endpoint.kind == "time_to_event":
        result = _cox_time_to_event_analysis(
            comparison,
            control_arm_id=control_arm_id,
            treatment_id=treatment_id,
        )
    elif endpoint.kind == "longitudinal":
        result = _mmrm_like_longitudinal_analysis(
            comparison,
            endpoint=endpoint,
            control_arm_id=control_arm_id,
            treatment_id=treatment_id,
        )
    else:
        result = _simple_two_arm_analysis(
            comparison,
            endpoint=endpoint,
            control_arm_id=control_arm_id,
            treatment_id=treatment_id,
            external_control=external_control,
        )

    result.treatment_n = treatment_n
    result.control_n = control_n
    result.concurrent_control_only = concurrent_only
    return result


def select_best_treatment_result(
    analysis_frame: pd.DataFrame,
    *,
    endpoint: EndpointSpec,
    control_arm_id: str,
    treatment_ids: list[str],
    external_control: ExternalControlSpec,
    arm_lookup: dict[str, ArmSpec] | None = None,
    arm_activation_enrollment: dict[str, int] | None = None,
) -> AnalysisResult:
    best_result: AnalysisResult | None = None
    for treatment_id in treatment_ids:
        result = analyze_treatment_comparison(
            analysis_frame,
            endpoint=endpoint,
            control_arm_id=control_arm_id,
            treatment_id=treatment_id,
            external_control=external_control,
            arm_lookup=arm_lookup,
            arm_activation_enrollment=arm_activation_enrollment,
        )
        if best_result is None:
            best_result = result
            continue
        if np.isfinite(result.effect) and (
            not np.isfinite(best_result.effect) or result.effect > best_result.effect
        ):
            best_result = result

    if best_result is None:
        raise ValueError("Need at least one treatment arm for comparison")
    return best_result


def posterior_probability_superior(
    result: AnalysisResult,
    *,
    samples: int,
    rng: np.random.Generator,
) -> float:
    if not np.isfinite(result.effect):
        return 0.5
    if not np.isfinite(result.se) or result.se <= 1e-9:
        return 1.0 if result.effect > 0.0 else 0.0
    draws = rng.normal(result.effect, result.se, size=max(int(samples), 200))
    return float(np.mean(draws > 0.0))


def _comparison_frame(
    analysis_frame: pd.DataFrame,
    *,
    control_arm_id: str,
    treatment_id: str,
    arm_lookup: dict[str, ArmSpec] | None,
    arm_activation_enrollment: dict[str, int] | None,
) -> tuple[pd.DataFrame, bool]:
    subset = analysis_frame.loc[
        analysis_frame["arm_id"].isin([control_arm_id, treatment_id])
    ].copy()
    if subset.empty:
        return subset, False

    concurrent_only = False
    if arm_lookup is not None:
        arm = arm_lookup.get(treatment_id)
    else:
        arm = None

    activation_n = None
    if arm_activation_enrollment is not None:
        activation_n = arm_activation_enrollment.get(treatment_id)

    if (
        arm is not None
        and arm.concurrent_control_only
        and activation_n is not None
        and "enrolled_index" in subset.columns
    ):
        control_mask = (subset["arm_id"] == control_arm_id) & (
            subset["enrolled_index"].astype(int) >= int(activation_n)
        )
        treatment_mask = subset["arm_id"] == treatment_id
        filtered = subset.loc[control_mask | treatment_mask].copy()
        if int(np.sum(filtered["arm_id"] == control_arm_id)) >= 4:
            subset = filtered
            concurrent_only = True
    return subset, concurrent_only


def _simple_two_arm_analysis(
    comparison: pd.DataFrame,
    *,
    endpoint: EndpointSpec,
    control_arm_id: str,
    treatment_id: str,
    external_control: ExternalControlSpec,
) -> AnalysisResult:
    treatment = comparison.loc[
        comparison["arm_id"] == treatment_id, "analysis_value"
    ].to_numpy(dtype=float)
    control = comparison.loc[
        comparison["arm_id"] == control_arm_id, "analysis_value"
    ].to_numpy(dtype=float)
    effect, p_value, effective_weight = two_arm_test(
        treatment=treatment,
        control=control,
        external_control_n=external_control.n if external_control.enabled else 0,
        external_control_mean=external_control.mean,
        external_control_std=external_control.std,
        external_control_weight=external_control.weight,
        dynamic_borrowing=external_control.dynamic_borrowing,
        commensurability_scale=external_control.commensurability_scale,
        robust_mixture=external_control.robust_mixture,
    )
    treatment_var = (
        float(np.var(treatment, ddof=1)) if treatment.size > 1 else float("nan")
    )
    control_var = float(np.var(control, ddof=1)) if control.size > 1 else float("nan")
    se = (
        float(
            np.sqrt(
                treatment_var / max(float(treatment.size), 1.0)
                + control_var / max(float(control.size), 1.0)
            )
        )
        if treatment.size > 1 and control.size > 1
        else float("nan")
    )
    return AnalysisResult(
        treatment_id=treatment_id,
        effect=float(effect),
        effect_raw=float(effect),
        p_value=float(p_value),
        se=float(se),
        effective_external_weight=float(effective_weight),
        method="two_arm_z",
        effect_measure=(
            "risk_difference" if endpoint.kind == "binary" else "mean_difference"
        ),
        treatment_n=treatment.size,
        control_n=control.size,
    )


def _cox_time_to_event_analysis(
    comparison: pd.DataFrame,
    *,
    control_arm_id: str,
    treatment_id: str,
) -> AnalysisResult:
    data = comparison.copy()
    if "event_observed" not in data.columns:
        raise ValueError(
            "time_to_event analysis requires event_observed in analysis frame"
        )

    duration = data["endpoint_value"].to_numpy(dtype=float)
    event = data["event_observed"].astype(bool).to_numpy(dtype=bool)
    if (
        np.sum(data["arm_id"] == treatment_id) < 4
        or np.sum(data["arm_id"] == control_arm_id) < 4
    ):
        return AnalysisResult(
            treatment_id=treatment_id,
            effect=float("nan"),
            effect_raw=float("nan"),
            p_value=float("nan"),
            se=float("nan"),
            effective_external_weight=0.0,
            method="cox_ph",
            effect_measure="hazard_ratio",
            treatment_n=int(np.sum(data["arm_id"] == treatment_id)),
            control_n=int(np.sum(data["arm_id"] == control_arm_id)),
        )

    model_frame = pd.DataFrame(
        {
            "duration": duration,
            "event": event.astype(int),
            "treatment_indicator": (data["arm_id"] == treatment_id).astype(int),
        }
    )
    for column in _numeric_covariates(data):
        model_frame[column] = pd.to_numeric(data[column], errors="coerce").fillna(
            float(pd.to_numeric(data[column], errors="coerce").median())
        )

    try:
        model = CoxPHFitter(penalizer=0.05)
        model.fit(model_frame, duration_col="duration", event_col="event")
        summary = model.summary.loc["treatment_indicator"]
        log_hr = float(summary["coef"])
        se = float(summary["se(coef)"])
        p_value = float(summary["p"])
        hazard_ratio = float(np.exp(log_hr))
        return AnalysisResult(
            treatment_id=treatment_id,
            effect=float(-log_hr),
            effect_raw=hazard_ratio,
            p_value=p_value,
            se=se,
            effective_external_weight=0.0,
            method="cox_ph",
            effect_measure="hazard_ratio",
            treatment_n=int(np.sum(data["arm_id"] == treatment_id)),
            control_n=int(np.sum(data["arm_id"] == control_arm_id)),
        )
    except Exception:
        treatment_mask = data["arm_id"] == treatment_id
        control_mask = data["arm_id"] == control_arm_id
        logrank = logrank_test(
            duration[treatment_mask],
            duration[control_mask],
            event_observed_A=event[treatment_mask],
            event_observed_B=event[control_mask],
        )
        tx_events = max(float(np.sum(event[treatment_mask])), 0.5)
        ct_events = max(float(np.sum(event[control_mask])), 0.5)
        tx_time = max(float(np.sum(duration[treatment_mask])), 1e-6)
        ct_time = max(float(np.sum(duration[control_mask])), 1e-6)
        hazard_ratio = (tx_events / tx_time) / max(ct_events / ct_time, 1e-9)
        log_hr = float(np.log(max(hazard_ratio, 1e-9)))
        se = float(np.sqrt(1.0 / tx_events + 1.0 / ct_events))
        return AnalysisResult(
            treatment_id=treatment_id,
            effect=float(-log_hr),
            effect_raw=float(hazard_ratio),
            p_value=float(logrank.p_value),
            se=se,
            effective_external_weight=0.0,
            method="logrank_rate_ratio",
            effect_measure="hazard_ratio",
            treatment_n=int(np.sum(treatment_mask)),
            control_n=int(np.sum(control_mask)),
        )


def _mmrm_like_longitudinal_analysis(
    comparison: pd.DataFrame,
    *,
    endpoint: EndpointSpec,
    control_arm_id: str,
    treatment_id: str,
) -> AnalysisResult:
    long_df = _expand_longitudinal_visits(comparison)
    if long_df.empty:
        return AnalysisResult(
            treatment_id=treatment_id,
            effect=float("nan"),
            effect_raw=float("nan"),
            p_value=float("nan"),
            se=float("nan"),
            effective_external_weight=0.0,
            method="mmrm_cluster_ols",
            effect_measure="lsmean_difference",
            treatment_n=int(np.sum(comparison["arm_id"] == treatment_id)),
            control_n=int(np.sum(comparison["arm_id"] == control_arm_id)),
        )

    visit_levels = sorted(long_df["visit_day"].dropna().astype(int).unique().tolist())
    if len(visit_levels) < 2:
        return _simple_two_arm_analysis(
            comparison,
            endpoint=endpoint,
            control_arm_id=control_arm_id,
            treatment_id=treatment_id,
            external_control=ExternalControlSpec(enabled=False),
        )

    treatment_indicator = (long_df["arm_id"] == treatment_id).astype(float).to_numpy()
    baseline = _numeric_series(long_df, "baseline", default=0.0).to_numpy(dtype=float)
    block_index = _numeric_series(long_df, "block_index", default=0.0).to_numpy(
        dtype=float
    )

    columns: list[np.ndarray] = [
        np.ones(len(long_df), dtype=float),
        treatment_indicator,
        baseline,
        block_index,
    ]
    names = ["intercept", "treatment", "baseline", "block_index"]

    reference_visit = visit_levels[0]
    final_visit = visit_levels[-1]
    for visit in visit_levels[1:]:
        visit_indicator = (
            long_df["visit_day"].astype(int).to_numpy() == int(visit)
        ).astype(float)
        columns.append(visit_indicator)
        names.append(f"visit_{visit}")
        interaction = treatment_indicator * visit_indicator
        columns.append(interaction)
        names.append(f"treatment_visit_{visit}")

    for covariate in _numeric_covariates(long_df):
        if covariate in {"baseline", "block_index"}:
            continue
        values = pd.to_numeric(long_df[covariate], errors="coerce")
        if values.notna().sum() < 4:
            continue
        filled = values.fillna(float(values.median()))
        if float(filled.std(ddof=1)) <= 1e-9:
            continue
        columns.append(filled.to_numpy(dtype=float))
        names.append(covariate)

    x = np.column_stack(columns)
    y = pd.to_numeric(long_df["visit_value"], errors="coerce").to_numpy(dtype=float)
    beta, cov = _cluster_robust_ols(
        x=x,
        y=y,
        clusters=long_df["patient_id"].astype(str).to_numpy(),
    )

    contrast = np.zeros(len(names), dtype=float)
    contrast[names.index("treatment")] = 1.0
    if final_visit != reference_visit:
        name = f"treatment_visit_{final_visit}"
        if name in names:
            contrast[names.index(name)] = 1.0
    effect = float(contrast @ beta)
    se = float(np.sqrt(max(float(contrast @ cov @ contrast), 0.0)))
    if not np.isfinite(se) or se <= 1e-12:
        p_value = 1.0
    else:
        z_value = effect / se
        p_value = float(2.0 * (1.0 - norm.cdf(abs(z_value))))

    return AnalysisResult(
        treatment_id=treatment_id,
        effect=effect,
        effect_raw=effect,
        p_value=p_value,
        se=se,
        effective_external_weight=0.0,
        method="mmrm_cluster_ols",
        effect_measure="lsmean_difference",
        treatment_n=int(np.sum(comparison["arm_id"] == treatment_id)),
        control_n=int(np.sum(comparison["arm_id"] == control_arm_id)),
    )


def _expand_longitudinal_visits(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        visits = item.get("visit_values")
        if not isinstance(visits, (list, np.ndarray)):
            continue
        dropout_day = _coerce_float(item.get("dropout_day"))
        dropped_out = bool(item.get("dropped_out", False))
        for visit in visits:
            if not isinstance(visit, dict):
                continue
            visit_day = _coerce_float(visit.get("day"))
            visit_value = _coerce_float(visit.get("change"))
            if visit_day is None or visit_value is None:
                continue
            if dropped_out and dropout_day is not None and visit_day > dropout_day:
                continue
            row = dict(item)
            row["visit_day"] = float(visit_day)
            row["visit_value"] = float(visit_value)
            rows.append(row)
    return pd.DataFrame(rows)


def _cluster_robust_ols(
    *,
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta

    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        mask = clusters == cluster
        x_cluster = x[mask]
        residual_cluster = residual[mask][:, None]
        meat += x_cluster.T @ (residual_cluster @ residual_cluster.T) @ x_cluster

    n_obs = max(x.shape[0], 1)
    n_params = x.shape[1]
    n_clusters = max(len(unique_clusters), 1)
    correction = 1.0
    if n_clusters > 1 and n_obs > n_params:
        correction = (n_clusters / (n_clusters - 1.0)) * (
            (n_obs - 1.0) / max(n_obs - n_params, 1.0)
        )
    cov = correction * bread @ meat @ bread
    return beta, cov


def _numeric_covariates(frame: pd.DataFrame) -> list[str]:
    preferred = ["age", "weight", "egfr", "biomarker_z", "baseline", "block_index"]
    columns: list[str] = []
    for column in preferred:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() >= 4 and float(values.std(ddof=1)) > 1e-9:
                columns.append(column)
    return columns


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
    return None


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.full(len(frame), float(default), dtype=float), index=frame.index
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().any():
        return values.fillna(float(values.median()))
    return pd.Series(
        np.full(len(frame), float(default), dtype=float), index=frame.index
    )

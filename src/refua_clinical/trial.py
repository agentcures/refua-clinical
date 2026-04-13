"""Clinical trial simulation engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .analysis import select_best_treatment_result
from .estimands import apply_estimand
from .models import (
    ArmSpec,
    InterimUpdate,
    ReplicateResult,
    SimulationConfig,
    TrialSimulationResult,
)
from .pkpd import simulate_pd_outcomes, simulate_pk_metrics
from .stopping import evaluate_interim_decision
from .virtual_patients import generate_virtual_population


def simulate_trials(config: SimulationConfig) -> TrialSimulationResult:
    _validate_arms(config.arms)
    virtual_population = generate_virtual_population(
        config.population, seed=config.seed
    )
    vp_table = virtual_population.table

    replicates: list[ReplicateResult] = []
    for idx in range(config.replicates):
        replicate_seed = config.seed + idx * 17 + 11
        replicates.append(
            _simulate_one_replicate(
                config, vp_table, replicate_id=idx + 1, seed=replicate_seed
            )
        )

    summary = summarize_simulation(replicates)
    summary["endpoint_kind"] = config.endpoint.kind
    run_id = _build_run_id(config.trial_id)
    return TrialSimulationResult(
        run_id=run_id, config=config, summary=summary, replicates=replicates
    )


def summarize_simulation(replicates: list[ReplicateResult]) -> dict[str, Any]:
    if not replicates:
        raise ValueError("Cannot summarize empty replicate set")

    effects = np.array([rep.treatment_effect for rep in replicates], dtype=float)
    p_values = np.array([rep.p_value for rep in replicates], dtype=float)
    target_hits = np.array(
        [float(rep.achieved_target) for rep in replicates], dtype=float
    )
    safety = np.array([rep.safety_event_rate for rep in replicates], dtype=float)
    resp_t = np.array([rep.responders_treatment for rep in replicates], dtype=float)
    resp_c = np.array([rep.responders_control for rep in replicates], dtype=float)
    enrolled = np.array([rep.enrolled_n for rep in replicates], dtype=float)
    stop_success = np.array(
        [float(rep.stop_reason == "success") for rep in replicates],
        dtype=float,
    )
    stop_futility = np.array(
        [float(rep.stop_reason == "futility") for rep in replicates],
        dtype=float,
    )
    ext_weight = np.array(
        [rep.effective_external_weight for rep in replicates], dtype=float
    )
    raw_effects = np.array(
        [
            float(rep.effect_raw)
            if rep.effect_raw is not None and np.isfinite(float(rep.effect_raw))
            else float("nan")
            for rep in replicates
        ],
        dtype=float,
    )
    event_rates = np.array(
        [
            float(rep.event_rate)
            if rep.event_rate is not None and np.isfinite(float(rep.event_rate))
            else float("nan")
            for rep in replicates
        ],
        dtype=float,
    )
    active_arm_counts = np.array(
        [float(len(rep.active_arm_ids)) for rep in replicates],
        dtype=float,
    )
    dropped_arm_counts = np.array(
        [float(len(rep.dropped_arm_ids)) for rep in replicates],
        dtype=float,
    )

    summary = {
        "replicates": int(len(replicates)),
        "power": float(np.nanmean(target_hits)),
        "mean_effect": float(np.nanmean(effects)),
        "effect_p10": float(np.nanpercentile(effects, 10)),
        "effect_p90": float(np.nanpercentile(effects, 90)),
        "median_p_value": float(np.nanmedian(p_values)),
        "safety_event_rate": float(np.nanmean(safety)),
        "responder_rate_treatment": float(np.nanmean(resp_t)),
        "responder_rate_control": float(np.nanmean(resp_c)),
        "allocation_interims_mean": float(
            np.mean([len(rep.allocation_trace) for rep in replicates])
        ),
        "expected_sample_size": float(np.nanmean(enrolled)),
        "stop_success_rate": float(np.nanmean(stop_success)),
        "stop_futility_rate": float(np.nanmean(stop_futility)),
        "effective_external_weight_mean": float(np.nanmean(ext_weight)),
        "active_arm_count_mean": float(np.nanmean(active_arm_counts)),
        "dropped_arm_count_mean": float(np.nanmean(dropped_arm_counts)),
        "endpoint_kind": str(replicates[0].decision_cards[0].get("endpoint_kind"))
        if replicates[0].decision_cards
        else None,
        "analysis_method": replicates[0].analysis_method,
        "effect_measure": replicates[0].effect_measure,
    }
    if np.isfinite(event_rates).any():
        summary["event_rate"] = float(np.nanmean(event_rates))
    if np.isfinite(raw_effects).any():
        summary["mean_effect_raw"] = float(np.nanmean(raw_effects))
    return summary


def trial_result_to_mapping(result: TrialSimulationResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(result.config),
        "summary": result.summary,
        "replicates": [
            {
                "replicate_id": rep.replicate_id,
                "treatment_effect": rep.treatment_effect,
                "p_value": rep.p_value,
                "achieved_target": rep.achieved_target,
                "responders_treatment": rep.responders_treatment,
                "responders_control": rep.responders_control,
                "safety_event_rate": rep.safety_event_rate,
                "enrolled_n": rep.enrolled_n,
                "stop_reason": rep.stop_reason,
                "stop_interim_index": rep.stop_interim_index,
                "effective_external_weight": rep.effective_external_weight,
                "decision_cards": rep.decision_cards,
                "allocation_trace": [asdict(update) for update in rep.allocation_trace],
                "event_rate": rep.event_rate,
                "active_arm_ids": rep.active_arm_ids,
                "dropped_arm_ids": rep.dropped_arm_ids,
                "arm_enrollment_counts": rep.arm_enrollment_counts,
                "analysis_method": rep.analysis_method,
                "effect_measure": rep.effect_measure,
                "effect_raw": rep.effect_raw,
            }
            for rep in result.replicates
        ],
    }


def _simulate_one_replicate(
    config: SimulationConfig,
    population: pd.DataFrame,
    *,
    replicate_id: int,
    seed: int,
) -> ReplicateResult:
    rng = np.random.default_rng(seed)
    total_n = config.enrollment.total_n

    replace = len(population) < total_n
    sample = population.sample(
        n=total_n, replace=replace, random_state=seed
    ).reset_index(drop=True)
    sample = _assign_operational_strata(sample, config=config, rng=rng)
    sample_rows = sample.to_dict(orient="records")

    patient_ids = sample["patient_id"].to_numpy(dtype=int)
    block_index = np.arange(total_n, dtype=int) // max(
        config.enrollment.accrual_per_block, 1
    )
    drift = block_index.astype(float) * float(config.enrollment.drift_per_block)

    control_arm = _control_arm(config.arms)
    treatment_ids = [arm.arm_id for arm in config.arms if not arm.is_control]

    potential_outcomes = _simulate_potential_outcomes(config, sample, drift, rng)

    allocation = _initial_allocation(config.arms)
    enrolled_rows: list[dict[str, Any]] = []
    allocation_trace: list[InterimUpdate] = []
    decision_cards: list[dict[str, Any]] = []
    arm_counts = {arm.arm_id: 0 for arm in config.arms}
    dropped_arm_ids: set[str] = set()
    arm_lookup = {arm.arm_id: arm for arm in config.arms}
    arm_activation_enrollment: dict[str, int] = {}

    stop_reason: str | None = None
    stop_interim_index: int | None = None
    interim_count = 0

    for idx in range(total_n):
        active_arm_ids = _active_arm_ids(
            config.arms,
            enrolled_n=idx,
            interim_index=interim_count,
            dropped_arm_ids=dropped_arm_ids,
            arm_counts=arm_counts,
        )
        active_arms = [arm for arm in config.arms if arm.arm_id in active_arm_ids]
        for active_arm_id in active_arm_ids:
            arm_activation_enrollment.setdefault(active_arm_id, idx + 1)
        if not any(not arm.is_control for arm in active_arms):
            stop_reason = "no_active_treatment"
            stop_interim_index = interim_count if interim_count > 0 else None
            break

        allocation = _normalize_allocation_for_active_arms(
            allocation,
            active_arms,
            arm_counts=arm_counts,
        )
        arm_id = _sample_arm_id(active_arms, allocation, rng)
        arm_counts[arm_id] = arm_counts.get(arm_id, 0) + 1
        arm_outcome = potential_outcomes[arm_id].iloc[idx]
        sample_row = sample_rows[idx]

        rescue_prob = _rescue_probability(config, arm_outcome)
        rescue_use = bool(rng.binomial(1, rescue_prob))

        row = {
            "patient_id": int(patient_ids[idx]),
            "enrolled_index": idx + 1,
            "block_index": int(block_index[idx]),
            "site_id": str(sample_row["site_id"]),
            "country_id": str(sample_row["country_id"]),
            "arm_id": arm_id,
            "endpoint_value": float(arm_outcome["endpoint_value"]),
            "change": float(arm_outcome["change"]),
            "baseline": float(arm_outcome["baseline"]),
            "final_score": float(arm_outcome["final_score"]),
            "responder": bool(arm_outcome["responder"]),
            "safety_event": bool(arm_outcome["safety_event"]),
            "dropped_out": bool(arm_outcome["dropped_out"]),
            "rescue_use": rescue_use,
            "event_observed": bool(arm_outcome.get("event_observed", False)),
            "visit_values": arm_outcome.get("visit_values"),
            "dropout_day": arm_outcome.get("dropout_day"),
            "auc": float(arm_outcome["auc"]),
            "cavg": float(arm_outcome["cavg"]),
            "cmax": float(arm_outcome["cmax"]),
            "ctrough": float(arm_outcome["ctrough"]),
        }
        for column, value in sample_row.items():
            if column in {
                "patient_id",
                "site_id",
                "country_id",
            } or column in row:
                continue
            row[str(column)] = _coerce_numpy_scalar(value)
        enrolled_rows.append(row)

        should_adapt = (
            config.adaptive.enabled
            and (idx + 1) >= config.adaptive.burn_in_n
            and (idx + 1) < total_n
            and ((idx + 1) % max(config.adaptive.interim_every, 1) == 0)
            and interim_count < config.stopping.max_interims
        )
        if should_adapt:
            interim_count += 1
            observed_raw = pd.DataFrame(enrolled_rows)
            analysis = apply_estimand(
                observed_raw,
                control_arm_id=control_arm.arm_id,
                estimand=config.estimand,
            )

            allocation, posterior = _adaptive_allocation(
                config,
                analysis,
                rng,
                active_arm_ids=active_arm_ids,
            )
            dropped_now = _dropped_treatment_arms(
                config,
                posterior=posterior,
                arm_counts=arm_counts,
                active_arm_ids=active_arm_ids,
            )
            if dropped_now:
                dropped_arm_ids.update(dropped_now)
                allocation = _normalize_allocation_for_active_arms(
                    allocation,
                    [
                        arm
                        for arm in config.arms
                        if arm.arm_id
                        in _active_arm_ids(
                            config.arms,
                            enrolled_n=idx + 1,
                            interim_index=interim_count,
                            dropped_arm_ids=dropped_arm_ids,
                            arm_counts=arm_counts,
                        )
                    ],
                    arm_counts=arm_counts,
                )
            allocation_trace.append(
                InterimUpdate(
                    enrolled_n=idx + 1,
                    allocation=allocation,
                    posterior_best_probability=posterior,
                )
            )

            decision = evaluate_interim_decision(
                analysis,
                control_arm_id=control_arm.arm_id,
                treatment_ids=treatment_ids,
                enrolled_n=idx + 1,
                total_n=total_n,
                interim_index=interim_count,
                stopping=config.stopping,
                external_control=config.external_control,
                rng=rng,
                endpoint=config.endpoint,
                arm_lookup=arm_lookup,
                arm_activation_enrollment=arm_activation_enrollment,
            )
            decision["endpoint_kind"] = config.endpoint.kind
            decision["dropped_arms"] = sorted(dropped_now)
            decision["active_arms"] = sorted(active_arm_ids)
            decision_cards.append(decision)

            if bool(decision.get("stop", False)):
                stop_reason = (
                    str(decision.get("reason")) if decision.get("reason") else None
                )
                stop_interim_index = int(interim_count)
                break

    enrolled = pd.DataFrame(enrolled_rows)
    analysis_final = apply_estimand(
        enrolled,
        control_arm_id=control_arm.arm_id,
        estimand=config.estimand,
    )

    best_result = select_best_treatment_result(
        analysis_final,
        endpoint=config.endpoint,
        control_arm_id=control_arm.arm_id,
        treatment_ids=treatment_ids,
        external_control=config.external_control,
        arm_lookup=arm_lookup,
        arm_activation_enrollment=arm_activation_enrollment,
    )
    treatment_arm_id = best_result.treatment_id

    responders_t = float(
        analysis_final.loc[
            analysis_final["arm_id"] == treatment_arm_id,
            "analysis_responder",
        ].mean()
    )
    responders_c = float(
        analysis_final.loc[
            analysis_final["arm_id"] == control_arm.arm_id,
            "analysis_responder",
        ].mean()
    )
    safety_rate = (
        float(enrolled["safety_event"].mean()) if not enrolled.empty else float("nan")
    )
    event_rate = (
        float(enrolled["event_observed"].mean())
        if "event_observed" in enrolled.columns and not enrolled.empty
        else None
    )
    final_active_arm_ids = sorted(
        _active_arm_ids(
            config.arms,
            enrolled_n=len(enrolled_rows),
            interim_index=interim_count,
            dropped_arm_ids=dropped_arm_ids,
            arm_counts=arm_counts,
        )
    )

    achieved_target = bool(
        (stop_reason == "success")
        or _meets_endpoint_target(
            endpoint=config.endpoint,
            effect=best_result.effect,
            effect_raw=best_result.effect_raw,
            p_value=best_result.p_value,
        )
    )

    return ReplicateResult(
        replicate_id=replicate_id,
        treatment_effect=float(best_result.effect),
        p_value=float(best_result.p_value),
        achieved_target=achieved_target,
        responders_treatment=responders_t,
        responders_control=responders_c,
        safety_event_rate=safety_rate,
        enrolled_n=int(len(enrolled_rows)),
        stop_reason=stop_reason,
        stop_interim_index=stop_interim_index,
        effective_external_weight=float(best_result.effective_external_weight),
        decision_cards=decision_cards,
        allocation_trace=allocation_trace,
        event_rate=event_rate,
        active_arm_ids=final_active_arm_ids,
        dropped_arm_ids=sorted(dropped_arm_ids),
        arm_enrollment_counts={key: int(value) for key, value in arm_counts.items()},
        analysis_method=best_result.method,
        effect_measure=best_result.effect_measure,
        effect_raw=best_result.effect_raw,
    )


def _simulate_potential_outcomes(
    config: SimulationConfig,
    sample: pd.DataFrame,
    drift: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    site_country_shift = _operational_shift(sample, config=config, rng=rng)
    outcomes: dict[str, pd.DataFrame] = {}

    for arm in config.arms:
        pk = simulate_pk_metrics(
            sample,
            arm=arm,
            pk_model=config.pk_model,
            duration_days=config.endpoint.assessment_day,
            rng=rng,
        )
        out = simulate_pd_outcomes(
            sample,
            arm=arm,
            endpoint=config.endpoint,
            pd_model=config.pd_model,
            pk=pk,
            drift_block=drift,
            rng=rng,
        )
        if config.endpoint.kind == "longitudinal":
            adjusted_visits: list[list[dict[str, float]]] = []
            for idx, visits in enumerate(out["visit_values"].to_list()):
                patient_visits: list[dict[str, float]] = []
                if isinstance(visits, (list, np.ndarray)):
                    for visit in visits:
                        if not isinstance(visit, dict):
                            continue
                        patient_visits.append(
                            {
                                "day": float(visit.get("day", 0.0)),
                                "change": float(visit.get("change", 0.0))
                                + float(site_country_shift[idx]),
                            }
                        )
                adjusted_visits.append(patient_visits)
            out["visit_values"] = pd.Series(
                adjusted_visits,
                index=out.index,
                dtype=object,
            )
            out["change"] = out["change"] + site_country_shift
            out["endpoint_value"] = out["endpoint_value"] + site_country_shift
        elif config.endpoint.kind != "time_to_event":
            out["change"] = out["change"] + site_country_shift
            out["endpoint_value"] = out["endpoint_value"] + site_country_shift
        if config.endpoint.kind == "time_to_event":
            out["responder"] = ~out["event_observed"].astype(bool)
        else:
            out["responder"] = out["endpoint_value"] >= config.endpoint.target_difference
        outcomes[arm.arm_id] = out

    return outcomes


def _operational_shift(
    sample: pd.DataFrame,
    *,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if not config.heterogeneity.enabled:
        return np.zeros(len(sample), dtype=float)

    site_ids = sample["site_id"].astype(str)
    country_ids = sample["country_id"].astype(str)

    site_levels = {
        site: rng.normal(0.0, config.heterogeneity.site_sd)
        for site in site_ids.unique()
    }
    country_levels = {
        country: rng.normal(0.0, config.heterogeneity.country_sd)
        for country in country_ids.unique()
    }

    site_effect = site_ids.map(site_levels).to_numpy(dtype=float)
    country_effect = country_ids.map(country_levels).to_numpy(dtype=float)
    return site_effect + country_effect


def _assign_operational_strata(
    sample: pd.DataFrame,
    *,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frame = sample.copy()
    if not config.heterogeneity.enabled:
        frame["site_id"] = "site_001"
        frame["country_id"] = "country_001"
        return frame

    n_sites = max(int(config.heterogeneity.n_sites), 1)
    n_countries = max(int(config.heterogeneity.n_countries), 1)

    site_idx = rng.integers(1, n_sites + 1, size=len(frame))
    country_idx = rng.integers(1, n_countries + 1, size=len(frame))

    frame["site_id"] = [f"site_{int(value):03d}" for value in site_idx]
    frame["country_id"] = [f"country_{int(value):03d}" for value in country_idx]
    return frame


def _initial_allocation(arms: list[ArmSpec]) -> dict[str, float]:
    ratio = 1.0 / len(arms)
    return {arm.arm_id: ratio for arm in arms}


def _sample_arm_id(
    arms: list[ArmSpec], allocation: dict[str, float], rng: np.random.Generator
) -> str:
    arm_ids = [arm.arm_id for arm in arms]
    probs = np.array([allocation.get(arm_id, 0.0) for arm_id in arm_ids], dtype=float)
    probs = probs / np.clip(np.sum(probs), 1e-9, None)
    return str(rng.choice(arm_ids, p=probs))


def _adaptive_allocation(
    config: SimulationConfig,
    analysis: pd.DataFrame,
    rng: np.random.Generator,
    *,
    active_arm_ids: list[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    control_id = _control_arm(config.arms).arm_id
    if active_arm_ids is None:
        treatment_ids = [arm.arm_id for arm in config.arms if not arm.is_control]
    else:
        treatment_ids = [
            arm.arm_id
            for arm in config.arms
            if not arm.is_control and arm.arm_id in set(active_arm_ids)
        ]

    if analysis.empty or any(
        len(analysis[analysis["arm_id"] == t]) < 4 for t in treatment_ids
    ):
        active_arms = (
            [arm for arm in config.arms if arm.arm_id in set(active_arm_ids)]
            if active_arm_ids is not None
            else config.arms
        )
        default = _initial_allocation(active_arms)
        return default, {arm.arm_id: 1.0 / len(active_arms) for arm in active_arms}

    if not treatment_ids:
        return {control_id: 1.0}, {control_id: 1.0}

    posterior = _posterior_best_probabilities(
        analysis,
        control_id=control_id,
        treatment_ids=treatment_ids,
        samples=config.adaptive.posterior_samples,
        endpoint_kind=config.endpoint.kind,
        rng=rng,
    )

    min_alloc = float(np.clip(config.adaptive.min_allocation, 0.0, 0.45))
    active_arm_count = max(len(treatment_ids) + 1, 1)
    control_share = max(min_alloc, 1.0 / active_arm_count)

    tx_probs = np.array(
        [posterior.get(arm_id, 0.0) for arm_id in treatment_ids], dtype=float
    )
    if np.sum(tx_probs) <= 0.0:
        tx_probs = np.ones(len(treatment_ids), dtype=float)
    tx_probs = tx_probs / np.sum(tx_probs)

    free_mass = max(1.0 - control_share, 1e-6)
    alloc = {
        control_id: control_share,
    }

    for arm_id, value in zip(treatment_ids, tx_probs, strict=True):
        alloc[arm_id] = free_mass * float(value)

    for arm_id in treatment_ids:
        alloc[arm_id] = max(alloc[arm_id], min_alloc)

    total = sum(alloc.values())
    alloc = {key: value / total for key, value in alloc.items()}

    posterior_with_control = {control_id: 0.0, **posterior}
    return alloc, posterior_with_control


def _posterior_best_probabilities(
    analysis: pd.DataFrame,
    *,
    control_id: str,
    treatment_ids: list[str],
    samples: int,
    endpoint_kind: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    draws: dict[str, np.ndarray] = {}

    if endpoint_kind == "binary":
        for arm_id in [control_id, *treatment_ids]:
            values = analysis.loc[
                analysis["arm_id"] == arm_id, "analysis_responder"
            ].to_numpy(dtype=float)
            success = float(np.sum(values))
            n = float(values.size)
            draws[arm_id] = rng.beta(1.0 + success, 1.0 + n - success, size=samples)
    else:
        for arm_id in [control_id, *treatment_ids]:
            values = analysis.loc[
                analysis["arm_id"] == arm_id, "analysis_value"
            ].to_numpy(dtype=float)
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if values.size > 1 else 10.0
            se = max(sd / np.sqrt(max(values.size, 1)), 1e-3)
            draws[arm_id] = rng.normal(mean, se, size=samples)

    tx_draws = {arm_id: draws[arm_id] - draws[control_id] for arm_id in treatment_ids}
    matrix = np.vstack([tx_draws[arm_id] for arm_id in treatment_ids])
    winners = np.argmax(matrix, axis=0)

    probabilities: dict[str, float] = {}
    for idx, arm_id in enumerate(treatment_ids):
        probabilities[arm_id] = float(np.mean(winners == idx))
    return probabilities


def _normalize_allocation_for_active_arms(
    allocation: dict[str, float],
    arms: list[ArmSpec],
    *,
    arm_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    if not arms:
        return {}
    active_ids = [arm.arm_id for arm in arms]
    baseline_share = 1.0 / len(active_ids)
    clipped = {
        arm_id: max(float(allocation.get(arm_id, baseline_share)), 0.0)
        for arm_id in active_ids
    }
    if arm_counts is not None:
        for arm in arms:
            if not arm.backfill_enabled:
                continue
            target_n = (
                int(arm.backfill_target_n)
                if arm.backfill_target_n is not None
                else max(
                    [arm_counts.get(other.arm_id, 0) for other in arms if not other.is_control]
                    or [0]
                )
            )
            current_n = int(arm_counts.get(arm.arm_id, 0))
            if current_n < target_n:
                clipped[arm.arm_id] = clipped.get(arm.arm_id, 0.0) * max(
                    float(arm.backfill_allocation_multiplier),
                    1.0,
                )
    total = sum(clipped.values())
    if total <= 1e-9:
        return _initial_allocation(arms)
    return {arm_id: value / total for arm_id, value in clipped.items()}


def _active_arm_ids(
    arms: list[ArmSpec],
    *,
    enrolled_n: int,
    interim_index: int,
    dropped_arm_ids: set[str],
    arm_counts: dict[str, int],
) -> list[str]:
    active: list[str] = []
    for arm in arms:
        if enrolled_n < max(int(arm.opens_at_enrollment), 0):
            continue
        if arm.opens_at_interim is not None and interim_index < int(arm.opens_at_interim):
            continue
        if (
            arm.opens_after_arm_drop is not None
            and str(arm.opens_after_arm_drop) not in dropped_arm_ids
        ):
            continue
        if arm.closes_at_interim is not None and interim_index >= int(arm.closes_at_interim):
            continue
        if arm.arm_id in dropped_arm_ids:
            continue
        if arm.max_patients is not None and arm_counts.get(arm.arm_id, 0) >= int(arm.max_patients):
            continue
        active.append(arm.arm_id)
    return active


def _dropped_treatment_arms(
    config: SimulationConfig,
    *,
    posterior: dict[str, float],
    arm_counts: dict[str, int],
    active_arm_ids: list[str],
) -> set[str]:
    if not config.adaptive.allow_arm_dropping:
        return set()
    control_id = _control_arm(config.arms).arm_id
    active_set = set(active_arm_ids)
    dropped: set[str] = set()
    for arm in config.arms:
        if arm.is_control or arm.arm_id == control_id or arm.arm_id not in active_set:
            continue
        if arm_counts.get(arm.arm_id, 0) < 4:
            continue
        if float(posterior.get(arm.arm_id, 0.0)) <= float(config.adaptive.arm_drop_threshold):
            dropped.add(arm.arm_id)
    return dropped


def _rescue_probability(config: SimulationConfig, arm_outcome: pd.Series) -> float:
    if config.endpoint.kind == "time_to_event":
        trigger = float(bool(arm_outcome.get("event_observed", False)))
    else:
        trigger = float(arm_outcome["change"] < config.endpoint.target_difference)
    return float(
        np.clip(
            0.04 + 0.18 * trigger + 0.16 * float(arm_outcome["safety_event"]),
            0.0,
            0.90,
        )
    )


def _meets_endpoint_target(
    *,
    endpoint: Any,
    effect: float,
    effect_raw: float,
    p_value: float,
) -> bool:
    if not np.isfinite(effect) or not np.isfinite(p_value) or p_value > 0.05:
        return False
    if endpoint.kind == "time_to_event":
        return np.isfinite(effect_raw) and effect_raw <= float(endpoint.target_hazard_ratio)
    return effect >= float(endpoint.target_difference)


def _control_arm(arms: list[ArmSpec]) -> ArmSpec:
    for arm in arms:
        if arm.is_control:
            return arm
    raise ValueError("No control arm configured")


def _validate_arms(arms: list[ArmSpec]) -> None:
    control_count = sum(1 for arm in arms if arm.is_control)
    if control_count != 1:
        raise ValueError("Exactly one control arm must be configured")


def _coerce_numpy_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _build_run_id(trial_id: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{trial_id}-{ts}-{uuid4().hex[:8]}"

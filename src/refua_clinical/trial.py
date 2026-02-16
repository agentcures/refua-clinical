"""Clinical trial simulation engine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from .estimands import apply_estimand
from .models import ArmSpec, InterimUpdate, ReplicateResult, SimulationConfig, TrialSimulationResult
from .pkpd import simulate_pd_outcomes, simulate_pk_metrics, two_arm_test
from .stopping import evaluate_interim_decision
from .virtual_patients import generate_virtual_population


def simulate_trials(config: SimulationConfig) -> TrialSimulationResult:
    virtual_population = generate_virtual_population(config.population, seed=config.seed)
    vp_table = virtual_population.table

    replicates: list[ReplicateResult] = []
    for idx in range(config.replicates):
        replicate_seed = config.seed + idx * 17 + 11
        replicates.append(
            _simulate_one_replicate(config, vp_table, replicate_id=idx + 1, seed=replicate_seed)
        )

    summary = summarize_simulation(replicates)
    run_id = _build_run_id(config.trial_id)
    return TrialSimulationResult(
        run_id=run_id, config=config, summary=summary, replicates=replicates
    )


def summarize_simulation(replicates: list[ReplicateResult]) -> dict[str, Any]:
    if not replicates:
        raise ValueError("Cannot summarize empty replicate set")

    effects = np.array([rep.treatment_effect for rep in replicates], dtype=float)
    p_values = np.array([rep.p_value for rep in replicates], dtype=float)
    target_hits = np.array([float(rep.achieved_target) for rep in replicates], dtype=float)
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
    ext_weight = np.array([rep.effective_external_weight for rep in replicates], dtype=float)

    return {
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
    }


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
    sample = population.sample(n=total_n, replace=replace, random_state=seed).reset_index(drop=True)
    sample = _assign_operational_strata(sample, config=config, rng=rng)

    patient_ids = sample["patient_id"].to_numpy(dtype=int)
    block_index = np.arange(total_n, dtype=int) // max(config.enrollment.accrual_per_block, 1)
    drift = block_index.astype(float) * float(config.enrollment.drift_per_block)

    control_arm = _control_arm(config.arms)
    treatment_ids = [arm.arm_id for arm in config.arms if not arm.is_control]

    potential_outcomes = _simulate_potential_outcomes(config, sample, drift, rng)

    allocation = _initial_allocation(config.arms)
    enrolled_rows: list[dict[str, Any]] = []
    allocation_trace: list[InterimUpdate] = []
    decision_cards: list[dict[str, Any]] = []

    stop_reason: str | None = None
    stop_interim_index: int | None = None
    interim_count = 0

    for idx in range(total_n):
        arm_id = _sample_arm_id(config.arms, allocation, rng)
        arm_outcome = potential_outcomes[arm_id].iloc[idx]

        rescue_prob = float(
            np.clip(
                0.04
                + 0.18 * float(arm_outcome["change"] < config.endpoint.target_difference)
                + 0.16 * float(arm_outcome["safety_event"]),
                0.0,
                0.90,
            )
        )
        rescue_use = bool(rng.binomial(1, rescue_prob))

        row = {
            "patient_id": int(patient_ids[idx]),
            "enrolled_index": idx + 1,
            "block_index": int(block_index[idx]),
            "site_id": str(sample.loc[idx, "site_id"]),
            "country_id": str(sample.loc[idx, "country_id"]),
            "arm_id": arm_id,
            "endpoint_value": float(arm_outcome["endpoint_value"]),
            "change": float(arm_outcome["change"]),
            "responder": bool(arm_outcome["responder"]),
            "safety_event": bool(arm_outcome["safety_event"]),
            "dropped_out": bool(arm_outcome["dropped_out"]),
            "rescue_use": rescue_use,
        }
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

            allocation, posterior = _adaptive_allocation(config, analysis, rng)
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
            )
            decision_cards.append(decision)

            if bool(decision.get("stop", False)):
                stop_reason = str(decision.get("reason")) if decision.get("reason") else None
                stop_interim_index = int(interim_count)
                break

    enrolled = pd.DataFrame(enrolled_rows)
    analysis_final = apply_estimand(
        enrolled,
        control_arm_id=control_arm.arm_id,
        estimand=config.estimand,
    )

    treatment_arm_id = _best_treatment_arm(analysis_final, control_arm.arm_id)

    treatment_values = analysis_final.loc[
        analysis_final["arm_id"] == treatment_arm_id,
        "analysis_value",
    ].to_numpy(dtype=float)
    control_values = analysis_final.loc[
        analysis_final["arm_id"] == control_arm.arm_id,
        "analysis_value",
    ].to_numpy(dtype=float)

    effect, p_value, effective_weight = two_arm_test(
        treatment=treatment_values,
        control=control_values,
        external_control_n=config.external_control.n if config.external_control.enabled else 0,
        external_control_mean=config.external_control.mean,
        external_control_std=config.external_control.std,
        external_control_weight=config.external_control.weight,
        dynamic_borrowing=config.external_control.dynamic_borrowing,
        commensurability_scale=config.external_control.commensurability_scale,
        robust_mixture=config.external_control.robust_mixture,
    )

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
    safety_rate = float(enrolled["safety_event"].mean()) if not enrolled.empty else float("nan")

    achieved_target = bool(
        (stop_reason == "success")
        or (
            np.isfinite(effect)
            and np.isfinite(p_value)
            and p_value <= 0.05
            and effect >= float(config.endpoint.target_difference)
        )
    )

    return ReplicateResult(
        replicate_id=replicate_id,
        treatment_effect=float(effect),
        p_value=float(p_value),
        achieved_target=achieved_target,
        responders_treatment=responders_t,
        responders_control=responders_c,
        safety_event_rate=safety_rate,
        enrolled_n=int(len(enrolled_rows)),
        stop_reason=stop_reason,
        stop_interim_index=stop_interim_index,
        effective_external_weight=float(effective_weight),
        decision_cards=decision_cards,
        allocation_trace=allocation_trace,
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
        out["change"] = out["change"] + site_country_shift
        out["endpoint_value"] = out["endpoint_value"] + site_country_shift
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
        site: rng.normal(0.0, config.heterogeneity.site_sd) for site in site_ids.unique()
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
) -> tuple[dict[str, float], dict[str, float]]:
    control_id = _control_arm(config.arms).arm_id
    treatment_ids = [arm.arm_id for arm in config.arms if not arm.is_control]

    if analysis.empty or any(len(analysis[analysis["arm_id"] == t]) < 4 for t in treatment_ids):
        default = _initial_allocation(config.arms)
        return default, {arm.arm_id: 1.0 / len(config.arms) for arm in config.arms}

    posterior = _posterior_best_probabilities(
        analysis,
        control_id=control_id,
        treatment_ids=treatment_ids,
        samples=config.adaptive.posterior_samples,
        endpoint_kind=config.endpoint.kind,
        rng=rng,
    )

    min_alloc = float(np.clip(config.adaptive.min_allocation, 0.0, 0.45))
    control_share = max(min_alloc, 1.0 / len(config.arms))

    tx_probs = np.array([posterior.get(arm_id, 0.0) for arm_id in treatment_ids], dtype=float)
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
            values = analysis.loc[analysis["arm_id"] == arm_id, "analysis_responder"].to_numpy(
                dtype=float
            )
            success = float(np.sum(values))
            n = float(values.size)
            draws[arm_id] = rng.beta(1.0 + success, 1.0 + n - success, size=samples)
    else:
        for arm_id in [control_id, *treatment_ids]:
            values = analysis.loc[analysis["arm_id"] == arm_id, "analysis_value"].to_numpy(
                dtype=float
            )
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


def _control_arm(arms: list[ArmSpec]) -> ArmSpec:
    for arm in arms:
        if arm.is_control:
            return arm
    raise ValueError("No control arm configured")


def _best_treatment_arm(analysis: pd.DataFrame, control_id: str) -> str:
    tx = analysis[analysis["arm_id"] != control_id]
    grouped = tx.groupby("arm_id")["analysis_value"].mean()
    if grouped.empty:
        return control_id
    return str(grouped.sort_values(ascending=False).index[0])


def _build_run_id(trial_id: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{trial_id}-{ts}"

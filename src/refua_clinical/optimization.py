"""Multi-objective design optimization and Pareto front utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .io import config_from_mapping, config_to_mapping
from .models import SimulationConfig
from .trial import simulate_trials


def optimize_design_space(
    config: SimulationConfig,
    *,
    candidate_total_n: list[int] | None = None,
    candidate_interims: list[int] | None = None,
    candidate_burn_in_n: list[int] | None = None,
    candidate_min_allocations: list[float] | None = None,
    candidate_success_thresholds: list[float] | None = None,
    replicates_per_candidate: int = 60,
) -> dict[str, Any]:
    base_n = max(config.enrollment.total_n, 60)
    if candidate_total_n is None:
        candidate_total_n = sorted({int(base_n * 0.7), int(base_n), int(base_n * 1.3)})
    if candidate_interims is None:
        candidate_interims = sorted({20, 30, 45})
    if candidate_burn_in_n is None:
        candidate_burn_in_n = [int(config.adaptive.burn_in_n)]
    if candidate_min_allocations is None:
        candidate_min_allocations = [float(config.adaptive.min_allocation)]
    if candidate_success_thresholds is None:
        candidate_success_thresholds = [float(config.stopping.success_posterior_threshold)]

    candidates: list[dict[str, Any]] = []
    for total_n in sorted(set(candidate_total_n)):
        for interim_every in sorted(set(candidate_interims)):
            for burn_in_n in sorted(set(candidate_burn_in_n)):
                for min_allocation in sorted(set(candidate_min_allocations)):
                    for success_threshold in sorted(set(candidate_success_thresholds)):
                        if interim_every >= total_n or burn_in_n >= total_n:
                            continue
                        candidate_config = _clone_for_candidate(
                            config,
                            total_n=max(total_n, 30),
                            interim_every=max(interim_every, 10),
                            burn_in_n=max(int(burn_in_n), 10),
                            min_allocation=float(min_allocation),
                            success_threshold=float(success_threshold),
                            replicates=max(20, replicates_per_candidate),
                        )
                        result = simulate_trials(candidate_config)
                        summary = result.summary
                        expected_n = float(summary.get("expected_sample_size", total_n))
                        interim_mean = float(summary.get("allocation_interims_mean", 0.0))
                        cost = _expected_cost(candidate_config, expected_n, interim_mean)

                        candidates.append(
                            {
                                "total_n": total_n,
                                "interim_every": interim_every,
                                "burn_in_n": int(burn_in_n),
                                "min_allocation": float(min_allocation),
                                "success_threshold": float(success_threshold),
                                "power": float(summary["power"]),
                                "mean_effect": float(summary["mean_effect"]),
                                "safety_event_rate": float(summary["safety_event_rate"]),
                                "expected_sample_size": expected_n,
                                "expected_cost": cost,
                                "stop_success_rate": float(summary.get("stop_success_rate", 0.0)),
                                "stop_futility_rate": float(summary.get("stop_futility_rate", 0.0)),
                            }
                        )

    if not candidates:
        raise ValueError("No candidate designs available for optimization")

    scored = _score_candidates(candidates)
    pareto = _pareto_front(scored)

    best = sorted(scored, key=lambda item: float(item["utility_score"]), reverse=True)[
        0
    ]
    return {
        "best_candidate": best,
        "candidates": scored,
        "pareto_front": pareto,
        "objective_directions": {
            "power": "maximize",
            "mean_effect": "maximize",
            "safety_event_rate": "minimize",
            "expected_cost": "minimize",
        },
    }


def _clone_for_candidate(
    config: SimulationConfig,
    *,
    total_n: int,
    interim_every: int,
    burn_in_n: int,
    min_allocation: float,
    success_threshold: float,
    replicates: int,
) -> SimulationConfig:
    payload = config_to_mapping(config)
    payload["enrollment"]["total_n"] = int(total_n)
    payload["adaptive"]["interim_every"] = int(interim_every)
    payload["adaptive"]["burn_in_n"] = int(burn_in_n)
    payload["adaptive"]["min_allocation"] = float(min_allocation)
    payload["stopping"]["success_posterior_threshold"] = float(success_threshold)
    payload["replicates"] = int(replicates)
    payload["seed"] = int(config.seed) + int(total_n) + int(interim_every)
    return config_from_mapping(payload)


def _expected_cost(
    config: SimulationConfig, expected_n: float, interim_mean: float
) -> float:
    return float(
        expected_n * config.costs.cost_per_patient
        + interim_mean * config.costs.cost_per_interim
    )


def _score_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = list(candidates)
    powers = np.array([float(item["power"]) for item in frame], dtype=float)
    effects = np.array([float(item["mean_effect"]) for item in frame], dtype=float)
    safety = np.array([float(item["safety_event_rate"]) for item in frame], dtype=float)
    costs = np.array([float(item["expected_cost"]) for item in frame], dtype=float)

    power_norm = _minmax(powers)
    effect_norm = _minmax(effects)
    safety_norm = 1.0 - _minmax(safety)
    cost_norm = 1.0 - _minmax(costs)

    utility = (
        0.40 * power_norm + 0.25 * effect_norm + 0.20 * safety_norm + 0.15 * cost_norm
    )
    for idx, item in enumerate(frame):
        item["utility_score"] = float(utility[idx])
        item["normalized"] = {
            "power": float(power_norm[idx]),
            "mean_effect": float(effect_norm[idx]),
            "safety_event_rate": float(safety_norm[idx]),
            "expected_cost": float(cost_norm[idx]),
        }
    return frame


def _minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-9:
        return np.full(values.shape, 0.5)
    return (values - lo) / (hi - lo)


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_power = float(a["power"])
    b_power = float(b["power"])
    a_effect = float(a["mean_effect"])
    b_effect = float(b["mean_effect"])
    a_safety = float(a["safety_event_rate"])
    b_safety = float(b["safety_event_rate"])
    a_cost = float(a["expected_cost"])
    b_cost = float(b["expected_cost"])

    not_worse = (
        a_power >= b_power
        and a_effect >= b_effect
        and a_safety <= b_safety
        and a_cost <= b_cost
    )
    strictly_better = (
        a_power > b_power
        or a_effect > b_effect
        or a_safety < b_safety
        or a_cost < b_cost
    )
    return bool(not_worse and strictly_better)


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = any(
            _dominates(other, candidate)
            for other in candidates
            if other is not candidate
        )
        if not dominated:
            front.append(candidate)

    return sorted(front, key=lambda item: float(item["utility_score"]), reverse=True)


def optimization_to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    best = payload["best_candidate"]
    lines.append("# Design Optimization")
    lines.append("")
    lines.append("## Best Candidate")
    lines.append(f"- N: {best['total_n']}")
    lines.append(f"- Interim every: {best['interim_every']}")
    lines.append(f"- Burn-in N: {best['burn_in_n']}")
    lines.append(f"- Min allocation: {float(best['min_allocation']):.2f}")
    lines.append(f"- Success threshold: {float(best['success_threshold']):.2f}")
    lines.append(f"- Utility: {float(best['utility_score']):.3f}")
    lines.append(f"- Power: {float(best['power']):.3f}")
    lines.append(f"- Mean effect: {float(best['mean_effect']):.3f}")
    lines.append(f"- Safety rate: {float(best['safety_event_rate']):.3f}")
    lines.append(f"- Expected cost: {float(best['expected_cost']):,.0f}")
    lines.append("")

    lines.append("## Pareto Front")
    for idx, candidate in enumerate(payload.get("pareto_front", []), start=1):
        lines.append(
            f"{idx}. N={candidate['total_n']}, interim={candidate['interim_every']}, "
            f"burn_in={candidate['burn_in_n']}, min_alloc={float(candidate['min_allocation']):.2f}, "
            f"power={float(candidate['power']):.3f}, "
            f"safety={float(candidate['safety_event_rate']):.3f}, "
            f"cost={float(candidate['expected_cost']):,.0f}"
        )

    return "\n".join(lines) + "\n"

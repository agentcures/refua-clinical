"""Value-of-information utilities for trial expansion decisions."""

from __future__ import annotations

from typing import Any

from .io import clone_config
from .models import SimulationConfig
from .trial import _population_table_for_config, _simulate_trials_with_population


def estimate_value_of_information(
    config: SimulationConfig,
    *,
    candidate_extra_n: list[int] | None = None,
    candidate_success_thresholds: list[float] | None = None,
    candidate_min_allocations: list[float] | None = None,
    replicates_per_scenario: int = 60,
) -> dict[str, Any]:
    if candidate_extra_n is None:
        candidate_extra_n = [0, 30, 60, 90]
    if candidate_success_thresholds is None:
        candidate_success_thresholds = [
            float(config.stopping.success_posterior_threshold)
        ]
    if candidate_min_allocations is None:
        candidate_min_allocations = [float(config.adaptive.min_allocation)]

    baseline_n = int(config.enrollment.total_n)
    scenarios: list[dict[str, Any]] = []
    population_cache: dict[int, Any] = {}

    for extra_n in sorted({int(max(0, value)) for value in candidate_extra_n}):
        for success_threshold in sorted(set(candidate_success_thresholds)):
            for min_allocation in sorted(set(candidate_min_allocations)):
                scenario_n = baseline_n + extra_n
                scenario_config = _clone_for_n(
                    config,
                    scenario_n,
                    success_threshold=float(success_threshold),
                    min_allocation=float(min_allocation),
                    replicates=max(replicates_per_scenario, 20),
                )
                population = _population_table_for_config(
                    scenario_config,
                    cache=population_cache,
                )
                result = _simulate_trials_with_population(scenario_config, population)

                utility = _decision_utility(
                    power=float(result.summary["power"]),
                    effect=float(result.summary["mean_effect"]),
                    safety=float(result.summary["safety_event_rate"]),
                    expected_n=float(
                        result.summary.get("expected_sample_size", scenario_n)
                    ),
                    config=scenario_config,
                )
                scenarios.append(
                    {
                        "extra_n": extra_n,
                        "total_n": scenario_n,
                        "success_threshold": float(success_threshold),
                        "min_allocation": float(min_allocation),
                        "power": float(result.summary["power"]),
                        "mean_effect": float(result.summary["mean_effect"]),
                        "safety_event_rate": float(result.summary["safety_event_rate"]),
                        "expected_sample_size": float(
                            result.summary.get("expected_sample_size", scenario_n)
                        ),
                        "utility": utility,
                    }
                )

    baseline = scenarios[0]
    for scenario in scenarios:
        scenario["information_gain"] = float(scenario["utility"] - baseline["utility"])

    best = max(scenarios, key=lambda item: float(item["information_gain"]))
    recommendation = (
        f"Expand by +{best['extra_n']} patients"
        if float(best["information_gain"]) > 0.0 and int(best["extra_n"]) > 0
        else "No expansion recommended"
    )

    return {
        "baseline": baseline,
        "scenarios": scenarios,
        "best_scenario": best,
        "recommendation": recommendation,
    }


def _clone_for_n(
    config: SimulationConfig,
    total_n: int,
    *,
    success_threshold: float,
    min_allocation: float,
    replicates: int,
) -> SimulationConfig:
    cloned = clone_config(config)
    cloned.enrollment.total_n = int(total_n)
    cloned.stopping.success_posterior_threshold = float(success_threshold)
    cloned.adaptive.min_allocation = float(min_allocation)
    cloned.replicates = int(replicates)
    cloned.seed = int(config.seed) + int(total_n)
    return cloned


def _decision_utility(
    *,
    power: float,
    effect: float,
    safety: float,
    expected_n: float,
    config: SimulationConfig,
) -> float:
    benefit = 100.0 * power + 1.8 * effect - 110.0 * max(safety - 0.30, 0.0)
    cost_penalty = (
        expected_n * config.costs.cost_per_patient
        + config.costs.cost_per_interim
        * (expected_n / max(config.adaptive.interim_every, 1))
    ) / 100_000.0
    return float(config.costs.utility_scale * benefit - cost_penalty)


def voi_to_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Value of Information", ""]
    lines.append(f"- Recommendation: {payload['recommendation']}")
    lines.append("")
    lines.append("## Scenarios")
    for item in payload.get("scenarios", []):
        lines.append(
            f"- +{item['extra_n']} (N={item['total_n']}): "
            f"threshold={float(item['success_threshold']):.2f}, "
            f"min_alloc={float(item['min_allocation']):.2f}, "
            f"utility={float(item['utility']):.2f}, gain={float(item['information_gain']):.2f}, "
            f"power={float(item['power']):.3f}"
        )
    return "\n".join(lines) + "\n"

"""Protocol recommendation and rendering."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .io import clone_config
from .models import CandidateProtocolScore, ProtocolRecommendation, SimulationConfig
from .research import list_references
from .trial import _population_table_for_config, _simulate_trials_with_population


def recommend_protocol(
    config: SimulationConfig,
    *,
    replicates_per_candidate: int = 80,
    candidate_total_n: list[int] | None = None,
    candidate_interims: list[int] | None = None,
    candidate_burn_in_n: list[int] | None = None,
    candidate_min_allocations: list[float] | None = None,
    candidate_success_thresholds: list[float] | None = None,
) -> ProtocolRecommendation:
    base_n = max(config.enrollment.total_n, 60)
    if candidate_total_n is None:
        candidate_total_n = sorted(
            {int(base_n * 0.75), int(base_n), int(base_n * 1.25)}
        )
    if candidate_interims is None:
        candidate_interims = sorted({20, 30, 45})
    if candidate_burn_in_n is None:
        candidate_burn_in_n = [int(config.adaptive.burn_in_n)]
    if candidate_min_allocations is None:
        candidate_min_allocations = [float(config.adaptive.min_allocation)]
    if candidate_success_thresholds is None:
        candidate_success_thresholds = [
            float(config.stopping.success_posterior_threshold)
        ]

    candidates: list[CandidateProtocolScore] = []
    population_cache: dict[int, Any] = {}

    for total_n in candidate_total_n:
        for interim_every in candidate_interims:
            for burn_in_n in candidate_burn_in_n:
                for min_allocation in candidate_min_allocations:
                    for success_threshold in candidate_success_thresholds:
                        if interim_every >= total_n or burn_in_n >= total_n:
                            continue
                        candidate_config = _clone_with_design(
                            config,
                            total_n=max(total_n, 30),
                            interim_every=max(interim_every, 10),
                            burn_in_n=max(int(burn_in_n), 10),
                            min_allocation=float(min_allocation),
                            success_threshold=float(success_threshold),
                            replicates=max(20, replicates_per_candidate),
                        )
                        population = _population_table_for_config(
                            candidate_config,
                            cache=population_cache,
                        )
                        result = _simulate_trials_with_population(
                            candidate_config, population
                        )
                        power = float(result.summary["power"])
                        effect = float(result.summary["mean_effect"])
                        safety = float(result.summary["safety_event_rate"])
                        expected_sample_size = float(
                            result.summary.get("expected_sample_size", total_n)
                        )
                        expected_cost = float(
                            expected_sample_size
                            * candidate_config.costs.cost_per_patient
                            + float(result.summary.get("allocation_interims_mean", 0.0))
                            * candidate_config.costs.cost_per_interim
                        )
                        utility = _utility(
                            total_n=total_n,
                            power=power,
                            effect=effect,
                            safety=safety,
                            expected_cost=expected_cost,
                        )
                        candidates.append(
                            CandidateProtocolScore(
                                total_n=total_n,
                                interim_every=interim_every,
                                power=power,
                                expected_effect=effect,
                                safety_rate=safety,
                                expected_sample_size=expected_sample_size,
                                expected_cost=expected_cost,
                                utility=utility,
                                burn_in_n=int(burn_in_n),
                                min_allocation=float(min_allocation),
                                success_threshold=float(success_threshold),
                            )
                        )

    if not candidates:
        raise ValueError("No valid protocol candidates were generated")

    ranked = sorted(candidates, key=lambda item: item.utility, reverse=True)
    best = ranked[0]
    recommended_config = _clone_with_design(
        config,
        total_n=best.total_n,
        interim_every=best.interim_every,
        burn_in_n=best.burn_in_n,
        min_allocation=best.min_allocation,
        success_threshold=best.success_threshold,
        replicates=config.replicates,
    )

    protocol = _build_protocol_payload(recommended_config, best, ranked)
    return ProtocolRecommendation(protocol=protocol, candidates=ranked)


def render_protocol_markdown(protocol: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {protocol['title']}")
    lines.append("")
    lines.append(f"- **Protocol ID:** {protocol['protocol_id']}")
    lines.append(f"- **Indication:** {protocol['indication']}")
    lines.append(f"- **Phase:** {protocol['phase']}")
    lines.append(f"- **Generated:** {protocol['generated_at']}")
    lines.append("")

    lines.append("## Objective")
    lines.append(protocol["objective"])
    lines.append("")

    lines.append("## Design")
    design = protocol["design"]
    randomization_mode = "adaptive" if design["adaptive_randomization"] else "fixed"
    lines.append(f"- Randomized, {randomization_mode} multi-arm trial")
    lines.append(f"- Planned enrollment: {design['planned_enrollment']}")
    lines.append(f"- Interim cadence: every {design['interim_every']} patients")
    lines.append(f"- Burn-in before adaptation: {design['burn_in_n']} patients")
    lines.append(f"- Minimum arm allocation: {design['min_allocation']:.2f}")
    lines.append(f"- Assessment day: {design['assessment_day']}")
    lines.append("")

    lines.append("## Arms")
    for arm in design["arms"]:
        prefix = "Control" if arm["is_control"] else "Treatment"
        interval_raw = arm.get("dosing_interval_hours")
        if isinstance(interval_raw, int | float) and float(interval_raw) > 0.0:
            interval_hours = float(interval_raw)
            if abs(interval_hours % 24.0) <= 1e-6:
                every_days = max(int(round(interval_hours / 24.0)), 1)
                cadence = f"q{every_days}d"
            else:
                cadence = f"every {interval_hours:.1f}h"
        else:
            cadence = f"{arm['schedule_per_day']}x/day"
        lines.append(
            f"- {prefix}: {arm['label']} ({arm['arm_id']}), "
            f"{arm['dose_mg']} mg, {cadence}"
        )
    lines.append("")

    lines.append("## Eligibility")
    for item in protocol["eligibility"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Endpoints")
    endpoint = protocol["endpoint"]
    lines.append(f"- Primary endpoint: {endpoint['name']} ({endpoint['kind']})")
    lines.append(f"- Success threshold: {endpoint['target_difference']}")
    if endpoint["kind"] == "time_to_event":
        lines.append(f"- Event horizon day: {endpoint['event_horizon_day']}")
        lines.append(f"- Target hazard ratio: {endpoint['target_hazard_ratio']:.2f}")
    if endpoint["kind"] == "longitudinal":
        lines.append(
            f"- Planned visit days: {', '.join(str(int(day)) for day in endpoint['visit_days'])}"
        )
    lines.append("")

    lines.append("## Statistical Analysis")
    stats = protocol["statistical_analysis"]
    lines.append(f"- Primary comparison: {stats['primary_comparison']}")
    lines.append(f"- Time-trend adjustment: {stats['time_trend_adjustment']}")
    lines.append(f"- External control borrowing: {stats['external_control']}")
    lines.append(
        f"- Success posterior threshold: {stats['success_posterior_threshold']:.2f}"
    )
    lines.append("")

    lines.append("## Simulated Operating Characteristics")
    perf = protocol["simulated_performance"]
    lines.append(f"- Simulated power: {perf['power']:.3f}")
    lines.append(f"- Expected effect: {perf['expected_effect']:.3f}")
    lines.append(f"- Safety event rate: {perf['safety_rate']:.3f}")
    lines.append(f"- Expected sample size: {perf['expected_sample_size']:.1f}")
    lines.append(f"- Expected operational cost: {perf['expected_cost']:.0f}")
    lines.append(f"- Utility score: {perf['utility']:.2f}")
    lines.append("")

    lines.append("## Research Basis")
    for item in protocol["research_basis"]:
        lines.append(f"- {item['topic']}: [{item['title']}]({item['url']})")

    lines.append("")
    lines.append("## Disclaimer")
    lines.append(protocol["disclaimer"])

    return "\n".join(lines) + "\n"


def _utility(
    *,
    total_n: int,
    power: float,
    effect: float,
    safety: float,
    expected_cost: float,
) -> float:
    safety_penalty = max(safety - 0.30, 0.0) * 120.0
    enrollment_penalty = float(total_n) * 0.04
    cost_penalty = float(expected_cost) / 120_000.0
    return (
        power * 100.0
        + effect * 1.5
        - safety_penalty
        - enrollment_penalty
        - cost_penalty
    )


def _clone_with_design(
    config: SimulationConfig,
    *,
    total_n: int,
    interim_every: int,
    burn_in_n: int,
    min_allocation: float,
    success_threshold: float,
    replicates: int,
) -> SimulationConfig:
    cloned = clone_config(config)
    cloned.replicates = int(replicates)
    cloned.enrollment.total_n = int(total_n)
    cloned.adaptive.interim_every = int(interim_every)
    cloned.adaptive.burn_in_n = int(burn_in_n)
    cloned.adaptive.min_allocation = float(min_allocation)
    cloned.stopping.success_posterior_threshold = float(success_threshold)
    cloned.seed = int(config.seed) + int(total_n) + int(interim_every)
    return cloned


def _build_protocol_payload(
    config: SimulationConfig,
    best: CandidateProtocolScore,
    ranked: list[CandidateProtocolScore],
) -> dict[str, Any]:
    research = list_references()
    eligibility = _eligibility_from_covariates(config)

    return {
        "protocol_id": f"{config.trial_id}-protocol-{datetime.now(UTC).strftime('%Y%m%d')}",
        "title": f"{config.indication} {config.phase} Adaptive PK/PD-Informed Protocol",
        "generated_at": datetime.now(UTC).isoformat(),
        "trial_id": config.trial_id,
        "indication": config.indication,
        "phase": config.phase,
        "objective": config.objective,
        "design": {
            "adaptive_randomization": bool(config.adaptive.enabled),
            "planned_enrollment": int(config.enrollment.total_n),
            "interim_every": int(config.adaptive.interim_every),
            "burn_in_n": int(config.adaptive.burn_in_n),
            "min_allocation": float(config.adaptive.min_allocation),
            "assessment_day": int(config.endpoint.assessment_day),
            "arms": [asdict(arm) for arm in config.arms],
        },
        "eligibility": eligibility,
        "endpoint": {
            "name": config.endpoint.name,
            "kind": config.endpoint.kind,
            "target_difference": config.endpoint.target_difference,
            "responder_threshold": config.endpoint.responder_threshold,
            "event_horizon_day": config.endpoint.event_horizon_day,
            "target_hazard_ratio": config.endpoint.target_hazard_ratio,
            "visit_days": config.endpoint.visit_days,
        },
        "statistical_analysis": {
            "primary_comparison": "Best treatment arm vs concurrent control",
            "time_trend_adjustment": (
                "Include enrollment block term to reduce non-concurrent control bias "
                "in platform-like settings"
            ),
            "external_control": (
                "Power-prior weighted borrowing"
                if config.external_control.enabled
                else "Not used in primary analysis"
            ),
            "alpha": 0.05,
            "success_posterior_threshold": float(
                config.stopping.success_posterior_threshold
            ),
        },
        "simulated_performance": {
            "power": best.power,
            "expected_effect": best.expected_effect,
            "safety_rate": best.safety_rate,
            "expected_sample_size": best.expected_sample_size,
            "expected_cost": best.expected_cost,
            "utility": best.utility,
        },
        "candidate_rankings": [asdict(item) for item in ranked],
        "research_basis": research,
        "disclaimer": (
            "This protocol is simulation-derived decision support for research planning and "
            "must undergo qualified clinical, statistical, ethics, and regulatory review "
            "before real-world use."
        ),
    }


def _eligibility_from_covariates(config: SimulationConfig) -> list[str]:
    lines: list[str] = []
    for covariate in config.population.covariates:
        minimum = covariate.params.get("min")
        maximum = covariate.params.get("max")
        if isinstance(minimum, int | float) and isinstance(maximum, int | float):
            lines.append(
                f"{covariate.name}: {float(minimum):.1f} to {float(maximum):.1f}"
            )
        elif isinstance(minimum, int | float):
            lines.append(f"{covariate.name}: >= {float(minimum):.1f}")
        elif isinstance(maximum, int | float):
            lines.append(f"{covariate.name}: <= {float(maximum):.1f}")
    if not lines:
        lines.append("General adult population with indication-specific eligibility.")
    return lines

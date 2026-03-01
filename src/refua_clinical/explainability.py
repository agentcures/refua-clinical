"""Explainability, narrative generation, and actionable trial advice."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import config_from_mapping, config_to_mapping
from .models import SimulationConfig
from .trial import simulate_trials


def build_advice_report(
    run_payload: dict[str, Any],
    *,
    protocol_payload: dict[str, Any] | None = None,
    optimization_payload: dict[str, Any] | None = None,
    voi_payload: dict[str, Any] | None = None,
    admet_payload: dict[str, Any] | None = None,
    include_sensitivity: bool = False,
    sensitivity_replicates: int = 40,
    sensitivity_delta: float = 0.15,
) -> dict[str, Any]:
    summary = _summary_from_run(run_payload)
    config = _config_from_run(run_payload)
    decision_signals = _decision_signals_from_run(run_payload)

    sensitivity: dict[str, Any] | None = None
    if include_sensitivity and config is not None:
        sensitivity = run_parameter_sensitivity(
            config,
            replicates=max(sensitivity_replicates, 20),
            delta_fraction=float(np.clip(sensitivity_delta, 0.05, 0.40)),
        )

    recommendations = _recommendations(
        summary=summary,
        config=config,
        protocol_payload=protocol_payload,
        optimization_payload=optimization_payload,
        voi_payload=voi_payload,
        admet_payload=admet_payload,
        sensitivity=sensitivity,
        decision_signals=decision_signals,
    )

    narrative = _render_narrative(
        summary=summary,
        recommendations=recommendations,
        admet_payload=admet_payload,
        sensitivity=sensitivity,
        decision_signals=decision_signals,
        optimization_payload=optimization_payload,
        voi_payload=voi_payload,
    )

    return {
        "summary": summary,
        "narrative": narrative,
        "recommendations": recommendations,
        "sensitivity": sensitivity,
        "decision_cards": decision_signals,
        "explainability": {
            "decision_basis": [
                "simulated operating characteristics (power/effect/safety)",
                "interim decision-card traces (posterior/predictive probabilities)",
                "explicit threshold rules for trial decisions",
                "optional one-way sensitivity analysis",
                "optional ADMET risk translation",
                "optional optimization and VOI scenario analysis",
            ],
            "limitations": [
                "simulation quality depends on model assumptions",
                "ADMET translation is heuristic and should be calibrated with program data",
                "recommendations require clinical/statistical review before execution",
            ],
        },
    }


def run_parameter_sensitivity(
    config: SimulationConfig,
    *,
    replicates: int = 40,
    delta_fraction: float = 0.15,
    parameters: list[str] | None = None,
) -> dict[str, Any]:
    if parameters is None:
        parameters = [
            "enrollment.total_n",
            "pd_model.emax",
            "pd_model.ec50_auc",
            "pd_model.safety_intercept",
            "pk_model.cl_l_per_hour",
            "adaptive.interim_every",
        ]

    baseline_cfg = _with_replicates(config, replicates)
    baseline = simulate_trials(baseline_cfg).summary

    drivers: list[dict[str, Any]] = []
    for parameter in parameters:
        plus_cfg = _perturb_parameter(baseline_cfg, parameter, +delta_fraction)
        minus_cfg = _perturb_parameter(baseline_cfg, parameter, -delta_fraction)

        plus_summary = simulate_trials(plus_cfg).summary
        minus_summary = simulate_trials(minus_cfg).summary

        impact = _impact_score(baseline, plus_summary, minus_summary)
        drivers.append(
            {
                "parameter": parameter,
                "impact_score": impact,
                "plus": {
                    "power": plus_summary["power"],
                    "mean_effect": plus_summary["mean_effect"],
                    "safety_event_rate": plus_summary["safety_event_rate"],
                },
                "minus": {
                    "power": minus_summary["power"],
                    "mean_effect": minus_summary["mean_effect"],
                    "safety_event_rate": minus_summary["safety_event_rate"],
                },
            }
        )

    drivers.sort(key=lambda item: float(item["impact_score"]), reverse=True)

    return {
        "baseline": {
            "power": baseline["power"],
            "mean_effect": baseline["mean_effect"],
            "safety_event_rate": baseline["safety_event_rate"],
        },
        "delta_fraction": delta_fraction,
        "replicates": replicates,
        "drivers": drivers,
    }


def render_advice_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    narrative = report.get("narrative", {})
    recommendations = report.get("recommendations", [])
    sensitivity = report.get("sensitivity")

    lines: list[str] = []
    lines.append("# Clinical Trial Advice Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(str(narrative.get("executive_summary", "")))
    lines.append("")
    lines.append("## Key Metrics")
    lines.append(f"- Simulated power: {float(summary.get('power', 0.0)):.3f}")
    lines.append(f"- Mean effect: {float(summary.get('mean_effect', 0.0)):.3f}")
    lines.append(
        f"- Safety event rate: {float(summary.get('safety_event_rate', 0.0)):.3f}"
    )
    lines.append(
        f"- Expected sample size: {float(summary.get('expected_sample_size', 0.0)):.1f}"
    )
    lines.append(
        f"- Stop for success: {float(summary.get('stop_success_rate', 0.0)):.3f}"
    )
    lines.append(
        f"- Stop for futility: {float(summary.get('stop_futility_rate', 0.0)):.3f}"
    )
    lines.append("")

    lines.append("## Recommendations")
    for idx, item in enumerate(recommendations, start=1):
        lines.append(
            f"{idx}. [{item.get('priority', 'medium').upper()}] {item.get('action')} "
            f"- {item.get('rationale')}"
        )
    lines.append("")

    if sensitivity is not None:
        lines.append("## Sensitivity Drivers")
        for driver in sensitivity.get("drivers", [])[:5]:
            lines.append(
                f"- {driver['parameter']}: impact={float(driver['impact_score']):.2f}, "
                f"power(+/-)={float(driver['plus']['power']):.3f}/{float(driver['minus']['power']):.3f}"
            )
        lines.append("")

    decision_cards = report.get("decision_cards")
    if (
        isinstance(decision_cards, dict)
        and float(decision_cards.get("decision_card_count", 0.0)) > 0
    ):
        lines.append("## Interim Decision Cards")
        lines.append(
            "- Mean posterior superiority: "
            f"{float(decision_cards.get('mean_posterior_superiority', 0.0)):.3f}"
        )
        lines.append(
            "- Mean predictive success: "
            f"{float(decision_cards.get('mean_predictive_success', 0.0)):.3f}"
        )
        lines.append(
            f"- Stop-for-success recommendation rate: "
            f"{float(decision_cards.get('stop_for_success_rate', 0.0)):.3f}"
        )
        lines.append(
            f"- Stop-for-futility recommendation rate: "
            f"{float(decision_cards.get('stop_for_futility_rate', 0.0)):.3f}"
        )
        lines.append("")

    lines.append("## Explainability")
    for item in report.get("explainability", {}).get("decision_basis", []):
        lines.append(f"- {item}")

    lines.append("")
    lines.append("## Disclaimer")
    lines.append(
        "Simulation-derived advice for research planning only; final decisions require clinical, "
        "biostatistical, ethics, and regulatory review."
    )

    return "\n".join(lines) + "\n"


def write_advice_artifacts(
    report: dict[str, Any],
    *,
    output_json: Path | None,
    output_markdown: Path | None,
) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_advice_markdown(report), encoding="utf-8")


def _summary_from_run(run_payload: dict[str, Any]) -> dict[str, float]:
    summary_raw = run_payload.get("summary")
    if not isinstance(summary_raw, dict):
        raise ValueError("Run payload missing summary object")

    keys = [
        "power",
        "mean_effect",
        "effect_p10",
        "effect_p90",
        "safety_event_rate",
        "responder_rate_treatment",
        "responder_rate_control",
        "allocation_interims_mean",
        "expected_sample_size",
        "stop_success_rate",
        "stop_futility_rate",
        "effective_external_weight_mean",
        "replicates",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        summary[key] = float(summary_raw.get(key, 0.0))
    return summary


def _config_from_run(run_payload: dict[str, Any]) -> SimulationConfig | None:
    config_raw = run_payload.get("config")
    if not isinstance(config_raw, dict):
        return None
    return config_from_mapping(config_raw)


def _decision_signals_from_run(run_payload: dict[str, Any]) -> dict[str, float]:
    replicates_raw = run_payload.get("replicates")
    if not isinstance(replicates_raw, list):
        return {
            "decision_card_count": 0.0,
            "stop_for_success_rate": 0.0,
            "stop_for_futility_rate": 0.0,
            "mean_posterior_superiority": float("nan"),
            "mean_predictive_success": float("nan"),
        }

    cards: list[dict[str, Any]] = []
    for replicate in replicates_raw:
        if not isinstance(replicate, dict):
            continue
        decision_cards = replicate.get("decision_cards")
        if not isinstance(decision_cards, list):
            continue
        for card in decision_cards:
            if isinstance(card, dict):
                cards.append(card)

    if not cards:
        return {
            "decision_card_count": 0.0,
            "stop_for_success_rate": 0.0,
            "stop_for_futility_rate": 0.0,
            "mean_posterior_superiority": float("nan"),
            "mean_predictive_success": float("nan"),
        }

    posterior = np.array(
        [float(card.get("posterior_superiority", np.nan)) for card in cards],
        dtype=float,
    )
    predictive = np.array(
        [float(card.get("predictive_success", np.nan)) for card in cards],
        dtype=float,
    )
    recommendations = [str(card.get("recommendation", "")) for card in cards]

    return {
        "decision_card_count": float(len(cards)),
        "stop_for_success_rate": float(
            np.mean([rec == "stop_for_success" for rec in recommendations])
        ),
        "stop_for_futility_rate": float(
            np.mean([rec == "stop_for_futility" for rec in recommendations])
        ),
        "mean_posterior_superiority": float(np.nanmean(posterior)),
        "mean_predictive_success": float(np.nanmean(predictive)),
    }


def _with_replicates(config: SimulationConfig, replicates: int) -> SimulationConfig:
    payload = config_to_mapping(config)
    payload["replicates"] = int(replicates)
    return config_from_mapping(payload)


def _perturb_parameter(
    config: SimulationConfig, path: str, direction: float
) -> SimulationConfig:
    payload = config_to_mapping(config)
    keys = path.split(".")

    cursor: dict[str, Any] = payload
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            raise ValueError(f"Parameter path does not resolve to mapping: {path}")
        cursor = child

    leaf = keys[-1]
    value = cursor.get(leaf)
    if not isinstance(value, int | float):
        raise ValueError(f"Parameter path is not numeric: {path}")

    if isinstance(value, int):
        updated = int(max(1, round(float(value) * (1.0 + direction))))
        if path == "adaptive.interim_every":
            total_n = int(payload["enrollment"]["total_n"])
            updated = int(np.clip(updated, 5, max(total_n - 1, 5)))
        cursor[leaf] = updated
    else:
        updated_float = float(max(1e-6, float(value) * (1.0 + direction)))
        cursor[leaf] = updated_float

    return config_from_mapping(payload)


def _impact_score(
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
) -> float:
    power_delta = max(
        abs(float(plus["power"]) - float(baseline["power"])),
        abs(float(minus["power"]) - float(baseline["power"])),
    )
    effect_delta = max(
        abs(float(plus["mean_effect"]) - float(baseline["mean_effect"])),
        abs(float(minus["mean_effect"]) - float(baseline["mean_effect"])),
    )
    safety_delta = max(
        abs(float(plus["safety_event_rate"]) - float(baseline["safety_event_rate"])),
        abs(float(minus["safety_event_rate"]) - float(baseline["safety_event_rate"])),
    )
    return 120.0 * power_delta + 2.5 * effect_delta + 80.0 * safety_delta


def _recommendations(
    *,
    summary: dict[str, float],
    config: SimulationConfig | None,
    protocol_payload: dict[str, Any] | None,
    optimization_payload: dict[str, Any] | None,
    voi_payload: dict[str, Any] | None,
    admet_payload: dict[str, Any] | None,
    sensitivity: dict[str, Any] | None,
    decision_signals: dict[str, float],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    power = summary["power"]
    safety = summary["safety_event_rate"]
    effect = summary["mean_effect"]
    responder_gap = (
        summary["responder_rate_treatment"] - summary["responder_rate_control"]
    )
    stop_futility = summary["stop_futility_rate"]
    expected_sample_size = summary["expected_sample_size"]
    planned_n = (
        float(config.enrollment.total_n) if config is not None else expected_sample_size
    )

    if power < 0.80:
        current_n = int(config.enrollment.total_n) if config is not None else 180
        suggested_n = int(np.ceil(current_n * max(1.0, 0.80 / max(power, 0.35))))
        items.append(
            {
                "priority": "high",
                "action": f"Increase total enrollment to ~{suggested_n} and re-simulate power",
                "rationale": (
                    f"Observed power {power:.3f} is below typical decision threshold (0.80)."
                ),
            }
        )

    if safety > 0.25:
        items.append(
            {
                "priority": "high",
                "action": "Reduce high-dose allocation and add sentinel safety monitoring windows",
                "rationale": (
                    f"Simulated safety event rate {safety:.3f} is above preferred "
                    "operational range."
                ),
            }
        )

    if stop_futility > 0.30:
        items.append(
            {
                "priority": "high",
                "action": (
                    "Revisit inclusion criteria or dose-response assumptions to reduce "
                    "high futility-stop probability"
                ),
                "rationale": (
                    f"Futility stopping rate ({stop_futility:.3f}) suggests weak signal "
                    "under current assumptions."
                ),
            }
        )

    if expected_sample_size < planned_n * 0.80:
        items.append(
            {
                "priority": "medium",
                "action": (
                    "Preserve early-stop rules and resource plan based on "
                    "expected sample-size savings"
                ),
                "rationale": (
                    f"Expected sample size ({expected_sample_size:.1f}) is materially lower "
                    f"than planned enrollment ({planned_n:.0f})."
                ),
            }
        )

    external_weight = summary["effective_external_weight_mean"]
    if external_weight > 0.15:
        items.append(
            {
                "priority": "medium",
                "action": "Run sensitivity analyses across external-control borrowing weights",
                "rationale": (
                    f"Mean effective external borrowing weight is {external_weight:.2f}, "
                    "so conclusions may be borrowing-sensitive."
                ),
            }
        )

    if effect < 3.0 or responder_gap < 0.10:
        items.append(
            {
                "priority": "medium",
                "action": "Consider biomarker-enriched inclusion criteria and dose optimization",
                "rationale": (
                    f"Effect ({effect:.2f}) and responder lift ({responder_gap:.2f}) "
                    "suggest limited separation."
                ),
            }
        )

    if optimization_payload is not None:
        best = optimization_payload.get("best_candidate")
        if isinstance(best, dict):
            best_power = float(best.get("power", power))
            if best_power > power + 0.03:
                items.append(
                    {
                        "priority": "medium",
                        "action": (
                            "Use optimization-best design candidate as next protocol iteration "
                            "and confirm with robustness checks"
                        ),
                        "rationale": (
                            f"Optimization indicates power improvement from {power:.3f} "
                            f"to {best_power:.3f}."
                        ),
                    }
                )

    if voi_payload is not None:
        recommendation = str(voi_payload.get("recommendation", ""))
        best_scenario = voi_payload.get("best_scenario")
        if isinstance(best_scenario, dict):
            gain = float(best_scenario.get("information_gain", 0.0))
            if gain > 0.0 and "No expansion" not in recommendation:
                items.append(
                    {
                        "priority": "medium",
                        "action": recommendation,
                        "rationale": (
                            f"VOI scenario analysis estimates positive information gain "
                            f"({gain:.2f} utility units)."
                        ),
                    }
                )

    if decision_signals["decision_card_count"] > 0:
        futility_card_rate = decision_signals["stop_for_futility_rate"]
        if futility_card_rate > 0.25:
            items.append(
                {
                    "priority": "medium",
                    "action": (
                        "Tighten interim go/no-go criteria and monitor leading indicators "
                        "of predictive success"
                    ),
                    "rationale": (
                        f"Interim decision cards frequently recommend futility "
                        f"({futility_card_rate:.2f})."
                    ),
                }
            )

    if config is not None and not config.adaptive.enabled:
        items.append(
            {
                "priority": "medium",
                "action": "Enable response-adaptive randomization after burn-in",
                "rationale": (
                    "Adaptive allocation can improve information efficiency "
                    "across dose arms."
                ),
            }
        )

    if admet_payload is not None:
        red_flags = [
            str(item)
            for item in admet_payload.get("red_flags", [])
            if isinstance(item, str)
        ]
        safety_score = float(admet_payload.get("safety_score", 0.5))
        if red_flags:
            items.append(
                {
                    "priority": "high" if safety_score < 0.6 else "medium",
                    "action": (
                        "Add targeted risk mitigation (ECG/LFT windows, stopping "
                        "rules, and exclusion criteria) "
                        "for flagged liabilities"
                    ),
                    "rationale": (
                        f"ADMET red flags detected: {', '.join(red_flags[:5])}."
                    ),
                }
            )

    if protocol_payload is not None:
        protocol = protocol_payload.get("protocol")
        if isinstance(protocol, dict):
            perf = protocol.get("simulated_performance")
            if isinstance(perf, dict):
                protocol_power = float(perf.get("power", power))
                if protocol_power > power + 0.03:
                    items.append(
                        {
                            "priority": "medium",
                            "action": (
                                "Adopt the recommended protocol design as the "
                                "new base scenario"
                            ),
                            "rationale": (
                                "Recommended design improves modeled power from "
                                f"{power:.3f} to {protocol_power:.3f}."
                            ),
                        }
                    )

    if sensitivity is not None:
        drivers = sensitivity.get("drivers", [])
        if isinstance(drivers, list) and drivers:
            top = drivers[0]
            if isinstance(top, dict):
                items.append(
                    {
                        "priority": "medium",
                        "action": (
                            f"Prioritize calibration of '{top.get('parameter')}' "
                            "with observed data"
                        ),
                        "rationale": (
                            "Sensitivity analysis indicates highest decision impact "
                            f"(score {float(top.get('impact_score', 0.0)):.2f})."
                        ),
                    }
                )

    if not items:
        items.append(
            {
                "priority": "low",
                "action": "Proceed with current design and schedule periodic simulation refresh",
                "rationale": (
                    "Current run meets core efficacy/safety heuristics under "
                    "modeled assumptions."
                ),
            }
        )

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        items, key=lambda item: priority_rank.get(str(item.get("priority")), 9)
    )


def _render_narrative(
    *,
    summary: dict[str, float],
    recommendations: list[dict[str, Any]],
    admet_payload: dict[str, Any] | None,
    sensitivity: dict[str, Any] | None,
    decision_signals: dict[str, float],
    optimization_payload: dict[str, Any] | None,
    voi_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    executive = (
        "Simulation estimates "
        f"power={summary['power']:.3f}, mean effect={summary['mean_effect']:.2f}, "
        f"safety event rate={summary['safety_event_rate']:.3f}, and expected sample size "
        f"{summary['expected_sample_size']:.1f}. "
        "Recommendations focus on maximizing treatment signal while controlling "
        "safety and execution risk."
    )

    evidence = [
        (
            f"Responder lift is "
            f"{(summary['responder_rate_treatment'] - summary['responder_rate_control']):.2f} "
            "(treatment minus control)."
        ),
        (
            "Effect distribution 10th-90th percentile: "
            f"{summary['effect_p10']:.2f} to {summary['effect_p90']:.2f}."
        ),
        f"Adaptive interims observed (mean): {summary['allocation_interims_mean']:.2f}.",
        (
            "Early-stop rates: success="
            f"{summary['stop_success_rate']:.2f}, futility={summary['stop_futility_rate']:.2f}."
        ),
    ]

    if admet_payload is not None:
        evidence.append(
            f"ADMET score={float(admet_payload.get('admet_score', 0.5)):.2f}, "
            f"safety score={float(admet_payload.get('safety_score', 0.5)):.2f}."
        )

    if decision_signals["decision_card_count"] > 0:
        evidence.append(
            "Interim decision-card means: posterior superiority="
            f"{decision_signals['mean_posterior_superiority']:.2f}, predictive success="
            f"{decision_signals['mean_predictive_success']:.2f}."
        )

    if optimization_payload is not None:
        best = optimization_payload.get("best_candidate")
        if isinstance(best, dict):
            evidence.append(
                f"Optimization best candidate: N={int(best.get('total_n', 0))}, "
                f"interim every {int(best.get('interim_every', 0))}, "
                f"utility={float(best.get('utility_score', 0.0)):.2f}."
            )

    if voi_payload is not None:
        evidence.append(f"VOI recommendation: {voi_payload.get('recommendation')}.")

    if sensitivity is not None:
        drivers = sensitivity.get("drivers", [])
        if isinstance(drivers, list) and drivers:
            top_driver = drivers[0]
            if isinstance(top_driver, dict):
                evidence.append(
                    f"Top sensitivity driver: {top_driver.get('parameter')} "
                    f"(impact score {float(top_driver.get('impact_score', 0.0)):.2f})."
                )

    actions = [item.get("action") for item in recommendations]

    return {
        "executive_summary": executive,
        "evidence": evidence,
        "actionable_takeaways": actions,
    }


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def summarize_recommendations(
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "priority": str(item.get("priority", "medium")),
            "action": str(item.get("action", "")),
            "rationale": str(item.get("rationale", "")),
        }
        for item in recommendations
    ]


def simulation_config_to_dict(config: SimulationConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return asdict(config)

import pandas as pd

import refua_clinical.voi as voi_module
from refua_clinical.models import default_simulation_config
from refua_clinical.transportability import (
    assess_transportability,
    transportability_to_markdown,
)
from refua_clinical.voi import estimate_value_of_information, voi_to_markdown


def test_voi_returns_recommendation_and_scenarios() -> None:
    config = default_simulation_config()
    config.replicates = 10
    config.enrollment.total_n = 120

    payload = estimate_value_of_information(
        config,
        candidate_extra_n=[0, 20],
        candidate_success_thresholds=[0.95, 0.99],
        replicates_per_scenario=20,
    )

    assert payload["scenarios"]
    assert "recommendation" in payload
    assert int(payload["baseline"]["extra_n"]) == 0

    markdown = voi_to_markdown(payload)
    assert "Value of Information" in markdown


def test_information_gain_uses_matching_threshold_baseline(monkeypatch) -> None:
    class _Result:
        def __init__(self, summary: dict[str, float]) -> None:
            self.summary = summary

    def fake_simulate(config, population):  # type: ignore[no-untyped-def]
        del population
        power = (
            0.10
            + 0.40 * float(config.stopping.success_posterior_threshold)
            + 0.001 * float(config.enrollment.total_n)
        )
        return _Result(
            {
                "power": power,
                "mean_effect": 4.0,
                "safety_event_rate": 0.05,
                "expected_sample_size": float(config.enrollment.total_n),
            }
        )

    monkeypatch.setattr(voi_module, "_simulate_trials_with_population", fake_simulate)
    monkeypatch.setattr(
        voi_module, "_population_table_for_config", lambda config, cache: None
    )

    config = default_simulation_config()
    payload = estimate_value_of_information(
        config,
        candidate_extra_n=[0, 30],
        candidate_success_thresholds=[0.90, 0.95],
        candidate_min_allocations=[0.15],
        replicates_per_scenario=20,
    )
    by_key = {
        (int(item["extra_n"]), float(item["success_threshold"])): item
        for item in payload["scenarios"]
    }
    assert by_key[(0, 0.95)]["information_gain"] == 0.0
    matched_gain = by_key[(30, 0.95)]["utility"] - by_key[(0, 0.95)]["utility"]
    other_gain = by_key[(30, 0.95)]["utility"] - by_key[(0, 0.90)]["utility"]
    assert by_key[(30, 0.95)]["information_gain"] == matched_gain
    assert by_key[(30, 0.95)]["information_gain"] != other_gain
    assert int(payload["baseline"]["extra_n"]) == 0
    assert float(payload["baseline"]["success_threshold"]) == float(
        payload["best_scenario"]["success_threshold"]
    )


def test_transportability_assessment_reports_shift() -> None:
    reference = pd.DataFrame(
        {
            "age": [55, 58, 60, 62, 64, 66],
            "weight": [70, 72, 74, 76, 78, 80],
            "egfr": [95, 90, 88, 85, 82, 80],
        }
    )
    target = pd.DataFrame(
        {
            "age": [62, 64, 67, 69, 71, 73],
            "weight": [75, 77, 79, 82, 84, 86],
            "egfr": [84, 82, 79, 77, 74, 72],
        }
    )

    payload = assess_transportability(reference, target, method="ps_weighted")
    assert payload["covariate_smd"]
    assert payload["risk_level"] in {"low", "moderate", "high"}
    assert 0.0 <= float(payload["overlap_score"]) <= 1.0
    assert "weighting" in payload

    markdown = transportability_to_markdown(payload)
    assert "Transportability Assessment" in markdown
    assert "Weighting Repair" in markdown

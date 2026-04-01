import pandas as pd

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

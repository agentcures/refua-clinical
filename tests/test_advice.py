import json

from refua_clinical.admet_integration import (
    apply_admet_adjustments,
    summarize_admet_profile,
)
from refua_clinical.explainability import build_advice_report, render_advice_markdown
from refua_clinical.io import config_to_mapping
from refua_clinical.models import default_simulation_config


def _mock_admet_profile() -> dict:
    return {
        "smiles": "CCO",
        "admet_score": 0.42,
        "adme_score": 0.48,
        "safety_score": 0.38,
        "red_flags": ["hERG", "DILI"],
        "yellow_flags": ["CYP3A4_Veith"],
        "num_predictions": 44,
        "scores": {
            "score_Bioavailability_Ma": 0.35,
            "score_hERG": 0.30,
            "score_DILI": 0.33,
            "score_admet": 0.42,
        },
    }


def test_apply_admet_adjustments_modifies_risk_sensitive_parameters() -> None:
    config = default_simulation_config()
    profile = _mock_admet_profile()

    adjusted, adjustments = apply_admet_adjustments(config, profile)

    assert adjusted.pk_model.bioavailability < config.pk_model.bioavailability
    assert adjusted.pd_model.ec50_auc > config.pd_model.ec50_auc
    assert adjusted.pd_model.safety_intercept > config.pd_model.safety_intercept
    assert adjusted.enrollment.dropout_base > config.enrollment.dropout_base
    assert adjustments["critical_flag_count"] >= 1


def test_build_advice_report_produces_narrative_and_actions() -> None:
    config = default_simulation_config()
    run_payload = {
        "run_id": "mock-run",
        "config": config_to_mapping(config),
        "summary": {
            "replicates": 60,
            "power": 0.62,
            "mean_effect": 2.8,
            "effect_p10": -0.5,
            "effect_p90": 6.5,
            "median_p_value": 0.10,
            "safety_event_rate": 0.31,
            "responder_rate_treatment": 0.39,
            "responder_rate_control": 0.33,
            "allocation_interims_mean": 3.0,
        },
    }

    admet_summary = summarize_admet_profile(_mock_admet_profile())

    report = build_advice_report(
        run_payload,
        admet_payload=admet_summary,
        include_sensitivity=False,
    )

    assert "executive_summary" in report["narrative"]
    assert report["recommendations"]

    markdown = render_advice_markdown(report)
    assert "Clinical Trial Advice Report" in markdown
    assert "Recommendations" in markdown
    assert json.dumps(report)

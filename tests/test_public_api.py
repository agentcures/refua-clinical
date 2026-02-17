import refua_clinical as rc


def test_public_api_is_object_oriented() -> None:
    assert hasattr(rc, "ClinicalStudy")
    assert hasattr(rc, "ClinicalRun")
    assert hasattr(rc, "ClinicalWorkup")


def test_legacy_functional_api_removed_from_top_level() -> None:
    removed_names = [
        "simulate_trials",
        "recommend_protocol",
        "optimize_design_space",
        "estimate_value_of_information",
        "build_advice_report",
        "apply_admet_adjustments",
        "default_simulation_config",
        "default_covariates",
        "apply_refua_adjustments",
        "summarize_refua_payload",
    ]
    for name in removed_names:
        assert not hasattr(rc, name), f"{name} should not be a top-level API export"

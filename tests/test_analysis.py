import numpy as np
import pandas as pd

from refua_clinical.analysis import analyze_treatment_comparison
from refua_clinical.models import ArmSpec, EndpointSpec, ExternalControlSpec


def test_time_to_event_analysis_estimates_hazard_ratio() -> None:
    rng = np.random.default_rng(19)
    n_control = 80
    n_treatment = 80
    control_time = rng.exponential(scale=24.0, size=n_control)
    treatment_time = rng.exponential(scale=96.0, size=n_treatment)
    horizon = 84.0

    frame = pd.DataFrame(
        {
            "arm_id": ["control"] * n_control + ["high"] * n_treatment,
            "endpoint_value": np.concatenate(
                [np.minimum(control_time, horizon), np.minimum(treatment_time, horizon)]
            ),
            "analysis_value": np.concatenate(
                [np.minimum(control_time, horizon), np.minimum(treatment_time, horizon)]
            ),
            "event_observed": np.concatenate(
                [control_time <= horizon, treatment_time <= horizon]
            ),
            "analysis_responder": np.concatenate(
                [control_time > horizon, treatment_time > horizon]
            ),
            "baseline": np.full(n_control + n_treatment, 55.0),
            "age": np.full(n_control + n_treatment, 60.0),
            "weight": np.full(n_control + n_treatment, 78.0),
            "egfr": np.full(n_control + n_treatment, 85.0),
            "biomarker_z": np.zeros(n_control + n_treatment),
            "block_index": np.repeat(np.arange(16), 10),
            "enrolled_index": np.arange(1, n_control + n_treatment + 1),
        }
    )

    result = analyze_treatment_comparison(
        frame,
        endpoint=EndpointSpec(kind="time_to_event", target_hazard_ratio=0.8),
        control_arm_id="control",
        treatment_id="high",
        external_control=ExternalControlSpec(enabled=False),
    )

    assert result.method in {"cox_ph", "logrank_rate_ratio"}
    assert result.effect_measure == "hazard_ratio"
    assert np.isfinite(result.p_value)
    assert float(result.effect_raw) < 1.0


def test_longitudinal_analysis_uses_mmrm_style_estimator() -> None:
    rng = np.random.default_rng(23)
    visit_days = [28, 56, 84]
    rows: list[dict[str, object]] = []
    for idx in range(24):
        rows.append(
            {
                "patient_id": idx + 1,
                "arm_id": "control",
                "analysis_value": 4.0,
                "analysis_responder": False,
                "baseline": float(rng.normal(58.0, 7.0)),
                "age": float(rng.normal(60.0, 10.0)),
                "weight": float(rng.normal(80.0, 12.0)),
                "egfr": float(rng.normal(84.0, 18.0)),
                "biomarker_z": float(rng.normal(0.0, 1.0)),
                "block_index": idx // 6,
                "dropout_day": 84.0,
                "dropped_out": False,
                "visit_values": [
                    {"day": 28.0, "change": float(rng.normal(1.5, 0.8))},
                    {"day": 56.0, "change": float(rng.normal(2.5, 0.8))},
                    {"day": 84.0, "change": float(rng.normal(3.5, 0.8))},
                ],
            }
        )
    for idx in range(24):
        rows.append(
            {
                "patient_id": 100 + idx + 1,
                "arm_id": "high",
                "analysis_value": 9.0,
                "analysis_responder": True,
                "baseline": float(rng.normal(58.0, 7.0)),
                "age": float(rng.normal(60.0, 10.0)),
                "weight": float(rng.normal(80.0, 12.0)),
                "egfr": float(rng.normal(84.0, 18.0)),
                "biomarker_z": float(rng.normal(0.0, 1.0)),
                "block_index": idx // 6,
                "dropout_day": 56.0 if idx % 7 == 0 else 84.0,
                "dropped_out": idx % 7 == 0,
                "visit_values": [
                    {"day": 28.0, "change": float(rng.normal(3.5, 0.8))},
                    {"day": 56.0, "change": float(rng.normal(6.0, 0.8))},
                    {"day": 84.0, "change": float(rng.normal(8.5, 0.8))},
                ],
            }
        )

    frame = pd.DataFrame(rows)
    result = analyze_treatment_comparison(
        frame,
        endpoint=EndpointSpec(kind="longitudinal", visit_days=visit_days),
        control_arm_id="control",
        treatment_id="high",
        external_control=ExternalControlSpec(enabled=False),
    )

    assert result.method == "mmrm_cluster_ols"
    assert result.effect_measure == "lsmean_difference"
    assert np.isfinite(result.p_value)
    assert float(result.effect) > 0.0


def test_concurrent_control_filter_applies_for_late_platform_arm() -> None:
    frame = pd.DataFrame(
        {
            "arm_id": ["control"] * 40 + ["late"] * 12,
            "analysis_value": [2.0] * 20 + [4.0] * 20 + [6.0] * 12,
            "analysis_responder": [False] * 40 + [True] * 12,
            "enrolled_index": list(range(1, 41)) + list(range(31, 43)),
            "baseline": [55.0] * 52,
            "block_index": [0] * 52,
        }
    )
    arm_lookup = {
        "control": ArmSpec("control", "Control", 0.0, is_control=True),
        "late": ArmSpec(
            "late",
            "Late Arm",
            180.0,
            opens_at_interim=2,
            concurrent_control_only=True,
        ),
    }

    result = analyze_treatment_comparison(
        frame,
        endpoint=EndpointSpec(kind="continuous"),
        control_arm_id="control",
        treatment_id="late",
        external_control=ExternalControlSpec(enabled=False),
        arm_lookup=arm_lookup,
        arm_activation_enrollment={"late": 31},
    )

    assert result.concurrent_control_only is True
    assert result.control_n == 10

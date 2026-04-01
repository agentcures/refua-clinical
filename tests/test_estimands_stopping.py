import numpy as np
import pandas as pd

from refua_clinical.estimands import apply_estimand, estimand_summary
from refua_clinical.models import (
    ArmSpec,
    EndpointSpec,
    EstimandSpec,
    ExternalControlSpec,
    StoppingSpec,
)
from refua_clinical.stopping import evaluate_interim_decision


def test_estimand_composite_penalizes_intercurrent_events() -> None:
    observed = pd.DataFrame(
        {
            "arm_id": ["control", "control", "high", "high"],
            "endpoint_value": [3.0, 4.0, 8.0, 9.0],
            "responder": [False, False, True, True],
            "dropped_out": [False, True, False, False],
            "rescue_use": [False, False, True, False],
        }
    )

    analysis = apply_estimand(
        observed,
        control_arm_id="control",
        estimand=EstimandSpec(strategy="composite", rescue_penalty=5.0),
    )
    summary = estimand_summary(analysis)

    assert summary["responder_rate"] <= 0.5
    assert float(analysis.loc[1, "analysis_value"]) < float(
        observed.loc[1, "endpoint_value"]
    )


def test_interim_stopping_returns_decision_card() -> None:
    rng = np.random.default_rng(17)
    analysis = pd.DataFrame(
        {
            "arm_id": ["control"] * 20 + ["high"] * 20,
            "analysis_value": np.concatenate(
                [rng.normal(2.0, 1.0, size=20), rng.normal(7.0, 1.0, size=20)]
            ),
            "analysis_responder": np.concatenate(
                [rng.binomial(1, 0.35, size=20), rng.binomial(1, 0.70, size=20)]
            ).astype(bool),
        }
    )

    decision = evaluate_interim_decision(
        analysis,
        control_arm_id="control",
        treatment_ids=["high"],
        enrolled_n=40,
        total_n=120,
        interim_index=1,
        stopping=StoppingSpec(enabled=True, min_interim_n=20),
        external_control=ExternalControlSpec(enabled=False),
        rng=rng,
    )

    assert decision["recommendation"] in {
        "continue",
        "stop_for_success",
        "stop_for_futility",
    }
    assert "posterior_superiority" in decision


def test_interim_stopping_supports_time_to_event_endpoint() -> None:
    rng = np.random.default_rng(29)
    control_duration = rng.exponential(scale=35.0, size=30)
    treatment_duration = rng.exponential(scale=55.0, size=30)
    horizon = 84.0
    analysis = pd.DataFrame(
        {
            "arm_id": ["control"] * 30 + ["late"] * 30,
            "endpoint_value": np.concatenate(
                [
                    np.minimum(control_duration, horizon),
                    np.minimum(treatment_duration, horizon),
                ]
            ),
            "analysis_value": np.concatenate(
                [
                    np.minimum(control_duration, horizon),
                    np.minimum(treatment_duration, horizon),
                ]
            ),
            "event_observed": np.concatenate(
                [control_duration <= horizon, treatment_duration <= horizon]
            ),
            "analysis_responder": np.concatenate(
                [control_duration > horizon, treatment_duration > horizon]
            ),
            "baseline": rng.normal(55.0, 8.0, size=60),
            "age": rng.normal(60.0, 10.0, size=60),
            "weight": rng.normal(78.0, 12.0, size=60),
            "egfr": rng.normal(84.0, 18.0, size=60),
            "biomarker_z": rng.normal(0.0, 1.0, size=60),
            "block_index": np.repeat(np.arange(6), 10),
            "enrolled_index": np.arange(1, 61),
        }
    )

    decision = evaluate_interim_decision(
        analysis,
        endpoint=EndpointSpec(kind="time_to_event", target_hazard_ratio=0.80),
        control_arm_id="control",
        treatment_ids=["late"],
        enrolled_n=60,
        total_n=120,
        interim_index=2,
        stopping=StoppingSpec(enabled=True, min_interim_n=20),
        external_control=ExternalControlSpec(enabled=False),
        rng=rng,
        arm_lookup={
            "control": ArmSpec("control", "Control", 0.0, is_control=True),
            "late": ArmSpec("late", "Late", 140.0, concurrent_control_only=True),
        },
        arm_activation_enrollment={"late": 21},
    )

    assert decision["effect_measure"] == "hazard_ratio"
    assert decision["analysis_method"] in {"cox_ph", "logrank_rate_ratio"}

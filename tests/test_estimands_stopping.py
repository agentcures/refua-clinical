import numpy as np
import pandas as pd

from refua_clinical.estimands import apply_estimand, estimand_summary
from refua_clinical.models import EstimandSpec, ExternalControlSpec, StoppingSpec
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

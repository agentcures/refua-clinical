import numpy as np
import pandas as pd

from refua_clinical.models import ArmSpec, EndpointSpec, PDModelSpec, PKModelSpec
from refua_clinical.pkpd import simulate_pd_outcomes, simulate_pk_metrics


def test_pk_exposure_increases_with_dose() -> None:
    rng = np.random.default_rng(5)
    covariates = pd.DataFrame(
        {
            "age": rng.normal(60, 10, size=300),
            "weight": rng.normal(78, 12, size=300),
            "egfr": rng.normal(85, 20, size=300),
            "biomarker_z": rng.normal(0, 1, size=300),
        }
    )

    pk_model = PKModelSpec()
    control = ArmSpec("control", "Control", 0.0, is_control=True)
    low = ArmSpec("low", "Low", 60.0)
    high = ArmSpec("high", "High", 140.0)

    pk_control = simulate_pk_metrics(
        covariates, arm=control, pk_model=pk_model, duration_days=84, rng=rng
    )
    pk_low = simulate_pk_metrics(covariates, arm=low, pk_model=pk_model, duration_days=84, rng=rng)
    pk_high = simulate_pk_metrics(
        covariates, arm=high, pk_model=pk_model, duration_days=84, rng=rng
    )

    assert float(pk_control.auc.mean()) == 0.0
    assert float(pk_high.auc.mean()) > float(pk_low.auc.mean())


def test_pd_effect_is_higher_for_treatment_than_control() -> None:
    rng = np.random.default_rng(7)
    covariates = pd.DataFrame(
        {
            "age": rng.normal(60, 10, size=250),
            "weight": rng.normal(78, 12, size=250),
            "egfr": rng.normal(85, 20, size=250),
            "biomarker_z": rng.normal(0, 1, size=250),
        }
    )

    pk_model = PKModelSpec()
    pd_model = PDModelSpec()
    endpoint = EndpointSpec(kind="continuous", target_difference=5.0)

    control = ArmSpec("control", "Control", 0.0, is_control=True)
    high = ArmSpec("high", "High", 140.0)

    drift = np.zeros(len(covariates), dtype=float)

    control_pk = simulate_pk_metrics(
        covariates, arm=control, pk_model=pk_model, duration_days=84, rng=rng
    )
    high_pk = simulate_pk_metrics(
        covariates, arm=high, pk_model=pk_model, duration_days=84, rng=rng
    )

    control_pd = simulate_pd_outcomes(
        covariates,
        arm=control,
        endpoint=endpoint,
        pd_model=pd_model,
        pk=control_pk,
        drift_block=drift,
        rng=rng,
    )
    high_pd = simulate_pd_outcomes(
        covariates,
        arm=high,
        endpoint=endpoint,
        pd_model=pd_model,
        pk=high_pk,
        drift_block=drift,
        rng=rng,
    )

    assert float(high_pd["change"].mean()) > float(control_pd["change"].mean())

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
    pk_low = simulate_pk_metrics(
        covariates, arm=low, pk_model=pk_model, duration_days=84, rng=rng
    )
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


def test_biologic_pk_supports_route_and_interval_dosing() -> None:
    rng = np.random.default_rng(11)
    covariates = pd.DataFrame(
        {
            "age": rng.normal(58, 9, size=260),
            "weight": rng.normal(76, 11, size=260),
            "egfr": rng.normal(82, 18, size=260),
            "biomarker_z": rng.normal(0, 1, size=260),
        }
    )

    sc_model = PKModelSpec(
        modality="biologic",
        route="sc",
        bioavailability=0.65,
        ka_per_hour=0.03,
        cl_l_per_hour=0.25,
        v_l=6.0,
        tmdd_strength=0.35,
        tmdd_cavg_ref=15.0,
    )
    iv_model = PKModelSpec(
        modality="biologic",
        route="iv",
        bioavailability=1.0,
        ka_per_hour=4.0,
        cl_l_per_hour=0.25,
        v_l=6.0,
        tmdd_strength=0.35,
        tmdd_cavg_ref=15.0,
    )

    q2w = ArmSpec("bio_q2w", "Biologic Q2W", 200.0, dosing_interval_hours=336.0)
    q1w = ArmSpec("bio_q1w", "Biologic Q1W", 200.0, dosing_interval_hours=168.0)

    sc_q2w = simulate_pk_metrics(
        covariates, arm=q2w, pk_model=sc_model, duration_days=84, rng=rng
    )
    sc_q1w = simulate_pk_metrics(
        covariates, arm=q1w, pk_model=sc_model, duration_days=84, rng=rng
    )
    iv_q2w = simulate_pk_metrics(
        covariates, arm=q2w, pk_model=iv_model, duration_days=84, rng=rng
    )

    assert float(sc_q2w.auc.mean()) > 0.0
    assert float(sc_q2w.cmax.mean()) > float(sc_q2w.ctrough.mean())
    assert float(sc_q1w.cavg.mean()) > float(sc_q2w.cavg.mean())
    assert float(iv_q2w.cmax.mean()) > float(sc_q2w.cmax.mean())

"""Population PK/PD simulation components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-untyped]

from .models import ArmSpec, EndpointSpec, PDModelSpec, PKModelSpec


@dataclass(slots=True)
class PKMetrics:
    auc: np.ndarray
    cavg: np.ndarray
    cmax: np.ndarray
    ctrough: np.ndarray
    cl: np.ndarray
    v: np.ndarray


class _ModalityProfile(Protocol):
    def effective_bioavailability(self, *, route: str, base: float) -> float: ...

    def effective_clearance(
        self,
        *,
        cl: np.ndarray,
        total_dose: float,
        duration_days: int,
        bioavailability: float,
        pk_model: PKModelSpec,
    ) -> np.ndarray: ...

    def limit_ka(self, ka_per_hour: float) -> float: ...


@dataclass(frozen=True, slots=True)
class _SmallMoleculeProfile:
    def effective_bioavailability(self, *, route: str, base: float) -> float:
        if route == "iv":
            return 1.0
        return float(base)

    def effective_clearance(
        self,
        *,
        cl: np.ndarray,
        total_dose: float,
        duration_days: int,
        bioavailability: float,
        pk_model: PKModelSpec,
    ) -> np.ndarray:
        return cl

    def limit_ka(self, ka_per_hour: float) -> float:
        return max(float(ka_per_hour), 1e-6)


@dataclass(frozen=True, slots=True)
class _BiologicProfile:
    def effective_bioavailability(self, *, route: str, base: float) -> float:
        if route == "iv":
            return 1.0
        if route == "sc":
            return float(np.clip(base, 0.20, 1.0))
        return float(np.clip(base, 0.02, 0.80))

    def effective_clearance(
        self,
        *,
        cl: np.ndarray,
        total_dose: float,
        duration_days: int,
        bioavailability: float,
        pk_model: PKModelSpec,
    ) -> np.ndarray:
        cavg_linear = (
            bioavailability
            * total_dose
            / (max(int(duration_days), 1) * np.clip(cl, 1e-6, None))
        )
        tmdd_strength = max(float(pk_model.tmdd_strength), 0.0)
        tmdd_cavg_ref = max(float(pk_model.tmdd_cavg_ref), 1e-6)
        tmdd_multiplier = 1.0 + tmdd_strength * np.exp(-cavg_linear / tmdd_cavg_ref)
        return cl * tmdd_multiplier

    def limit_ka(self, ka_per_hour: float) -> float:
        return float(np.clip(float(ka_per_hour), 1e-4, 0.5))


class _RouteProfile(Protocol):
    def concentration_metrics(
        self,
        *,
        arm: ArmSpec,
        v: np.ndarray,
        k: np.ndarray,
        tau: float,
        bioavailability: float,
        ka_per_hour: float,
        modality: _ModalityProfile,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class _IVRouteProfile:
    def concentration_metrics(
        self,
        *,
        arm: ArmSpec,
        v: np.ndarray,
        k: np.ndarray,
        tau: float,
        bioavailability: float,
        ka_per_hour: float,
        modality: _ModalityProfile,
    ) -> tuple[np.ndarray, np.ndarray]:
        acc_elim = 1.0 / np.clip(1.0 - np.exp(-k * tau), 1e-6, None)
        cmax = arm.dose_mg / np.clip(v, 1e-6, None) * acc_elim
        ctrough = cmax * np.exp(-k * tau)
        return cmax, ctrough


@dataclass(frozen=True, slots=True)
class _FirstOrderAbsorptionRouteProfile:
    def concentration_metrics(
        self,
        *,
        arm: ArmSpec,
        v: np.ndarray,
        k: np.ndarray,
        tau: float,
        bioavailability: float,
        ka_per_hour: float,
        modality: _ModalityProfile,
    ) -> tuple[np.ndarray, np.ndarray]:
        ka = modality.limit_ka(ka_per_hour)
        delta = np.clip(ka - k, 1e-6, None)
        term = bioavailability * arm.dose_mg * ka / (np.clip(v, 1e-6, None) * delta)
        acc_elim = 1.0 / np.clip(1.0 - np.exp(-k * tau), 1e-6, None)
        acc_abs = 1.0 / np.clip(1.0 - np.exp(-ka * tau), 1e-6, None)

        ctrough = term * (np.exp(-k * tau) * acc_elim - np.exp(-ka * tau) * acc_abs)
        tmax = np.log(np.clip(ka / k, 1.01, None)) / delta
        tmax = np.clip(tmax, 0.0, tau)
        cmax = term * (np.exp(-k * tmax) * acc_elim - np.exp(-ka * tmax) * acc_abs)
        return cmax, ctrough


_MODALITY_PROFILES: dict[str, _ModalityProfile] = {
    "small_molecule": _SmallMoleculeProfile(),
    "biologic": _BiologicProfile(),
}
_ROUTE_PROFILES: dict[str, _RouteProfile] = {
    "oral": _FirstOrderAbsorptionRouteProfile(),
    "sc": _FirstOrderAbsorptionRouteProfile(),
    "iv": _IVRouteProfile(),
}


def simulate_pk_metrics(
    covariates: pd.DataFrame,
    *,
    arm: ArmSpec,
    pk_model: PKModelSpec,
    duration_days: int,
    rng: np.random.Generator,
) -> PKMetrics:
    count = len(covariates)
    if count < 1:
        raise ValueError("Need at least one patient for PK simulation")

    has_interval = (
        arm.dosing_interval_hours is not None and float(arm.dosing_interval_hours) > 0.0
    )
    if arm.dose_mg <= 0.0 or (arm.schedule_per_day <= 0 and not has_interval):
        zeros = np.zeros(count, dtype=float)
        cl = _individualized_cl(covariates, pk_model, rng)
        v = _individualized_v(covariates, pk_model, rng)
        return PKMetrics(auc=zeros, cavg=zeros, cmax=zeros, ctrough=zeros, cl=cl, v=v)

    cl = _individualized_cl(covariates, pk_model, rng)
    v = _individualized_v(covariates, pk_model, rng)
    duration_days = max(duration_days, 1)

    modality_key = str(pk_model.modality).strip().lower()
    route_key = str(pk_model.route).strip().lower()
    modality = _resolve_modality_profile(modality_key)
    route = _resolve_route_profile(route_key)

    tau, total_dose = _dosing_schedule(arm=arm, duration_days=duration_days)
    bioavailability = modality.effective_bioavailability(
        route=route_key,
        base=float(pk_model.bioavailability),
    )
    cl_effective = modality.effective_clearance(
        cl=cl,
        total_dose=total_dose,
        duration_days=duration_days,
        bioavailability=bioavailability,
        pk_model=pk_model,
    )

    auc = bioavailability * total_dose / np.clip(cl_effective, 1e-6, None)
    cavg = auc / duration_days

    k = np.clip(cl_effective / np.clip(v, 1e-6, None), 1e-6, None)
    cmax, ctrough = route.concentration_metrics(
        arm=arm,
        v=v,
        k=k,
        tau=tau,
        bioavailability=bioavailability,
        ka_per_hour=float(pk_model.ka_per_hour),
        modality=modality,
    )

    cmax = np.maximum(cmax, 0.0)
    ctrough = np.maximum(ctrough, 0.0)

    return PKMetrics(
        auc=auc, cavg=cavg, cmax=cmax, ctrough=ctrough, cl=cl_effective, v=v
    )


def _dosing_schedule(*, arm: ArmSpec, duration_days: int) -> tuple[float, float]:
    if arm.dosing_interval_hours is not None and float(arm.dosing_interval_hours) > 0.0:
        tau = max(float(arm.dosing_interval_hours), 1e-6)
        administrations = max(int(np.ceil(duration_days * 24.0 / tau)), 1)
    else:
        schedule = max(int(arm.schedule_per_day), 1)
        tau = 24.0 / float(schedule)
        administrations = max(int(duration_days * schedule), 1)
    total_dose = float(arm.dose_mg) * float(administrations)
    return tau, total_dose


def _resolve_modality_profile(raw: str) -> _ModalityProfile:
    profile = _MODALITY_PROFILES.get(raw)
    if profile is None:
        raise ValueError("pk_model.modality must be 'small_molecule' or 'biologic'")
    return profile


def _resolve_route_profile(raw: str) -> _RouteProfile:
    profile = _ROUTE_PROFILES.get(raw)
    if profile is None:
        raise ValueError("pk_model.route must be 'oral', 'iv', or 'sc'")
    return profile


def simulate_pd_outcomes(
    covariates: pd.DataFrame,
    *,
    arm: ArmSpec,
    endpoint: EndpointSpec,
    pd_model: PDModelSpec,
    pk: PKMetrics,
    drift_block: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = len(covariates)
    if n < 1:
        raise ValueError("Need at least one patient for PD simulation")

    baseline = rng.normal(pd_model.baseline_mean, pd_model.baseline_sd, size=n)
    placebo = pd_model.placebo_improvement_per_day * endpoint.assessment_day

    treatment_effect = _emax(pk.auc, pd_model.emax, pd_model.ec50_auc)
    if arm.is_control:
        treatment_effect = np.zeros_like(treatment_effect)

    biomarker = _series(covariates, "biomarker_z", default=0.0)
    effect_modifier = pd_model.biomarker_effect * biomarker
    if arm.is_control:
        effect_modifier = 0.15 * effect_modifier

    drift_penalty = drift_block

    change = placebo + treatment_effect + effect_modifier - drift_penalty
    change += rng.normal(0.0, pd_model.residual_sd, size=n)

    final_score = baseline - change

    safety_linear = (
        pd_model.safety_intercept
        + pd_model.safety_exposure_coef * pk.cavg
        + pd_model.safety_age_coef * ((_series(covariates, "age", 60.0) - 60.0) / 10.0)
    )
    safety_prob = expit(safety_linear)
    safety_event = rng.binomial(1, np.clip(safety_prob, 0.0, 1.0), size=n).astype(bool)

    drop_base = 0.08 + 0.35 * safety_prob + 0.10 * (drift_block > 0).astype(float)
    dropped_out = rng.binomial(1, np.clip(drop_base, 0.0, 0.95), size=n).astype(bool)

    if endpoint.kind == "binary":
        responders = change >= endpoint.responder_threshold
        endpoint_value = responders.astype(float)
    else:
        responders = change >= endpoint.target_difference
        endpoint_value = change

    return pd.DataFrame(
        {
            "baseline": baseline,
            "change": change,
            "final_score": final_score,
            "endpoint_value": endpoint_value,
            "responder": responders,
            "safety_event": safety_event,
            "dropped_out": dropped_out,
            "auc": pk.auc,
            "cavg": pk.cavg,
            "cmax": pk.cmax,
            "ctrough": pk.ctrough,
            "cl": pk.cl,
            "v": pk.v,
        }
    )


def _individualized_cl(
    covariates: pd.DataFrame,
    pk_model: PKModelSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    weight = _series(covariates, "weight", default=75.0)
    egfr = _series(covariates, "egfr", default=85.0)
    eta = rng.normal(0.0, pk_model.omega_cl, size=len(covariates))

    scale = (
        1.0
        + pk_model.covariate_effect_weight * ((weight - 75.0) / 75.0)
        + pk_model.covariate_effect_egfr * ((egfr - 85.0) / 85.0)
    )
    scale = np.clip(scale, 0.3, 2.5)
    return pk_model.cl_l_per_hour * scale * np.exp(eta)


def _individualized_v(
    covariates: pd.DataFrame,
    pk_model: PKModelSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    weight = _series(covariates, "weight", default=75.0)
    eta = rng.normal(0.0, pk_model.omega_v, size=len(covariates))
    scale = np.clip(weight / 75.0, 0.5, 2.5)
    return pk_model.v_l * scale * np.exp(eta)


def _emax(exposure: np.ndarray, emax: float, ec50: float) -> np.ndarray:
    bounded = np.clip(exposure, 0.0, None)
    return emax * bounded / (ec50 + bounded + 1e-6)


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _series(frame: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name in frame.columns:
        return frame[name].to_numpy(dtype=float)
    return np.full(len(frame), float(default), dtype=float)


def two_arm_test(
    *,
    treatment: np.ndarray,
    control: np.ndarray,
    external_control_n: int,
    external_control_mean: float,
    external_control_std: float,
    external_control_weight: float,
    dynamic_borrowing: bool = False,
    commensurability_scale: float = 1.75,
    robust_mixture: float = 0.25,
) -> tuple[float, float, float]:
    if treatment.size < 2 or control.size < 2:
        return float("nan"), float("nan"), 0.0

    control_mean = float(np.mean(control))
    control_var = float(np.var(control, ddof=1))
    control_n = float(control.size)
    effective_weight = float(np.clip(external_control_weight, 0.0, 1.0))

    if external_control_n > 0 and effective_weight > 0:
        if dynamic_borrowing:
            z_distance = abs(control_mean - float(external_control_mean)) / max(
                float(external_control_std),
                1e-6,
            )
            shrink = np.exp(
                -0.5 * (z_distance / max(float(commensurability_scale), 1e-6)) ** 2
            )
            effective_weight = float(effective_weight * shrink)
            effective_weight = float(
                effective_weight * (1.0 - np.clip(float(robust_mixture), 0.0, 0.9))
            )

        pseudo_n = max(float(external_control_n) * effective_weight, 0.0)
        if pseudo_n > 0:
            pooled_n = control_n + pseudo_n
            pooled_mean = (
                control_mean * control_n + float(external_control_mean) * pseudo_n
            ) / pooled_n
            pooled_var = (
                control_var * max(control_n - 1.0, 1.0)
                + (float(external_control_std) ** 2) * max(pseudo_n - 1.0, 1.0)
            ) / max(pooled_n - 2.0, 1.0)
            control_mean = pooled_mean
            control_var = pooled_var
            control_n = pooled_n

    treatment_mean = float(np.mean(treatment))
    treatment_var = float(np.var(treatment, ddof=1))
    treatment_n = float(treatment.size)

    diff = treatment_mean - control_mean
    se = np.sqrt(treatment_var / treatment_n + control_var / control_n)
    if se <= 1e-12:
        return diff, 1.0, effective_weight

    z = diff / se
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return diff, p_value, effective_weight

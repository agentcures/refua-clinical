"""Typed configuration and result models for refua-clinical."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DistributionKind = Literal["normal", "lognormal", "beta", "uniform", "categorical"]
EndpointKind = Literal["continuous", "binary"]
EstimandStrategy = Literal[
    "treatment_policy",
    "hypothetical",
    "composite",
    "while_on_treatment",
]
AlphaSpendingKind = Literal["obrien_fleming", "pocock", "linear"]


@dataclass(slots=True)
class CovariateSpec:
    name: str
    distribution: DistributionKind
    params: dict[str, Any]


@dataclass(slots=True)
class VirtualPopulationSpec:
    size: int = 6000
    covariates: list[CovariateSpec] = field(default_factory=list)
    correlation: list[list[float]] | None = None


@dataclass(slots=True)
class PKModelSpec:
    bioavailability: float = 1.0
    ka_per_hour: float = 0.9
    cl_l_per_hour: float = 9.5
    v_l: float = 110.0
    omega_cl: float = 0.30
    omega_v: float = 0.25
    residual_prop: float = 0.12
    covariate_effect_weight: float = 0.20
    covariate_effect_egfr: float = 0.35


@dataclass(slots=True)
class PDModelSpec:
    baseline_mean: float = 55.0
    baseline_sd: float = 10.0
    placebo_improvement_per_day: float = 0.030
    emax: float = 20.0
    ec50_auc: float = 70.0
    biomarker_effect: float = 2.0
    residual_sd: float = 5.5
    safety_intercept: float = -2.2
    safety_exposure_coef: float = 0.020
    safety_age_coef: float = 0.08


@dataclass(slots=True)
class ArmSpec:
    arm_id: str
    label: str
    dose_mg: float
    schedule_per_day: int = 1
    is_control: bool = False


@dataclass(slots=True)
class EndpointSpec:
    name: str = "change_from_baseline"
    kind: EndpointKind = "continuous"
    assessment_day: int = 84
    responder_threshold: float = 12.0
    target_difference: float = 6.0


@dataclass(slots=True)
class EnrollmentSpec:
    total_n: int = 180
    accrual_per_block: int = 30
    dropout_base: float = 0.08
    drift_per_block: float = 0.60


@dataclass(slots=True)
class AdaptiveDesignSpec:
    enabled: bool = True
    burn_in_n: int = 60
    interim_every: int = 30
    min_allocation: float = 0.15
    posterior_samples: int = 600


@dataclass(slots=True)
class ExternalControlSpec:
    enabled: bool = False
    n: int = 0
    mean: float = 0.0
    std: float = 1.0
    weight: float = 0.0
    dynamic_borrowing: bool = True
    commensurability_scale: float = 1.75
    robust_mixture: float = 0.25


@dataclass(slots=True)
class EstimandSpec:
    strategy: EstimandStrategy = "treatment_policy"
    rescue_penalty: float = 4.0
    control_imputation_shift: float = 0.0


@dataclass(slots=True)
class StoppingSpec:
    enabled: bool = True
    one_sided_alpha: float = 0.025
    alpha_spending: AlphaSpendingKind = "obrien_fleming"
    success_posterior_threshold: float = 0.99
    futility_posterior_threshold: float = 0.20
    min_interim_n: int = 60
    max_interims: int = 6


@dataclass(slots=True)
class HeterogeneitySpec:
    enabled: bool = False
    n_sites: int = 25
    n_countries: int = 6
    site_sd: float = 1.25
    country_sd: float = 0.80


@dataclass(slots=True)
class OperationalCostSpec:
    cost_per_patient: float = 25_000.0
    cost_per_interim: float = 45_000.0
    utility_scale: float = 1.0


@dataclass(slots=True)
class SimulationConfig:
    trial_id: str
    indication: str
    phase: str
    objective: str
    seed: int = 7
    replicates: int = 250
    population: VirtualPopulationSpec = field(default_factory=VirtualPopulationSpec)
    pk_model: PKModelSpec = field(default_factory=PKModelSpec)
    pd_model: PDModelSpec = field(default_factory=PDModelSpec)
    endpoint: EndpointSpec = field(default_factory=EndpointSpec)
    enrollment: EnrollmentSpec = field(default_factory=EnrollmentSpec)
    adaptive: AdaptiveDesignSpec = field(default_factory=AdaptiveDesignSpec)
    external_control: ExternalControlSpec = field(default_factory=ExternalControlSpec)
    estimand: EstimandSpec = field(default_factory=EstimandSpec)
    stopping: StoppingSpec = field(default_factory=StoppingSpec)
    heterogeneity: HeterogeneitySpec = field(default_factory=HeterogeneitySpec)
    costs: OperationalCostSpec = field(default_factory=OperationalCostSpec)
    arms: list[ArmSpec] = field(default_factory=list)


@dataclass(slots=True)
class InterimUpdate:
    enrolled_n: int
    allocation: dict[str, float]
    posterior_best_probability: dict[str, float]


@dataclass(slots=True)
class ReplicateResult:
    replicate_id: int
    treatment_effect: float
    p_value: float
    achieved_target: bool
    responders_treatment: float
    responders_control: float
    safety_event_rate: float
    enrolled_n: int
    stop_reason: str | None
    stop_interim_index: int | None
    effective_external_weight: float
    decision_cards: list[dict[str, Any]]
    allocation_trace: list[InterimUpdate]


@dataclass(slots=True)
class TrialSimulationResult:
    run_id: str
    config: SimulationConfig
    summary: dict[str, Any]
    replicates: list[ReplicateResult]


@dataclass(slots=True)
class CandidateProtocolScore:
    total_n: int
    interim_every: int
    power: float
    expected_effect: float
    safety_rate: float
    expected_sample_size: float = 0.0
    expected_cost: float = 0.0
    utility: float = 0.0


@dataclass(slots=True)
class ProtocolRecommendation:
    protocol: dict[str, Any]
    candidates: list[CandidateProtocolScore]


def default_covariates() -> list[CovariateSpec]:
    return [
        CovariateSpec(
            name="age",
            distribution="normal",
            params={"mean": 60.0, "sd": 10.0, "min": 18.0, "max": 85.0},
        ),
        CovariateSpec(
            name="weight",
            distribution="lognormal",
            params={"mean": 78.0, "cv": 0.25, "min": 45.0, "max": 140.0},
        ),
        CovariateSpec(
            name="egfr",
            distribution="normal",
            params={"mean": 85.0, "sd": 20.0, "min": 25.0, "max": 140.0},
        ),
        CovariateSpec(
            name="biomarker_z",
            distribution="normal",
            params={"mean": 0.0, "sd": 1.0, "min": -3.0, "max": 3.0},
        ),
    ]


def default_simulation_config() -> SimulationConfig:
    return SimulationConfig(
        trial_id="refua-clinical-demo",
        indication="Oncology",
        phase="Phase II",
        objective=(
            "Select a dose/regimen that maximizes efficacy while controlling safety risk, "
            "using copula-based virtual patients and adaptive randomization."
        ),
        population=VirtualPopulationSpec(
            size=6000,
            covariates=default_covariates(),
            correlation=[
                [1.0, 0.15, -0.25, 0.05],
                [0.15, 1.0, 0.10, 0.00],
                [-0.25, 0.10, 1.0, -0.10],
                [0.05, 0.00, -0.10, 1.0],
            ],
        ),
        arms=[
            ArmSpec(arm_id="control", label="Standard of Care", dose_mg=0.0, is_control=True),
            ArmSpec(arm_id="low", label="Investigational Low Dose", dose_mg=80.0),
            ArmSpec(arm_id="high", label="Investigational High Dose", dose_mg=140.0),
        ],
    )

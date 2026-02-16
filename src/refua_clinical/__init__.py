"""refua-clinical package API."""

from importlib.metadata import PackageNotFoundError, version

from .admet_integration import (
    apply_admet_adjustments,
    compute_admet_profile,
    load_admet_profile,
    summarize_admet_profile,
)
from .estimands import apply_estimand, arm_level_analysis, estimand_summary
from .explainability import (
    build_advice_report,
    render_advice_markdown,
    run_parameter_sensitivity,
)
from .models import (
    AdaptiveDesignSpec,
    ArmSpec,
    CandidateProtocolScore,
    CovariateSpec,
    EndpointSpec,
    EnrollmentSpec,
    EstimandSpec,
    ExternalControlSpec,
    HeterogeneitySpec,
    OperationalCostSpec,
    PDModelSpec,
    PKModelSpec,
    ProtocolRecommendation,
    ReplicateResult,
    SimulationConfig,
    StoppingSpec,
    TrialSimulationResult,
    VirtualPopulationSpec,
    default_covariates,
    default_simulation_config,
)
from .optimization import optimization_to_markdown, optimize_design_space
from .protocol import recommend_protocol, render_protocol_markdown
from .research import list_references
from .stopping import alpha_spent, evaluate_interim_decision
from .transportability import assess_transportability, load_tabular, transportability_to_markdown
from .trial import simulate_trials, summarize_simulation, trial_result_to_mapping
from .virtual_patients import (
    VirtualPopulation,
    generate_virtual_population,
    infer_population_spec_from_dataframe,
    summarize_covariates,
)
from .voi import estimate_value_of_information, voi_to_markdown

try:
    __version__ = version("refua-clinical")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "AdaptiveDesignSpec",
    "ArmSpec",
    "CandidateProtocolScore",
    "CovariateSpec",
    "EstimandSpec",
    "EndpointSpec",
    "EnrollmentSpec",
    "ExternalControlSpec",
    "HeterogeneitySpec",
    "OperationalCostSpec",
    "PKModelSpec",
    "PDModelSpec",
    "ProtocolRecommendation",
    "ReplicateResult",
    "SimulationConfig",
    "StoppingSpec",
    "TrialSimulationResult",
    "VirtualPopulation",
    "VirtualPopulationSpec",
    "__version__",
    "apply_admet_adjustments",
    "apply_estimand",
    "alpha_spent",
    "arm_level_analysis",
    "assess_transportability",
    "build_advice_report",
    "compute_admet_profile",
    "default_covariates",
    "default_simulation_config",
    "estimate_value_of_information",
    "estimand_summary",
    "evaluate_interim_decision",
    "generate_virtual_population",
    "infer_population_spec_from_dataframe",
    "load_tabular",
    "load_admet_profile",
    "list_references",
    "optimization_to_markdown",
    "optimize_design_space",
    "render_advice_markdown",
    "recommend_protocol",
    "run_parameter_sensitivity",
    "summarize_admet_profile",
    "render_protocol_markdown",
    "simulate_trials",
    "summarize_covariates",
    "summarize_simulation",
    "transportability_to_markdown",
    "trial_result_to_mapping",
    "voi_to_markdown",
]

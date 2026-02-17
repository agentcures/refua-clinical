"""refua-clinical public API.

This package now exposes an object-oriented API centered on ``ClinicalStudy``.
Legacy function-based top-level exports were removed.
"""

from importlib.metadata import PackageNotFoundError, version

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
)
from .object_api import (
    ClinicalAdvice,
    ClinicalOptimization,
    ClinicalProtocol,
    ClinicalRun,
    ClinicalStudy,
    ClinicalVOI,
    ClinicalWorkup,
)
from .refua_bridge import RefuaIntegrationPolicy
from .virtual_patients import VirtualPopulation

try:
    __version__ = version("refua-clinical")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = [
    "AdaptiveDesignSpec",
    "ArmSpec",
    "CandidateProtocolScore",
    "ClinicalAdvice",
    "ClinicalOptimization",
    "ClinicalProtocol",
    "ClinicalRun",
    "ClinicalStudy",
    "ClinicalVOI",
    "ClinicalWorkup",
    "CovariateSpec",
    "EndpointSpec",
    "EnrollmentSpec",
    "EstimandSpec",
    "ExternalControlSpec",
    "HeterogeneitySpec",
    "OperationalCostSpec",
    "PDModelSpec",
    "PKModelSpec",
    "ProtocolRecommendation",
    "RefuaIntegrationPolicy",
    "ReplicateResult",
    "SimulationConfig",
    "StoppingSpec",
    "TrialSimulationResult",
    "VirtualPopulation",
    "VirtualPopulationSpec",
    "__version__",
]

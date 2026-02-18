"""refua-clinical public API.

This package now exposes an object-oriented API centered on ``ClinicalStudy``.
Legacy function-based top-level exports were removed.
"""

import tomllib
from importlib.metadata import version as _distribution_version
from pathlib import Path

from .modality import apply_modality_preset, list_modality_presets
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
    ModalityKind,
    OperationalCostSpec,
    PDModelSpec,
    PKModelSpec,
    ProtocolRecommendation,
    ReplicateResult,
    RouteKind,
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


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not version:
        return None
    return str(version)


def _resolve_version() -> str:
    local_version = _read_version_from_pyproject()
    if local_version is not None:
        return local_version
    return _distribution_version("refua-clinical")


__version__ = _resolve_version()

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
    "ModalityKind",
    "OperationalCostSpec",
    "PDModelSpec",
    "PKModelSpec",
    "ProtocolRecommendation",
    "RefuaIntegrationPolicy",
    "ReplicateResult",
    "RouteKind",
    "SimulationConfig",
    "StoppingSpec",
    "TrialSimulationResult",
    "VirtualPopulation",
    "VirtualPopulationSpec",
    "apply_modality_preset",
    "list_modality_presets",
    "__version__",
]

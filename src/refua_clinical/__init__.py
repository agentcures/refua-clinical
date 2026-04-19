"""refua-clinical public API.

This package now exposes an object-oriented API centered on ``ClinicalStudy``.
Legacy function-based top-level exports were removed.
"""

from __future__ import annotations

import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
        TransportMethod,
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
    from .trial_management import ClinicalTrialManager, default_trial_store_path
    from .virtual_patients import VirtualPopulation
    from .modality import apply_modality_preset, list_modality_presets


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
    try:
        return _distribution_version("refua-clinical")
    except PackageNotFoundError:
        local_version = _read_version_from_pyproject()
        if local_version is not None:
            return local_version
        raise


__version__ = _resolve_version()

_LAZY_EXPORTS = {
    "AdaptiveDesignSpec": (".models", "AdaptiveDesignSpec"),
    "ArmSpec": (".models", "ArmSpec"),
    "CandidateProtocolScore": (".models", "CandidateProtocolScore"),
    "ClinicalAdvice": (".object_api", "ClinicalAdvice"),
    "ClinicalOptimization": (".object_api", "ClinicalOptimization"),
    "ClinicalProtocol": (".object_api", "ClinicalProtocol"),
    "ClinicalRun": (".object_api", "ClinicalRun"),
    "ClinicalStudy": (".object_api", "ClinicalStudy"),
    "ClinicalVOI": (".object_api", "ClinicalVOI"),
    "ClinicalWorkup": (".object_api", "ClinicalWorkup"),
    "ClinicalTrialManager": (".trial_management", "ClinicalTrialManager"),
    "CovariateSpec": (".models", "CovariateSpec"),
    "EndpointSpec": (".models", "EndpointSpec"),
    "EnrollmentSpec": (".models", "EnrollmentSpec"),
    "EstimandSpec": (".models", "EstimandSpec"),
    "ExternalControlSpec": (".models", "ExternalControlSpec"),
    "HeterogeneitySpec": (".models", "HeterogeneitySpec"),
    "ModalityKind": (".models", "ModalityKind"),
    "OperationalCostSpec": (".models", "OperationalCostSpec"),
    "PDModelSpec": (".models", "PDModelSpec"),
    "PKModelSpec": (".models", "PKModelSpec"),
    "ProtocolRecommendation": (".models", "ProtocolRecommendation"),
    "RefuaIntegrationPolicy": (".refua_bridge", "RefuaIntegrationPolicy"),
    "ReplicateResult": (".models", "ReplicateResult"),
    "RouteKind": (".models", "RouteKind"),
    "SimulationConfig": (".models", "SimulationConfig"),
    "StoppingSpec": (".models", "StoppingSpec"),
    "TransportMethod": (".models", "TransportMethod"),
    "TrialSimulationResult": (".models", "TrialSimulationResult"),
    "VirtualPopulation": (".virtual_patients", "VirtualPopulation"),
    "VirtualPopulationSpec": (".models", "VirtualPopulationSpec"),
    "apply_modality_preset": (".modality", "apply_modality_preset"),
    "default_trial_store_path": (".trial_management", "default_trial_store_path"),
    "list_modality_presets": (".modality", "list_modality_presets"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute = target
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS.keys()])

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
    "ClinicalTrialManager",
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
    "TransportMethod",
    "TrialSimulationResult",
    "VirtualPopulation",
    "VirtualPopulationSpec",
    "apply_modality_preset",
    "default_trial_store_path",
    "list_modality_presets",
    "__version__",
]

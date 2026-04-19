"""I/O and config parsing helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

import yaml

from .models import (
    AdaptiveDesignSpec,
    AlphaSpendingKind,
    ArmSpec,
    CovariateSpec,
    DistributionKind,
    EndpointKind,
    EndpointSpec,
    EnrollmentSpec,
    EstimandSpec,
    EstimandStrategy,
    ExternalControlSpec,
    HeterogeneitySpec,
    ModalityKind,
    OperationalCostSpec,
    PDModelSpec,
    PKModelSpec,
    RouteKind,
    SimulationConfig,
    StoppingSpec,
    TransportMethod,
    VirtualPopulationSpec,
)


def load_mapping(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        parsed = yaml.safe_load(raw)
        payload = {} if parsed is None else parsed

    if not isinstance(payload, dict):
        raise ValueError(f"Top-level document must be an object: {file_path}")
    return dict(payload)


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def config_to_mapping(config: SimulationConfig) -> dict[str, Any]:
    return asdict(config)


def clone_config(config: SimulationConfig) -> SimulationConfig:
    return copy.deepcopy(config)


def config_from_mapping(data: dict[str, Any]) -> SimulationConfig:
    population_raw = _mapping(data.get("population"))
    covariates_raw = population_raw.get("covariates", [])
    if not isinstance(covariates_raw, list) or not covariates_raw:
        raise ValueError("population.covariates must be a non-empty list")

    covariates: list[CovariateSpec] = []
    for raw in covariates_raw:
        item = _mapping(raw)
        covariates.append(
            CovariateSpec(
                name=_required_str(item, "name"),
                distribution=_parse_distribution_kind(
                    _required_str(item, "distribution")
                ),
                params=_mapping(item.get("params")),
            )
        )

    correlation_raw = population_raw.get("correlation")
    correlation: list[list[float]] | None
    if correlation_raw is None:
        correlation = None
    else:
        if not isinstance(correlation_raw, list):
            raise ValueError("population.correlation must be a 2D list")
        correlation = []
        for row in correlation_raw:
            if not isinstance(row, list):
                raise ValueError("population.correlation rows must be lists")
            correlation.append([float(value) for value in row])

    arms_raw = data.get("arms", [])
    if not isinstance(arms_raw, list) or not arms_raw:
        raise ValueError("arms must be a non-empty list")

    arms: list[ArmSpec] = []
    for raw in arms_raw:
        item = _mapping(raw)
        dosing_interval_raw = item.get("dosing_interval_hours")
        arms.append(
            ArmSpec(
                arm_id=_required_str(item, "arm_id"),
                label=str(item.get("label", item.get("arm_id"))),
                dose_mg=float(item.get("dose_mg", 0.0)),
                schedule_per_day=int(item.get("schedule_per_day", 1)),
                is_control=bool(item.get("is_control", False)),
                dosing_interval_hours=(
                    float(dosing_interval_raw)
                    if dosing_interval_raw is not None
                    else None
                ),
                opens_at_enrollment=int(item.get("opens_at_enrollment", 0)),
                opens_at_interim=(
                    int(item["opens_at_interim"])
                    if item.get("opens_at_interim") is not None
                    else None
                ),
                opens_after_arm_drop=(
                    str(item["opens_after_arm_drop"])
                    if item.get("opens_after_arm_drop") is not None
                    else None
                ),
                closes_at_interim=(
                    int(item["closes_at_interim"])
                    if item.get("closes_at_interim") is not None
                    else None
                ),
                max_patients=(
                    int(item["max_patients"])
                    if item.get("max_patients") is not None
                    else None
                ),
                titration_step_mg=float(item.get("titration_step_mg", 0.0)),
                titration_interval_days=(
                    int(item["titration_interval_days"])
                    if item.get("titration_interval_days") is not None
                    else None
                ),
                backfill_enabled=bool(item.get("backfill_enabled", False)),
                backfill_target_n=(
                    int(item["backfill_target_n"])
                    if item.get("backfill_target_n") is not None
                    else None
                ),
                backfill_allocation_multiplier=float(
                    item.get("backfill_allocation_multiplier", 1.0)
                ),
                concurrent_control_only=bool(item.get("concurrent_control_only", False)),
            )
        )

    control_arm_count = sum(1 for arm in arms if arm.is_control)
    if control_arm_count != 1:
        raise ValueError("Exactly one arm must set is_control=true")

    pk_raw = _mapping(data.get("pk_model"))
    pd_raw = _mapping(data.get("pd_model"))
    endpoint_raw = _mapping(data.get("endpoint"))
    enrollment_raw = _mapping(data.get("enrollment"))
    adaptive_raw = _mapping(data.get("adaptive"))
    external_control_raw = _mapping(data.get("external_control"))
    estimand_raw = _mapping(data.get("estimand"))
    stopping_raw = _mapping(data.get("stopping"))
    heterogeneity_raw = _mapping(data.get("heterogeneity"))
    costs_raw = _mapping(data.get("costs"))

    pk_values = _coerce_numeric_fields(pk_raw, PKModelSpec())
    pk_values["modality"] = _parse_modality_kind(
        str(pk_values.get("modality", "small_molecule"))
    )
    pk_values["route"] = _parse_route_kind(str(pk_values.get("route", "oral")))

    return SimulationConfig(
        trial_id=_required_str(data, "trial_id"),
        indication=_required_str(data, "indication"),
        phase=_required_str(data, "phase"),
        objective=_required_str(data, "objective"),
        seed=int(data.get("seed", 7)),
        replicates=int(data.get("replicates", 250)),
        population=VirtualPopulationSpec(
            size=int(population_raw.get("size", 6000)),
            covariates=covariates,
            correlation=correlation,
        ),
        pk_model=PKModelSpec(**pk_values),
        pd_model=PDModelSpec(**_coerce_numeric_fields(pd_raw, PDModelSpec())),
        endpoint=EndpointSpec(
            name=str(endpoint_raw.get("name", "change_from_baseline")),
            kind=_parse_endpoint_kind(str(endpoint_raw.get("kind", "continuous"))),
            assessment_day=int(endpoint_raw.get("assessment_day", 84)),
            visit_days=_parse_int_list(
                endpoint_raw.get("visit_days"),
                default=[28, 56, int(endpoint_raw.get("assessment_day", 84))],
            ),
            event_horizon_day=int(
                endpoint_raw.get("event_horizon_day", endpoint_raw.get("assessment_day", 168))
            ),
            responder_threshold=float(endpoint_raw.get("responder_threshold", 12.0)),
            target_difference=float(endpoint_raw.get("target_difference", 6.0)),
            target_hazard_ratio=float(endpoint_raw.get("target_hazard_ratio", 0.75)),
        ),
        enrollment=EnrollmentSpec(
            total_n=int(enrollment_raw.get("total_n", 180)),
            accrual_per_block=int(enrollment_raw.get("accrual_per_block", 30)),
            dropout_base=float(enrollment_raw.get("dropout_base", 0.08)),
            drift_per_block=float(enrollment_raw.get("drift_per_block", 0.60)),
        ),
        adaptive=AdaptiveDesignSpec(
            enabled=bool(adaptive_raw.get("enabled", True)),
            burn_in_n=int(adaptive_raw.get("burn_in_n", 60)),
            interim_every=int(adaptive_raw.get("interim_every", 30)),
            min_allocation=float(adaptive_raw.get("min_allocation", 0.15)),
            posterior_samples=int(adaptive_raw.get("posterior_samples", 600)),
            allow_arm_dropping=bool(adaptive_raw.get("allow_arm_dropping", True)),
            arm_drop_threshold=float(adaptive_raw.get("arm_drop_threshold", 0.05)),
        ),
        external_control=ExternalControlSpec(
            enabled=bool(external_control_raw.get("enabled", False)),
            n=int(external_control_raw.get("n", 0)),
            mean=float(external_control_raw.get("mean", 0.0)),
            std=float(external_control_raw.get("std", 1.0)),
            weight=float(external_control_raw.get("weight", 0.0)),
            dynamic_borrowing=bool(external_control_raw.get("dynamic_borrowing", True)),
            commensurability_scale=float(
                external_control_raw.get("commensurability_scale", 1.75)
            ),
            robust_mixture=float(external_control_raw.get("robust_mixture", 0.25)),
            transport_method=_parse_transport_method(
                str(external_control_raw.get("transport_method", "none"))
            ),
        ),
        estimand=EstimandSpec(
            strategy=_parse_estimand_strategy(
                str(estimand_raw.get("strategy", "treatment_policy"))
            ),
            rescue_penalty=float(estimand_raw.get("rescue_penalty", 4.0)),
            control_imputation_shift=float(
                estimand_raw.get("control_imputation_shift", 0.0)
            ),
        ),
        stopping=StoppingSpec(
            enabled=bool(stopping_raw.get("enabled", True)),
            one_sided_alpha=float(stopping_raw.get("one_sided_alpha", 0.025)),
            alpha_spending=_parse_alpha_spending(
                str(stopping_raw.get("alpha_spending", "obrien_fleming"))
            ),
            success_posterior_threshold=float(
                stopping_raw.get("success_posterior_threshold", 0.99)
            ),
            futility_posterior_threshold=float(
                stopping_raw.get("futility_posterior_threshold", 0.20)
            ),
            min_interim_n=int(stopping_raw.get("min_interim_n", 60)),
            max_interims=int(stopping_raw.get("max_interims", 6)),
        ),
        heterogeneity=HeterogeneitySpec(
            enabled=bool(heterogeneity_raw.get("enabled", False)),
            n_sites=int(heterogeneity_raw.get("n_sites", 25)),
            n_countries=int(heterogeneity_raw.get("n_countries", 6)),
            site_sd=float(heterogeneity_raw.get("site_sd", 1.25)),
            country_sd=float(heterogeneity_raw.get("country_sd", 0.80)),
            site_startup_delay_mean_days=float(
                heterogeneity_raw.get("site_startup_delay_mean_days", 0.0)
            ),
            site_startup_delay_sd_days=float(
                heterogeneity_raw.get("site_startup_delay_sd_days", 0.0)
            ),
        ),
        costs=OperationalCostSpec(
            cost_per_patient=float(costs_raw.get("cost_per_patient", 25_000.0)),
            cost_per_interim=float(costs_raw.get("cost_per_interim", 45_000.0)),
            utility_scale=float(costs_raw.get("utility_scale", 1.0)),
            cost_per_site_activation=float(
                costs_raw.get("cost_per_site_activation", 85_000.0)
            ),
            cost_per_study_month=float(costs_raw.get("cost_per_study_month", 30_000.0)),
        ),
        arms=arms,
    )


def merge_mappings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_mappings(_mapping(merged[key]), value)
        else:
            merged[key] = value
    return merged


def apply_set_overrides(data: dict[str, Any], values: list[str]) -> dict[str, Any]:
    updated = dict(data)
    for item in values:
        if "=" not in item:
            raise ValueError(
                f"Invalid --set value '{item}', expected dotted.path=value"
            )
        path, raw_value = item.split("=", 1)
        keys = [part for part in path.split(".") if part]
        if not keys:
            raise ValueError(f"Invalid --set path '{path}'")
        cursor: dict[str, Any] = updated
        for key in keys[:-1]:
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                existing = {}
                cursor[key] = existing
            cursor = existing
        cursor[keys[-1]] = _parse_inline_value(raw_value)
    return updated


def _parse_inline_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass

    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"Missing required key: {key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Key '{key}' must be non-empty")
    return text


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected mapping/object value")
    return dict(value)


def _coerce_numeric_fields(source: dict[str, Any], defaults: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_item in fields(defaults):
        key = field_item.name
        default_value = getattr(defaults, key)
        raw = source.get(key, default_value)
        if isinstance(default_value, bool):
            result[key] = bool(raw)
        elif isinstance(default_value, str):
            result[key] = str(raw)
        elif isinstance(default_value, int):
            result[key] = int(raw)
        elif default_value is None:
            result[key] = raw
        else:
            result[key] = float(raw)
    return result


def _parse_distribution_kind(raw: str) -> DistributionKind:
    normalized = raw.strip().lower()
    allowed = {"normal", "lognormal", "beta", "uniform", "categorical"}
    if normalized not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported covariate distribution '{raw}'. Allowed: {values}"
        )
    return cast(DistributionKind, normalized)


def _parse_endpoint_kind(raw: str) -> EndpointKind:
    normalized = raw.strip().lower()
    if normalized not in {"continuous", "binary", "time_to_event", "longitudinal"}:
        raise ValueError(
            "endpoint.kind must be one of "
            "'continuous', 'binary', 'time_to_event', or 'longitudinal'"
        )
    return cast(EndpointKind, normalized)


def _parse_modality_kind(raw: str) -> ModalityKind:
    normalized = raw.strip().lower()
    if normalized not in {"small_molecule", "biologic"}:
        raise ValueError("pk_model.modality must be 'small_molecule' or 'biologic'")
    return cast(ModalityKind, normalized)


def _parse_route_kind(raw: str) -> RouteKind:
    normalized = raw.strip().lower()
    if normalized not in {"oral", "iv", "sc"}:
        raise ValueError("pk_model.route must be 'oral', 'iv', or 'sc'")
    return cast(RouteKind, normalized)


def _parse_estimand_strategy(raw: str) -> EstimandStrategy:
    normalized = raw.strip().lower()
    allowed = {
        "treatment_policy",
        "hypothetical",
        "composite",
        "while_on_treatment",
    }
    if normalized not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(f"estimand.strategy must be one of: {values}")
    return cast(EstimandStrategy, normalized)


def _parse_alpha_spending(raw: str) -> AlphaSpendingKind:
    normalized = raw.strip().lower()
    allowed = {"obrien_fleming", "pocock", "linear"}
    if normalized not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(f"stopping.alpha_spending must be one of: {values}")
    return cast(AlphaSpendingKind, normalized)


def _parse_transport_method(raw: str) -> TransportMethod:
    normalized = raw.strip().lower()
    allowed = {"none", "ps_weighted", "entropy_balanced"}
    if normalized not in allowed:
        values = ", ".join(sorted(allowed))
        raise ValueError(f"external_control.transport_method must be one of: {values}")
    return cast(TransportMethod, normalized)


def _parse_int_list(value: Any, *, default: list[int]) -> list[int]:
    if value is None:
        return sorted(set(int(item) for item in default if int(item) > 0))
    if not isinstance(value, list):
        raise ValueError("endpoint.visit_days must be a list of integers")
    parsed = sorted(set(int(item) for item in value if int(item) > 0))
    if not parsed:
        raise ValueError("endpoint.visit_days must contain at least one positive day")
    return parsed
